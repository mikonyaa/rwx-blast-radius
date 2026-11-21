"""Replay a TPC fixture's input transport when only SQLGlot's encoder changed."""
import ast
from functools import lru_cache
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

SERDE = "sqlglot/serde.py"
PROFILE = Path(__file__).parent / "maps" / "serialization.json.gz"
PURE_CALLS = {"len", "type", "isinstance", "hasattr"}


def simple_read(expr):
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in PURE_CALLS:
                return False
        elif not isinstance(node, (ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Subscript,
                                   ast.Compare, ast.cmpop, ast.BoolOp, ast.boolop, ast.UnaryOp,
                                   ast.unaryop, ast.BinOp, ast.operator, ast.Tuple, ast.List)):
            return False
    return True


def payload_assignment(stmt):
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    target = stmt.targets[0]
    return (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            and target.value.id == "payload" and isinstance(target.slice, (ast.Name, ast.Constant))
            and simple_read(stmt.value))


@lru_cache(maxsize=1)
def profile():
    if PROFILE.is_file():
        return json.loads(gzip.decompress(PROFILE.read_bytes()))


def eligible(source, replacements):
    """Only local predicate/field edits in the pristine encoder, without external writes."""
    if set(replacements) != {SERDE}:
        return None
    before = (Path(source) / SERDE).read_text()
    after = replacements[SERDE]
    if not isinstance(after, str):
        return None
    old_rows, new_rows = before.splitlines(), after.splitlines()
    if len(old_rows) != len(new_rows):
        return None
    changed = [i for i, (a, b) in enumerate(zip(old_rows, new_rows), 1) if a != b]
    if not changed:
        return None
    for i in changed:
        a, b = old_rows[i - 1], new_rows[i - 1]
        if a[:len(a) - len(a.lstrip())] != b[:len(b) - len(b.lstrip())]:
            return None
    normalized = []
    for text, rows in ((before, old_rows), (after, new_rows)):
        tree = ast.parse(text)
        fn = next((f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "dump"), None)
        if not fn:
            return None
        safe = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and node.test.end_lineno == node.lineno and simple_read(node.test):
                if node.body[0].lineno > node.lineno and rows[node.lineno - 1].strip().endswith(":"):
                    safe.add(node.lineno)
                    if node.lineno in changed:
                        node.test = ast.Constant(None)
            elif payload_assignment(node) and node.lineno == node.end_lineno:
                safe.add(node.lineno)
                if node.lineno in changed:
                    node.targets[0].slice = ast.Constant(None)
                    node.value = ast.Constant(None)
        if not set(changed).issubset(safe):
            return None
        normalized.append(ast.dump(tree))
    if normalized[0] != normalized[1]:
        return None
    data = profile()
    if not data or any(str(i) in data["setup"] for i in changed):
        return None
    return {tag for tag in ("tpch", "tpcds")
            if not any(str(i) in data[tag]["other"] for i in changed)}


def same_inputs(source, fixtures):
    from sqlglot import parse_one
    from sqlglot import serde

    spec = importlib.util.spec_from_file_location("_rwx_pinned_encoder", Path(source) / SERDE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    for _, sql, _ in fixtures:
        expression = parse_one(sql)
        # The return of __reduce__ supplies (unchanged load, (dump(expression),)).
        # Compare pickle payloads, including primitive types and alias identity.
        # A mismatch or any exception keeps the test.
        expected = pickle.dumps(original.dump(expression), protocol=4)
        observed = pickle.dumps(serde.dump(expression), protocol=4)
        if observed != expected:
            return False
    return True


def build(source, destination):
    import query_probe
    root = Path(source).resolve()
    target = str(root / SERDE)
    sys.path.insert(0, str(root))
    os.chdir(root)
    data = {"setup": {}, **{tag: {"input": {}, "other": {}} for tag in ("tpch", "tpcds")}}
    rows = data["setup"]
    bit = 1

    def line_event(code, line):
        if code.co_filename == target:
            rows[line] = rows.get(line, 0) | bit
        return sys.monitoring.DISABLE

    tool = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tool, "fixture transport")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line_event)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)
    original_clone = query_probe.clone
    try:
        from tests import test_executor
        case = test_executor.TestExecutor()
        case.setUpClass()
        for tag in ("tpch", "tpcds"):
            rows, bit = data["setup"], 1
            sys.monitoring.restart_events()
            schema = getattr(test_executor, tag.upper() + "_SCHEMA")
            tables = getattr(case, tag + "_tables")
            test_executor.initializer(*original_clone((schema, tables)))

            def capture(obj):
                nonlocal rows
                if not (isinstance(obj, tuple) and len(obj) == 2 and obj[0] is test_executor.mp_execute):
                    return original_clone(obj)
                rows = data[tag]["input"]
                sys.monitoring.restart_events()
                try:
                    return original_clone(obj)
                finally:
                    rows = data[tag]["other"]
                    sys.monitoring.restart_events()

            query_probe.clone = capture
            for index, _ in enumerate(getattr(case, tag + "_sqls")):
                rows, bit = data[tag]["other"], 1 << index
                sys.monitoring.restart_events()
                query_probe.run_row(test_executor, case, tag, index)
        rows, bit = data["setup"], 1
        sys.monitoring.restart_events()
        case.tearDownClass()
    finally:
        query_probe.clone = original_clone
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.free_tool_id(tool)
    destination.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    build(sys.argv[1], Path(sys.argv[2]).resolve())
