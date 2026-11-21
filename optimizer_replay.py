"""Capture fixture rule inputs from the pinned TPC-DS executor test."""

from collections import defaultdict
from functools import lru_cache, wraps
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "maps" / "optimizer_fixtures.zip"
TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


def node_state(node):
    # SQLGlot.dump preserves child nodes but distinguishes neither a missing
    # argument vs an empty one nor absent metadata vs an empty container.
    layout = tuple(
        (key, 0 if value is None else 1 if type(value) is list and not value else 2)
        for key, value in node.args.items()
    )
    return (
        node._hash is not None,
        node.comments is None,
        node._meta is None,
        node._type is node,
        layout,
    )


def fingerprint(expression):
    """Include argument order, parent/index invariants, and cached hash validity."""
    from sqlglot import exp

    if not isinstance(expression, exp.Expr) or expression.parent is not None:
        return None
    states = []
    known = set()
    for node in expression.walk():
        if id(node) in known:
            return None
        known.add(id(node))
        if getattr(node, "__dict__", None):
            return None
        states.append(node_state(node))
        for k, value in node.args.items():
            entries = enumerate(value) if isinstance(value, list) else [(None, value)]
            for i, child in entries:
                if isinstance(child, exp.Expr) and (
                    child.parent is not node or child.arg_key != k or child.index != i
                ):
                    return None
    dump = expression.dump()
    fresh = exp.Expr.load(dump)
    for node, other in zip(expression.walk(), fresh.walk(), strict=True):
        if node._hash is not None and node._hash != hash(other):
            return None
    return hashlib.sha256(
        json.dumps([dump, states], separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def serialize(expression, kwargs):
    """SQLGlot's public pickle transport omits caches and empty metadata."""
    return pickle.dumps(
        (
            expression,
            kwargs,
            [node_state(n) for n in expression.walk()],
        ),
        protocol=4,
    )


def deserialize(payload):
    expression, kwargs, states = pickle.loads(payload)
    # Calculate all requested hashes before removing the child cache entries
    # that were absent in the original state.
    nodes = list(expression.walk())
    for current, (_, absent_comments, absent_meta, self_type, layout) in zip(
        nodes, states, strict=True
    ):
        current.args = {
            key: None if kind == 0 else [] if kind == 1 else current.args[key]
            for key, kind in layout
        }
        if self_type:
            current._type = current
        if not absent_comments and current.comments is None:
            current.comments = []
        if not absent_meta and current._meta is None:
            current._meta = {}
    for current, state in zip(nodes, states):
        if state[0]:
            hash(current)
    for current, state in zip(nodes, states):
        if not state[0]:
            current._hash = None
    return expression, kwargs


def result_signature(result, kwargs):
    expr = fingerprint(result)
    if expr is None:
        return None
    return [expr, hashlib.sha256(pickle.dumps(kwargs, protocol=4)).hexdigest()]


def build(source, output):
    from query_probe import clone, run_row

    root = source.resolve()
    rel = str(root) + "/"
    prefix = rel + "sqlglot/optimizer/"
    sys.path.insert(0, str(root))
    os.chdir(root)
    lines = defaultdict(dict)
    queries = {}
    rule_id = None
    row = None
    names = []

    def on_line(code, line):
        if code.co_filename.startswith(prefix):
            sites = lines[code.co_filename[len(rel) :]].setdefault(str(line), {})
            stage = rule_id or "other"
            sites[stage] = sites.get(stage, 0) | (1 << (row or 0))
        return sys.monitoring.DISABLE

    tool = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tool, "TPC optimizer rule fixtures")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, on_line)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)
    try:
        from sqlglot import executor
        from sqlglot.optimizer.optimizer import RULES
        from tests import test_executor

        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:

            def wrap(rule):
                name = rule.__name__
                names.append(name)

                @wraps(rule)
                def wrapper(expr, **kwargs):
                    nonlocal rule_id
                    key = f"{row}:{name}"
                    input_sig = result_signature(expr, kwargs)
                    frozen = serialize(expr, kwargs) if input_sig else None
                    rule_id = name
                    sys.monitoring.restart_events()
                    try:
                        result = rule(expr, **kwargs)
                    finally:
                        rule_id = None
                        sys.monitoring.restart_events()
                    expected = result_signature(result, kwargs)
                    if expected is not None and frozen is not None:
                        archive.writestr(key, frozen)
                        queries[key] = (input_sig, expected)
                    return result

                wrapper.__signature__ = inspect.signature(rule)
                return wrapper

            rules = [wrap(fn) for fn in RULES]
            original = executor.optimize

            def optimize(*args, **kwargs):
                return original(*args, rules=rules, **kwargs)

            executor.optimize = optimize
            case = test_executor.TestExecutor()
            case.setUpClass()
            test_executor.initializer(
                *clone((test_executor.TPCDS_SCHEMA, case.tpcds_tables))
            )
            try:
                for row in range(len(case.tpcds_sqls)):
                    sys.monitoring.restart_events()
                    run_row(test_executor, case, "tpcds", row)
            finally:
                executor.optimize = original
                case.tearDownClass()
            sys.monitoring.set_events(tool, 0)
            by_name = {rule.__name__: rule for rule in RULES}
            for key, (input_sig, expected) in list(queries.items()):
                try:
                    replica, kwargs = deserialize(archive.read(key))
                    same = result_signature(replica, kwargs) == input_sig
                    start = time.process_time()
                    result = by_name[key.split(":", 1)[1]](replica, **kwargs)
                    same = same and result_signature(result, kwargs) == expected
                except (ValueError, TypeError, KeyError, AttributeError):
                    same = False
                if same:
                    queries[key] = [expected, round(time.process_time() - start, 5)]
                else:
                    del queries[key]
            info = {"sites": lines, "rows": queries, "rules": names}
            archive.writestr("index", json.dumps(info, separators=(",", ":")))
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.free_tool_id(tool)


@lru_cache(maxsize=1)
def profile():
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            return json.loads(archive.read("index"))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError):
        return None


def eligible(root, paths):
    """Accept only expression-level scalar and comparison substitutions."""
    from query_probe import _changed_lines

    if len(paths) != 1:
        return None
    path, content = next(iter(paths.items()))
    if not path.startswith("sqlglot/optimizer/") or not isinstance(content, str):
        return None
    original = (Path(root) / path).read_text()
    old, new = ast.parse(original), ast.parse(content)

    def normalize(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) in (bool, int, str):
                node.value = type(node.value)()
            if isinstance(node, ast.Compare):
                node.ops = [
                    ast.Eq()
                    if isinstance(
                        op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                    )
                    else op
                    for op in node.ops
                ]
        return ast.dump(tree)

    if normalize(old) != normalize(new):
        return None
    return _changed_lines(root, paths)


def evaluate(keys, number, notify):
    from sqlglot.optimizer.optimizer import RULES

    rules = {r.__name__: r for r in RULES}
    data = profile()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for key in keys:
            expected, _ = data["rows"][key]
            expression, kwargs = deserialize(archive.read(key))
            result = rules[key.split(":", 1)[1]](expression, **kwargs)
            if result_signature(result, kwargs) != expected:
                return
    os.write(notify, f"{number}\n".encode())


def checkpoints(root, paths):
    lines = eligible(root, paths)
    if not lines:
        return None
    data = profile()
    if not data:
        return None
    keys = set()
    for path, line in lines:
        for stage, mask in data["sites"].get(path, {}).get(line, {}).items():
            if stage == "other":
                return None
            keys.update(
                f"{i}:{stage}" for i in range(mask.bit_length()) if mask >> i & 1
            )
    if not keys or any(k not in data["rows"] for k in keys):
        return None
    estimate = sum(data["rows"][k][1] for k in keys)
    return sorted(
        keys, key=lambda k: (int(k.split(":")[0]), data["rules"].index(k.split(":")[1]))
    ), estimate


def refine(root, paths, chosen, tests):
    from probe import _overlay, _schedule_cost, isolated_numbers
    import tempfile

    n = next((i for i in chosen if tests[i]["id"] == TEST), None)
    if n is None or not paths:
        return chosen
    plan = checkpoints(root, paths)
    if not plan:
        return chosen
    keys, estimate = plan
    seconds = {i: tests[i]["seconds"] for i in chosen}
    saved = _schedule_cost(chosen, seconds) - _schedule_cost(chosen - {n}, seconds)
    if estimate > 3.5 or saved < max(1.5, estimate * 2.5 + 0.4):
        return chosen
    with tempfile.TemporaryDirectory(prefix="rwx-rules-") as temp:
        view = Path(temp)
        _overlay(Path(root).resolve(), view, paths)
        if n in isolated_numbers(
            view, evaluate, (keys, n), 7.2, min(4.5, 0.70 + estimate * 2.8), base=root
        ):
            return chosen - {n}
    return chosen


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
