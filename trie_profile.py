"""Record keys inserted into pristine SQLGlot tries, including import-time construction."""

import ast
import gzip
import json
import os
from pathlib import Path
import sys


def terminal(tree):
    func = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "new_trie"
        ),
        None,
    )
    if func:
        for node in func.body:
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.body
                and isinstance(node.body[-1], ast.Assign)
            ):
                return node


def guarded_keys(old, after):
    before_loop, after_loop = terminal(old), terminal(after)
    if not before_loop or after_loop:
        return None
    # Unwrap one side-effect-free guard around the terminal assignment, and
    # require full source AST equality after the transformation.
    func = next(
        (
            node
            for node in after.body
            if isinstance(node, ast.FunctionDef) and node.name == "new_trie"
        ),
        None,
    )
    if func is None:
        return None
    for node in func.body:
        if not (
            isinstance(node, ast.For)
            and node.body
            and isinstance(node.body[-1], ast.If)
        ):
            continue
        guard = node.body[-1]
        if guard.orelse or len(guard.body) != 1:
            continue
        test = guard.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.left, ast.Name)
            and test.left.id == before_loop.target.id
        ):
            continue
        rhs = test.comparators[0]
        if isinstance(test.ops[0], ast.NotEq):
            items = [rhs]
        elif isinstance(test.ops[0], ast.NotIn) and isinstance(
            rhs, (ast.Tuple, ast.List)
        ):
            items = rhs.elts
        else:
            continue
        if not all(isinstance(e, ast.Constant) and type(e.value) is str for e in items):
            continue
        node.body[-1] = guard.body[0]
        if ast.dump(after) == ast.dump(old):
            return {e.value for e in items}
    return None


class TrieProfile:
    def __init__(self, directory, catalog):
        self.strings = None
        profile = Path(directory) / "trie_keys.json.gz"
        if profile.is_file():
            data = json.loads(gzip.decompress(profile.read_bytes()))
            if not data["unknown_type"]:
                self.strings = set(data["keys"])
                verified = set(data["verified"])
                self.uncertain = {
                    r["number"] for r in catalog if r["id"] not in verified
                }

    def candidates(self, tree, modified):
        if self.strings is None:
            return None
        keys = guarded_keys(tree, ast.parse(modified))
        if keys is not None and keys.isdisjoint(self.strings):
            return self.uncertain.copy()
        return None


def build(source, catalog, output):
    import pytest
    from coverage_build import Collector

    source = source.resolve()
    output = output.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    path = str(source / "sqlglot/trie.py")
    loop = terminal(ast.parse(Path(path).read_bytes()))
    event_line = loop.body[-1].lineno
    strings = set()
    unknown_type = False
    outside = 0

    class Records(Collector):
        def set_context(self, mask, nodeid=None):
            self.mask = mask

    trace = Records(source, nodes)

    def audit(event, args):
        nonlocal outside
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
            outside |= trace.mask

    def line(code, lineno):
        nonlocal unknown_type
        if lineno != event_line:
            return sys.monitoring.DISABLE
        key = sys._getframe(1).f_locals[loop.target.id]
        if type(key) is str:
            strings.add(key)
        elif type(key) not in (tuple, list):
            unknown_type = True

    tool = sys.monitoring.COVERAGE_ID

    def start(code, offset):
        if code.co_filename == path and code.co_name == "new_trie":
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.LINE)
        return sys.monitoring.DISABLE

    sys.addaudithook(audit)
    sys.monitoring.use_tool_id(tool, "pristine trie keys")
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, start)
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line)
    sys.monitoring.set_events(tool, sys.monitoring.events.PY_START)
    sys.path.insert(0, str(source))
    os.chdir(source)
    try:
        from sqlglot import Dialect

        assert Dialect.classes
        status = pytest.main(
            ["-q", "--tb=short", "-p", "no:cacheprovider", "tests"], plugins=[trace]
        )
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.free_tool_id(tool)
    if not trace.passed or status not in (0, 1) or not strings:
        raise RuntimeError("No usable trie trace")
    verified = [
        node
        for i, node in enumerate(nodes)
        if i in trace.passed and i not in trace.failed and not outside & (1 << i)
    ]
    report = {
        "verified": verified,
        "unknown_type": unknown_type,
        "keys": sorted(strings),
    }
    output.write_bytes(
        gzip.compress(json.dumps(report, separators=(",", ":")).encode(), mtime=0)
    )
    print(
        "distinct strings",
        len(strings),
        "verified tests",
        len(verified),
        "unknown types",
        unknown_type,
    )


if __name__ == "__main__":
    build(*(Path(s) for s in sys.argv[1:]))
