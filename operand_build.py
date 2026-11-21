"""Profile comparison operands on the unmodified SQLGlot test suite (offline)."""
import argparse
import ast
import builtins
import enum
import gzip
import importlib.abc
import importlib.machinery
import json
import operator
import os
from pathlib import Path
from coverage_build import LocalExecutor
import sys

OPS = {"Eq": operator.eq, "NotEq": operator.ne, "Is": operator.is_,
       "IsNot": operator.is_not, "Lt": operator.lt, "LtE": operator.le,
       "Gt": operator.gt, "GtE": operator.ge}
locations = []
observations = []
current = -1
complete = set()
unsafe = set()


def relation(x, y):
    same = x is y
    a, b = type(x), type(y)
    # Compare only built-in, side-effect-free values here. Unknown rich comparisons
    # will always cause the selector to keep the test.
    if x is None and y is None:
        return (1, 1, 2)
    if x is None or y is None:
        other = b if x is None else a
        if other.__eq__ is object.__eq__ or other in (int, float, str, bool, bytes, list, dict, tuple, set):
            return (int(same), int(same), 2)
    if a is b and issubclass(a, enum.Enum):
        if a.__eq__ in (object.__eq__, str.__eq__, int.__eq__):
            return (int(same), int(same), 2)
    if (a in (int, bool, str, bytes) and a is b) or (
            a is tuple and b is tuple and all(type(v) is int for v in x+y)):
        return (int(same), int(x == y), (x > y) - (x < y))
    return (int(same), 2, 2)


def observe(site, op, x, y):
    outcome = OPS[op](x, y)
    observations[site].setdefault(relation(x, y), set()).add(current)
    return outcome


class Instrument(ast.NodeTransformer):
    def __init__(self, path):
        self.path = path

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        op = type(node.ops[0]).__name__
        rhs = node.comparators[0]
        if op not in OPS or not (
            isinstance(rhs, ast.Attribute) and rhs.attr.isupper() or
            isinstance(rhs, ast.Constant) and (rhs.value is None or type(rhs.value) is int) or
            isinstance(rhs, ast.Tuple) and all(isinstance(v, ast.Constant) and type(v.value) is int for v in rhs.elts)
        ):
            return node
        site = len(locations)
        locations.append((self.path, f"{node.lineno}:{node.col_offset}"))
        observations.append({})
        call = ast.Call(ast.Name("_rwx_operand", ast.Load()),
                        [ast.Constant(site), ast.Constant(op), node.left, rhs], [])
        return ast.copy_location(call, node)


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = Path(self.path)
        tree = ast.parse(path.read_bytes(), self.path)
        relative = str(path.relative_to(root))
        return compile(ast.fix_missing_locations(Instrument(relative).visit(tree)), self.path, "exec")


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith(".py"):
            spec.loader = Loader(fullname, spec.origin)
        return spec


if __name__ == "__main__":
    import pytest
    import concurrent.futures
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("tests", type=Path)
    ap.add_argument("output", type=Path)
    options = ap.parse_args()
    root = options.root.resolve()
    output = options.output.resolve()
    records = json.loads(options.tests.read_text())["tests"]
    ids = [r["id"] for r in records]
    numbers = {s: i for i, s in enumerate(ids)}
    class Tracker:
        def pytest_collection_modifyitems(self, items):
            items[:] = [i for i in items if i.nodeid in numbers]

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            global current
            current = numbers[item.nodeid]
            try:
                yield
            finally:
                current = -1

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                complete.add(numbers[report.nodeid])
            if report.failed:
                unsafe.add(numbers[report.nodeid])

    output.parent.mkdir(exist_ok=True, parents=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_operand = observe
    concurrent.futures.ProcessPoolExecutor = LocalExecutor
    def audit(event, args):
        if event == "subprocess.Popen" and current >= 0:
            unsafe.add(current)
    sys.addaudithook(audit)
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    pytest.main(["-q", "--rootdir=.", "-p", "no:cacheprovider", "tests"], plugins=[Tracker()])
    complete.difference_update(unsafe)
    files = {}
    for (path, spot), states in zip(locations, observations):
        # Reloaded modules can contribute multiple observations at one site.
        entry = files.setdefault(path, {}).setdefault(spot, {})
        for state, group in states.items():
            entry.setdefault(",".join(map(str, state)), set()).update(group)
    for sites in files.values():
        for pos, states in sites.items():
            sites[pos] = {key: sorted(v) for key, v in states.items()}
    with gzip.open(output, "wt", compresslevel=6) as f:
        json.dump({"ids": ids, "complete": sorted(complete), "files": files}, f, separators=(",", ":"))
    print(f"{len(complete)} complete tests, map {output.stat().st_size} bytes")
