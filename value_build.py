"""Observe builtin comparison operands, integer operators and indexed reads on pristine SQLGlot."""

import ast
import builtins
from collections import defaultdict
import gzip
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import tracemalloc

import boolean_values
from value_profile import (
    ARITHMETIC,
    COMPARISONS,
    INDICES,
    UNKNOWN,
    encode,
    literal,
    pair_relation,
    pos,
)


def build(root, catalog, destination):
    import pytest
    from coverage_build import Collector

    os.chdir(root)
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    collector = Collector(root, nodes)
    locations = defaultdict(dict)
    sites = []
    guard_leaves = {}
    unknown = 0

    def observing():
        nonlocal unknown
        # Measurement allocations can be visible to a memory assertion. Leave
        # that test unclassified and perform its original operations directly.
        if collector.mask and tracemalloc.is_tracing():
            unknown |= collector.mask
        return not (collector.mask & unknown)

    def site(path, node, kind):
        row = locations[path].setdefault(pos(node, kind), {})
        sites.append(row)
        return len(sites) - 1

    def add(index, state):
        row = sites[index]
        if len(row) > 48 and state not in row:
            state = "?"
        row[state] = row.get(state, 0) | (collector.mask or -1)

    def guard(index):
        if not observing():
            return True
        add(index, boolean_values.snapshot(guard_leaves[index], sys._getframe(1)))
        return True

    def comparison(rel, left, right, op, x, y):
        if not observing():
            return COMPARISONS[op](x, y)
        add(rel, pair_relation(x, y))
        if left >= 0:
            add(left, encode(x))
        if right >= 0:
            add(right, encode(y))
        return COMPARISONS[op](x, y)

    def arithmetic(index, op, x, y):
        if not observing():
            return ARITHMETIC[op](x, y)
        if (
            type(x) in (bool, int)
            and type(y) in (bool, int)
            and x.bit_length() <= 31
            and y.bit_length() <= 31
        ):
            add(index, encode(x) + ":" + encode(y))
        else:
            add(index, "?")
        return ARITHMETIC[op](x, y)

    def subscript(index, obj, key):
        if not observing():
            return obj[key]
        state = "?"
        if type(obj) in (list, tuple, str, bytes) and -len(obj) <= key < len(obj):
            result = obj[key]
            mask = 0
            for i, other in enumerate(INDICES):
                if -len(obj) <= other < len(obj) and obj[other] is result:
                    mask |= 1 << i
            state = format(mask, "x")
        add(index, state)
        return obj[key]

    class Transform(ast.NodeTransformer):
        def __init__(self, path):
            self.path = path
            self.enabled = False

        def visit_FunctionDef(self, node):
            old = self.enabled
            self.enabled = True
            node.body = [self.visit(x) for x in node.body]
            self.enabled = old
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            old = self.enabled
            self.enabled = False
            node = self.generic_visit(node)
            self.enabled = old
            return node

        def visit_Lambda(self, node):
            old = self.enabled
            self.enabled = True
            node.body = self.visit(node.body)
            self.enabled = old
            return node

        def visit_If(self, node):
            original = node.test
            desc = boolean_values.describe(original) if self.enabled else None
            node = self.generic_visit(node)
            if desc:
                sid = site(self.path, original, "guard")
                guard_leaves[sid] = desc[1]
                extra = ast.Call(
                    ast.Name("_rwx_value_guard", ast.Load()), [ast.Constant(sid)], []
                )
                node.test = ast.copy_location(
                    ast.BoolOp(ast.And(), [extra, node.test]), original
                )
            return node

        visit_While = visit_If
        visit_IfExp = visit_If

        def visit_Compare(self, node):
            if not self.enabled or len(node.ops) != 1:
                return self.generic_visit(node)
            op = type(node.ops[0]).__name__
            if op not in COMPARISONS:
                return self.generic_visit(node)
            a, b = node.left, node.comparators[0]
            rel = site(self.path, node, "rel")
            lhs = site(self.path, node, "lhs") if literal(b) is not UNKNOWN else -1
            rhs = site(self.path, node, "rhs") if literal(a) is not UNKNOWN else -1
            expression = ast.Call(
                ast.Name("_rwx_value_compare", ast.Load()),
                [
                    *(ast.Constant(i) for i in (rel, lhs, rhs, op)),
                    self.visit(a),
                    self.visit(b),
                ],
                [],
            )
            return ast.copy_location(expression, node)

        def visit_BinOp(self, node):
            op = type(node.op).__name__
            if not self.enabled or op not in ARITHMETIC:
                return self.generic_visit(node)
            sid = site(self.path, node, "bin")
            return ast.copy_location(
                ast.Call(
                    ast.Name("_rwx_value_binary", ast.Load()),
                    [
                        ast.Constant(sid),
                        ast.Constant(op),
                        self.visit(node.left),
                        self.visit(node.right),
                    ],
                    [],
                ),
                node,
            )

        def visit_Subscript(self, node):
            idx = literal(node.slice)
            if (
                not self.enabled
                or type(idx) is not int
                or idx not in INDICES
                or not isinstance(node.ctx, ast.Load)
            ):
                return self.generic_visit(node)
            sid = site(self.path, node, "sub")
            return ast.copy_location(
                ast.Call(
                    ast.Name("_rwx_value_subscript", ast.Load()),
                    [ast.Constant(sid), self.visit(node.value), node.slice],
                    [],
                ),
                node,
            )

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = Path(self.path)
            tree = Transform(path.relative_to(root).as_posix()).visit(
                ast.parse(path.read_bytes(), self.path)
            )
            return compile(
                ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True
            )

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    def audit(event, args):
        nonlocal unknown
        if event in {"subprocess.Popen", "os.posix_spawn", "os.fork", "os.system"}:
            unknown |= collector.mask

    sys.meta_path.insert(0, Finder())
    sys.path.insert(0, str(root))
    sys.addaudithook(audit)
    builtins._rwx_value_guard = guard
    builtins._rwx_value_compare = comparison
    builtins._rwx_value_binary = arithmetic
    builtins._rwx_value_subscript = subscript
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    status = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--tb=short", "tests"], plugins=[collector]
    )
    if status not in (0, 1) or not collector.passed:
        raise RuntimeError(status)
    for rows in locations.values():
        for row in rows.values():
            for state, mask in list(row.items()):
                row[state] = "*" if mask < 0 else format(mask, "x")
    confirmed = [
        i for i in sorted(collector.passed - collector.failed) if not unknown & (1 << i)
    ]
    data = {"nodes": nodes, "complete": confirmed, "rows": locations}
    destination.write_bytes(
        gzip.compress(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
    )
    print(
        "Finished",
        len(confirmed),
        "fully observed tests,",
        len(sites),
        "expression profiles",
    )


if __name__ == "__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
