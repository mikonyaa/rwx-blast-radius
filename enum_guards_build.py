"""Gather observed token values for explicit token equality and membership tests."""

import ast
import builtins
from collections import defaultdict
import concurrent.futures
import gzip
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import pytest
from coverage_build import Collector, LocalExecutor
from enum_guards import guard, match_argument, site
from functools import lru_cache
import inspect
from types import MethodType, MemberDescriptorType


def build(source, catalog, output, reverse=False, matches=False):
    root, dest = source.resolve(), output.resolve()
    nodes = [row["id"] for row in json.loads(catalog.read_bytes())["tests"]]
    tracker = Collector(root, nodes, reverse=reverse)
    files = defaultdict(dict)
    sites = []
    depth = 0

    class Rewrite(ast.NodeTransformer):
        def visit_Compare(self, node):
            names = not matches and guard(node)
            self.generic_visit(node)
            if names:
                row = files[self.path].setdefault(site(node), {})
                number = len(sites)
                sites.append(row)
                fn = ast.Name("_rwx_observe_token_guard", ast.Load())
                node.left = ast.copy_location(
                    ast.Call(
                        fn,
                        [
                            ast.Constant(number),
                            node.left,
                            ast.Name("TokenType", ast.Load()),
                        ],
                        [],
                    ),
                    node.left,
                )
            return node

        def visit_Call(self, node):
            names = matches and match_argument(node)
            self.generic_visit(node)
            if names:
                row = files[self.path].setdefault(site(node), {})
                number = len(sites)
                sites.append(row)
                fn = ast.Name("_rwx_observe_match", ast.Load())
                return ast.copy_location(
                    ast.Call(
                        fn, [ast.Constant(number), node.func, *node.args], node.keywords
                    ),
                    node,
                )
            return node

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, name):
            tree = ast.parse(Path(self.path).read_bytes(), self.path)
            if not name.startswith("sqlglot."):
                return compile(tree, self.path, "exec")
            visitor = Rewrite()
            visitor.path = str(Path(self.path).relative_to(root))
            return compile(
                ast.fix_missing_locations(visitor.visit(tree)), self.path, "exec"
            )

        def exec_module(self, module):
            nonlocal depth
            depth += 1
            try:
                super().exec_module(module)
            finally:
                depth -= 1

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".", 1)[0] not in {"sqlglot", "tests"}:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if (
                spec
                and spec.origin
                and spec.origin.endswith(".py")
                and str(spec.origin).startswith(str(root) + os.sep)
            ):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    enum = None

    def observe(ix, value, cls):
        nonlocal enum
        if enum is None:
            mod = sys.modules.get("sqlglot.tokenizer_core")
            enum = vars(mod).get("TokenType") if mod else None
        row = sites[ix]
        key = (
            value.name
            if enum is not None and cls is enum and type(value) is cls
            else "?"
        )
        mask = tracker.mask if tracker.mask and not depth else -1
        row[key] = row.get(key, 0) | mask
        return value

    parser_path = str(root / "sqlglot/parser.py")
    source_tree = ast.parse(Path(parser_path).read_bytes())
    target = next(
        m
        for c in source_tree.body
        if isinstance(c, ast.ClassDef) and c.name == "Parser"
        for m in c.body
        if isinstance(m, ast.FunctionDef) and m.name == "_match"
    )
    expected = ast.parse("""if self._curr.token_type == token_type:
    if advance:
        self._advance()
    self._add_comments(expression)
    return True
return False""")
    if [ast.dump(v) for v in target.body] != [ast.dump(v) for v in expected.body]:
        raise ValueError("Unrecognized parser matcher body")
    token_cls = None

    @lru_cache(maxsize=None)
    def cursor(cls):
        item = inspect.getattr_static(cls, "_curr", None)
        if (
            isinstance(item, MemberDescriptorType)
            and cls.__getattribute__ is object.__getattribute__
        ):
            return item
        return None

    def observe_match(ix, function, token_type, *rest, **kw):
        nonlocal enum, token_cls
        if enum is None:
            mod = sys.modules.get("sqlglot.tokenizer_core")
            if mod:
                enum, token_cls = vars(mod).get("TokenType"), vars(mod).get("Token")
        row = sites[ix]
        key = "?"
        if (
            enum
            and token_cls
            and type(function) is MethodType
            and type(token_type) is enum
            and function.__func__.__code__.co_filename == parser_path
            and function.__func__.__code__.co_firstlineno == target.lineno
            and function.__func__.__qualname__ == "Parser._match"
        ):
            instance = function.__self__
            slot = cursor(type(instance))
            if slot:
                try:
                    token = slot.__get__(instance, type(instance))
                except AttributeError:
                    pass
                else:
                    if type(token) is token_cls and type(token.token_type) is enum:
                        key = token.token_type.name
        mask = tracker.mask if tracker.mask and not depth else -1
        row[key] = row.get(key, 0) | mask
        return function(token_type, *rest, **kw)

    def audit(event, args):
        if tracker.mask and event in {
            "os.fork",
            "os.exec",
            "os.posix_spawn",
            "subprocess.Popen",
            "os.system",
        }:
            tracker.failed.update(
                i for i in range(len(nodes)) if tracker.mask & (1 << i)
            )

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_observe_token_guard = observe
    builtins._rwx_observe_match = observe_match
    concurrent.futures.ProcessPoolExecutor = LocalExecutor
    sys.addaudithook(audit)
    status = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[tracker]
    )
    if status not in (0, 1) or not tracker.passed or enum is None:
        raise ValueError("Incomplete profile")
    members = [n for n, v in enum.__members__.items() if n == v.name]
    rows = {
        f: {
            pos: {k: format(v, "x") for k, v in row.items()} for pos, row in tab.items()
        }
        for f, tab in files.items()
    }
    data = {
        "nodes": nodes,
        "complete": sorted(tracker.passed - tracker.failed),
        "members": members,
        "files": rows,
    }
    dest.write_bytes(
        gzip.compress(
            json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0
        )
    )


def merge(first, second, destination):
    a, b = (json.loads(gzip.decompress(path.read_bytes())) for path in (first, second))
    if a["nodes"] != b["nodes"] or a["members"] != b["members"]:
        raise ValueError("Baseline profiles use different test or token catalogs")
    a["complete"] = sorted(set(a["complete"]) & set(b["complete"]))
    files = {}
    for path in a["files"].keys() & b["files"].keys():
        first_rows, second_rows = a["files"][path], b["files"][path]
        rows = {}
        for pos in first_rows.keys() & second_rows.keys():
            left, right = first_rows[pos], second_rows[pos]
            rows[pos] = {
                name: format(
                    int(left.get(name, "0"), 16) | int(right.get(name, "0"), 16), "x"
                )
                for name in left.keys() | right.keys()
            }
        files[path] = rows
    a["files"] = files
    destination.write_bytes(
        gzip.compress(
            json.dumps(a, separators=(",", ":"), sort_keys=True).encode(), mtime=0
        )
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        merge(*(Path(v) for v in sys.argv[2:5]))
    else:
        build(
            *(Path(v) for v in sys.argv[1:4]),
            reverse="--reverse" in sys.argv[4:],
            matches="--matcher" in sys.argv[4:],
        )
