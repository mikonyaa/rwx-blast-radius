"""Build lookup-key observations from the passing pristine test suite."""
import ast
import builtins
from collections import defaultdict
import concurrent.futures
from functools import lru_cache
import gzip
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
from types import BuiltinMethodType, MethodType, MemberDescriptorType

from coverage_build import Collector, LocalExecutor
from key_reads import lookup, site_key


def build(source, catalog, destination, reverse):
    import pytest
    root = source.resolve()
    dest = destination.resolve()
    tests = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    tracker = Collector(root, tests, reverse=reverse)
    files = defaultdict(dict)
    sites = []
    importing = 0

    class Instrument(ast.NodeTransformer):
        def __init__(self, path):
            self.path = path

        def visit_Call(self, node):
            self.generic_visit(node)
            if not lookup(node, self.path):
                return node
            row = files[self.path].setdefault(site_key(node), {})
            ix = len(sites)
            sites.append((node.func.attr, row))
            result = ast.Call(ast.Name("_rwx_key_read", ast.Load()),
                              [ast.Constant(ix), node.func, *node.args], [])
            return ast.copy_location(result, node)

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, name):
            rel = str(Path(self.path).relative_to(root))
            tree = ast.parse(Path(self.path).read_bytes(), self.path)
            tree = Instrument(rel).visit(tree)
            return compile(ast.fix_missing_locations(tree), self.path, "exec")

        def exec_module(self, module):
            nonlocal importing
            importing += 1
            try:
                super().exec_module(module)
            finally:
                importing -= 1

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    @lru_cache(maxsize=None)
    def storage(cls):
        from sqlglot.expressions.core import Expression
        field = inspect.getattr_static(cls, "_meta", None)
        if (isinstance(field, MemberDescriptorType)
                and field is inspect.getattr_static(Expression, "_meta")
                and cls.__getattribute__ is object.__getattribute__):
            return field
        return None

    pinned_method = next(part for cls in ast.parse(
        (root / "sqlglot/expressions/core.py").read_bytes()
    ).body if isinstance(cls, ast.ClassDef) and cls.name == "Expression"
        for part in cls.body if isinstance(part, ast.FunctionDef) and part.name == "meta_get")
    # Require the bounded getter semantics that make missing keys equivalent.
    body = [stmt for stmt in pinned_method.body if not isinstance(stmt, ast.Expr)]
    expected = ast.parse("meta = self._meta\nreturn meta.get(key, default) if meta is not None else default")
    if [ast.dump(node) for node in body] != [ast.dump(node) for node in expected.body]:
        raise ValueError("Unsupported metadata getter implementation")

    @lru_cache(maxsize=16)
    def builtin_getter(fn):
        code = fn.__code__
        return (code.co_filename == str(root / "sqlglot/expressions/core.py")
                and code.co_firstlineno == pinned_method.lineno and code.co_qualname == "Expression.meta_get")

    missing = object()
    def capture(ix, fn, key, *extra):
        method, row = sites[ix]
        if importing or not tracker.mask:
            row["*"] = -1
        elif "*" not in row:
            mapping = missing
            if method == "get":
                if type(fn) is BuiltinMethodType and type(fn.__self__) is dict and fn.__name__ == "get":
                    mapping = fn.__self__
            elif type(fn) is MethodType:
                if builtin_getter(fn.__func__) and (field := storage(type(fn.__self__))):
                    mapping = field.__get__(fn.__self__, type(fn.__self__))
            if mapping is None:
                pass
            elif type(mapping) is dict and len(mapping) <= 32 and all(type(k) is str for k in mapping):
                for name in mapping:
                    name = "k" + name
                    row[name] = row.get(name, 0) | tracker.mask
            else:
                row[""] = row.get("", 0) | tracker.mask
        return fn(key, *extra)

    def audit(event, args):
        if tracker.mask and event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system", "os.exec"}:
            tracker.failed.update(i for i in range(len(tests)) if tracker.mask & (1 << i))

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_key_read = capture
    concurrent.futures.ProcessPoolExecutor = LocalExecutor
    sys.addaudithook(audit)
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "tests"],
                         plugins=[tracker])
    if status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED) or not tracker.passed:
        raise RuntimeError(f"Baseline profiling failed ({status})")
    output = {path: {key: None if "*" in row else {k: format(v, "x") for k,v in row.items()}
                     for key, row in rows.items()} for path, rows in files.items()}
    data = {"nodes": tests, "complete": sorted(tracker.passed - tracker.failed), "files": output}
    dest.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))


def merge(first, second, destination):
    left, right = [json.loads(gzip.decompress(path.read_bytes())) for path in (first, second)]
    if left["nodes"] != right["nodes"]:
        raise ValueError("Profiles use different catalogs")
    files = {}
    for name in left["files"].keys() & right["files"].keys():
        a, b = left["files"][name], right["files"][name]
        merged = {}
        for pos in a.keys() & b.keys():
            x, y = a[pos], b[pos]
            if x is None or y is None:
                merged[pos] = None
                continue
            merged[pos] = {k: format(int(x.get(k, "0"), 16) | int(y.get(k, "0"), 16), "x")
                           for k in x.keys() | y.keys()}
        files[name] = merged
    data = {"nodes": left["nodes"], "complete": sorted(set(left["complete"]) & set(right["complete"])),
            "files": files}
    destination.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        merge(*(Path(arg) for arg in sys.argv[2:5]))
    else:
        build(*(Path(arg) for arg in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:])
