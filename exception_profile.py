"""Observe exception classes at pristine handler checks, then compare handler edits."""
import ast
import builtins
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys


def exception_names(expr):
    if expr is None:
        return None
    values = expr.elts if isinstance(expr, ast.Tuple) else [expr]
    names = set()
    for value in values:
        if not isinstance(value, ast.Name):
            return None
        cls = getattr(builtins, value.id, None)
        if not isinstance(cls, type) or not issubclass(cls, BaseException):
            return None
        names.add(value.id)
    return names


def shadowed(tree, names):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load) and node.id in names:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in names:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest in names:
            return True
        if isinstance(node, ast.arg) and node.arg in names:
            return True
        if isinstance(node, (ast.ExceptHandler, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            return True
        if isinstance(node, ast.alias) and (node.name == "*" or (node.asname or node.name.split(".")[0]) in names):
            return True
    return False


class GuardProfile:
    """Translate the pristine test catalog to the runtime catalog."""

    def __init__(self, directory, tests):
        self.sites = {}
        path = Path(directory) / self.filename
        if not path.is_file():
            return
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        self.sites = data["files"]
        catalog = {r["id"]: r["number"] for r in tests}
        self.numbers = [catalog.get(node) for node in data["nodes"]]
        self.unknown = set(catalog.values()) - {self.numbers[i] for i in data["complete"]}


class ExceptionProfile(GuardProfile):
    filename = "exceptions.json.gz"

    def filter(self, filename, old_tree, new_source, blocks, during_import):
        sites = self.sites.get(filename)
        if not sites or len(blocks) != 1:
            return None
        line, removed, added = blocks[0]
        if len(added) != 1 or len(removed) != 1 or not added[0].lstrip().startswith("except "):
            return None
        states = sites.get(str(line))
        if states is None:
            return None
        new_tree = ast.parse(new_source)
        before, after = [next((h for h in ast.walk(t) if isinstance(h, ast.ExceptHandler) and h.lineno == line), None)
                         for t in (old_tree, new_tree)]
        if before is None or after is None:
            return None
        old, new = exception_names(before.type), exception_names(after.type)
        if old is None or new is None or shadowed(old_tree, old | new):
            return None
        old = {getattr(builtins, name).__name__ for name in old}
        new = {getattr(builtins, name).__name__ for name in new}
        # Exclude edits to the body, variable binding, or other expressions on
        # the same line. Only the new, disposable tree is changed here.
        after.type = before.type
        if ast.dump(new_tree) != ast.dump(old_tree):
            return None
        selected = self.unknown.copy()
        for ancestry, indices in states.items():
            names = set(ancestry.split(","))
            if bool(old & names) != bool(new & names):
                if during_import or -1 in indices:
                    return None
                selected.update(self.numbers[i] for i in indices)
        selected.discard(None)
        return selected


def build(root, catalog, destination, type_guards=False):
    import importlib.abc
    import importlib.machinery
    import os
    import pytest
    from coverage_build import Collector

    if type_guards:
        import type_guard

    root, destination = Path(root).resolve(), Path(destination).resolve()
    nodes = [r["id"] for r in json.loads(Path(catalog).read_text())["tests"]]
    collector = Collector(root, nodes)
    files = defaultdict(dict)

    def observe(site, expected):
        exc = sys.exception()
        # Python itself tests exception matches with the actual type hierarchy.
        ancestry = ",".join(t.__name__ for t in type(exc).__mro__ if t.__module__ == "builtins")
        if collector.mask:
            site[ancestry] = site.get(ancestry, 0) | collector.mask
        else:
            site[ancestry] = -1
        return expected

    sites = []

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            filename = Path(self.path)
            tree = ast.parse(filename.read_bytes(), self.path)
            name = str(filename.relative_to(root))
            if type_guards:
                type_guard.instrument(tree, files[name], sites)
                return compile(ast.fix_missing_locations(tree), self.path, "exec")
            # except* matches subgroups; profiling ordinary exceptions there
            # would give the wrong answer. Ignore those handlers entirely.
            ordinary = [h for n in ast.walk(tree) if isinstance(n, ast.Try) for h in n.handlers]
            for h in ordinary:
                if exception_names(h.type) is None:
                    continue
                entry = files[name].setdefault(str(h.lineno), {})
                call = ast.Call(ast.Name("_rwx_observe_exception", ast.Load()), [ast.Constant(len(sites)), h.type], [])
                sites.append(entry)
                h.type = ast.copy_location(call, h.type)
            return compile(ast.fix_missing_locations(tree), self.path, "exec")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
            mask = collector.mask
            if mask:
                collector.failed.update(i for i in range(len(nodes)) if mask & (1 << i))

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_observe_exception = lambda index, expected: observe(sites[index], expected)
    if type_guards:
        builtins._rwx_type_guard = type_guard.recorder(sites, collector)
    sys.addaudithook(audit)
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short", "tests"], plugins=[collector])
    if status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED) or not collector.passed:
        raise RuntimeError(f"profiling failed with pytest status {status}")
    for rows in files.values():
        for entry in rows.values():
            for key, mask in entry.items():
                entry[key] = [-1] if mask < 0 else [i for i in range(len(nodes)) if mask & (1 << i)]
    data = {"nodes": nodes, "complete": sorted(collector.passed - collector.failed), "files": files}
    destination.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))
    print(f"{len(data['complete'])} fully observed tests; {len(sites)} handler sites")


if __name__ == "__main__":
    build(*sys.argv[1:4], type_guards="--type-guards" in sys.argv[4:])
