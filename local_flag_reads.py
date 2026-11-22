"""Observe consumers of immutable dialect options cached in ordinary local variables."""

import ast
from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path


@lru_cache(maxsize=2)
def profile(folder):
    path = folder / "local_flags.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.is_file() else {}


def candidates(folder, path, stmt, fresh):
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or fresh is None:
        return None
    if not (
        isinstance(stmt.value, ast.Constant)
        and isinstance(fresh.value, ast.Constant)
        and type(stmt.value.value) is type(fresh.value.value)
    ):
        return None
    data = profile(folder)
    return data.get(path, {}).get(str(stmt.lineno))


def bindings(fn, path):
    """Locate simple assignments that keep a flag in a nonescaping Python local."""
    nodes = list(ast.walk(fn))
    if any(
        isinstance(n, ast.Name)
        and n.id
        in {
            "locals",
            "vars",
            "eval",
            "exec",
            "inspect",
            "sys",
            "globals",
            "currentframe",
        }
        for n in nodes
    ):
        return {}
    nested = set()
    for node in nodes:
        if node is not fn and isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.ClassDef,
                ast.GeneratorExp,
            ),
        ):
            nested.update(ast.walk(node))
    result = {}
    for stmt in nodes:
        if stmt in nested or not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        target = (
            stmt.target
            if isinstance(stmt, ast.AnnAssign)
            else (stmt.targets[0] if len(stmt.targets) == 1 else None)
        )
        if not (
            isinstance(target, ast.Name)
            and isinstance(stmt.value, ast.Attribute)
            and stmt.value.attr.isupper()
        ):
            continue
        if (
            sum(
                isinstance(n, ast.Attribute)
                and n.lineno == stmt.value.lineno
                and n.attr == stmt.value.attr
                for n in nodes
            )
            != 1
        ):
            continue
        if any(isinstance(x, ast.Name) and x.id == target.id for x in nested):
            continue
        if any(
            isinstance(x, (ast.Global, ast.Nonlocal)) and target.id in x.names
            for x in nodes
        ):
            continue
        if (
            sum(
                isinstance(n, ast.Name)
                and n.id == target.id
                and isinstance(n.ctx, ast.Store)
                for n in nodes
            )
            != 1
        ):
            continue
        result[target.id] = (path, stmt.value.lineno, stmt.value.attr)
    return result


def build(source, catalog, destination, reverse=False):
    import builtins
    import importlib.abc
    import importlib.machinery
    import os
    import sys
    import pytest
    from coverage_build import Collector

    source = source.resolve()
    catalog, destination = catalog.resolve(), destination.resolve()
    nodes = [t["id"] for t in json.loads(catalog.read_bytes())["tests"]]
    collector = Collector(source, nodes, reverse=reverse)
    ids = {}
    used = defaultdict(int)

    class Names(ast.NodeTransformer):
        def __init__(self, local):
            self.local = local

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in self.local:
                n = self.local[node.id]
                call = ast.Call(
                    ast.Name("_rwx_use_dialect_local", ast.Load()),
                    [ast.Constant(n), node],
                    [],
                )
                return ast.copy_location(call, node)
            return node

        def visit_FunctionDef(self, node):
            return node

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef
        visit_GeneratorExp = visit_FunctionDef

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            filename = str(Path(self.path).resolve())
            tree = ast.parse(Path(filename).read_bytes(), filename)
            for fn in [
                x
                for x in ast.walk(tree)
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]:
                local = {
                    k: ids.setdefault(site, len(ids))
                    for k, site in bindings(fn, filename).items()
                }
                if local:
                    transform = Names(local)
                    fn.body = [transform.visit(stmt) for stmt in fn.body]
            return compile(ast.fix_missing_locations(tree), filename, "exec")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    def local_used(index, value):
        used[index] |= collector.mask if collector.mask else -1
        return value

    class Observe:
        def __init__(self, value, attr):
            self.value, self.attr = value, attr
            self.direct = 0
            self.captured = defaultdict(int)

        def __get__(self, instance, owner):
            f = sys._getframe(1)
            site = (f.f_code.co_filename, f.f_lineno, self.attr)
            if site in ids:
                self.captured[ids[site]] |= collector.mask if collector.mask else -1
            else:
                self.direct |= collector.mask if collector.mask else -1
            return self.value

    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_use_dialect_local = local_used
    from sqlglot.dialects import Dialect
    from semantic import name_of

    classes = set(Dialect.classes.values()) | {Dialect}
    attrs = {s[2] for s in ids}
    stored = []
    sem = json.loads(
        gzip.decompress((destination.parent / "semantic.json.gz").read_bytes())
    )
    for cls in classes:
        filename = Path(sys.modules[cls.__module__].__file__).resolve()
        rel = filename.relative_to(source).as_posix()
        lines = sem["bindings"].get(rel, {})
        klass = next(
            (
                x
                for x in ast.parse(filename.read_bytes()).body
                if isinstance(x, ast.ClassDef) and x.name == cls.__name__
            ),
            None,
        )
        if not klass:
            continue
        for stmt in klass.body:
            attr = name_of(stmt)
            if (
                attr not in attrs
                or "all" not in lines.get(str(stmt.lineno), {})
                or not isinstance(stmt.value, ast.Constant)
            ):
                continue
            value = vars(cls).get(attr)
            if (
                type(value) not in (int, bool, float, str)
                or type(value) is not type(stmt.value.value)
                or value != stmt.value.value
            ):
                continue
            obs = Observe(value, attr)
            setattr(cls, attr, obs)
            stored.append((rel, str(stmt.lineno), obs))

    def audit(event, args):
        if collector.mask and event in {
            "subprocess.Popen",
            "os.system",
            "os.fork",
            "os.posix_spawn",
            "os.exec",
        }:
            collector.failed.update(
                i for i in range(len(nodes)) if collector.mask & (1 << i)
            )

    sys.addaudithook(audit)
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    status = pytest.main(
        ["-q", "--tb=no", "-p", "no:cacheprovider", "tests"], plugins=[collector]
    )
    if status not in (0, 1) or not collector.passed:
        raise RuntimeError(status)
    unknown = set(range(len(nodes))) - (collector.passed - collector.failed)
    data = defaultdict(dict)
    for path, line, obs in stored:
        if obs.direct < 0 or not obs.captured:
            continue
        mask = obs.direct
        for site, bits in obs.captured.items():
            if bits < 0 or used[site] < 0:
                mask = -1
                break
            mask |= bits & used[site]
        if mask >= 0:
            data[path][line] = [
                node for i, node in enumerate(nodes) if i in unknown or mask & (1 << i)
            ]
    destination.write_bytes(
        gzip.compress(
            json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0
        )
    )
    print(
        "eligible",
        len(stored),
        "specialized",
        sum(map(len, data.values())),
        "named locals",
        len(ids),
        "unknown",
        len(unknown),
    )


def merge(left, right, destination):
    forward, reverse = [
        json.loads(gzip.decompress(path.read_bytes())) for path in (left, right)
    ]
    data = {}
    for filename in forward.keys() & reverse.keys():
        a, b = forward[filename], reverse[filename]
        data[filename] = {
            line: sorted(set(a[line]) | set(b[line])) for line in a.keys() & b.keys()
        }
    destination.write_bytes(
        gzip.compress(
            json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0
        )
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["--merge"]:
        merge(*(Path(arg) for arg in sys.argv[2:5]))
    else:
        build(
            *(Path(arg) for arg in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:]
        )
