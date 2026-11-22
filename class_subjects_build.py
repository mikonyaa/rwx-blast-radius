"""Collect per-test live subjects of isinstance calls on the pinned SQLGlot suite."""

import ast
import builtins
from collections import defaultdict
from concurrent import futures
from functools import lru_cache
import gzip
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
from types import ModuleType

from class_subjects import members, site


def build(root, catalog, output, reverse=False):
    import pytest
    from coverage_build import Collector, LocalExecutor

    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    collector = Collector(root, nodes, reverse=reverse)
    locations, specs = defaultdict(dict), []
    original = isinstance
    target_names, class_mros, sensitive = {}, {}, defaultdict(int)
    baseline_symbols = {}
    exp_module = None

    @lru_cache(maxsize=None)
    def identify(cls):
        if type(cls) is not type:
            return "?"
        # Only a real, unchanged object.__class__ can establish membership in a
        # class that was not tested on the baseline. Mocks may spoof __class__.
        getter = next(
            (
                b.__dict__["__getattribute__"]
                for b in cls.__mro__
                if "__getattribute__" in b.__dict__
            ),
            None,
        )
        descr = next(
            (b.__dict__["__class__"] for b in cls.__mro__ if "__class__" in b.__dict__),
            None,
        )
        if (
            getter is not object.__getattribute__
            or descr is not object.__dict__["__class__"]
        ):
            return "?"
        label = str(len(class_mros))
        class_mros[label] = [str(id(b)) for b in cls.__mro__]
        return label

    def record(i, fn, value, targets, exp):
        row, attrs = specs[i]
        vals = targets if type(targets) is tuple else (targets,)
        valid = (
            fn is original
            and exp is exp_module
            and all(
                type(v) is type and baseline_symbols.get(key) is v
                for key, v in zip(attrs, vals)
            )
        )
        tag = identify(type(value)) if valid else "?"
        row[tag] = row.get(tag, 0) | (collector.mask or -1)
        return fn(value, targets)

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if not (
                isinstance(n.func, ast.Name)
                and n.func.id == "isinstance"
                and len(n.args) == 2
                and not n.keywords
                and (attrs := members(n.args[1]))
            ):
                return n
            row = locations[self.path].setdefault(site(n), {})
            sid = len(specs)
            specs.append((row, attrs))
            return ast.copy_location(
                ast.Call(
                    ast.Name("_rwx_class_subject", ast.Load()),
                    [
                        ast.Constant(sid),
                        n.func,
                        n.args[0],
                        n.args[1],
                        ast.Name("exp", ast.Load()),
                    ],
                    [],
                ),
                n,
            )

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = Path(self.path)
            relative = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_bytes(), str(path))
            rr = Rewrite()
            rr.path = relative
            return compile(ast.fix_missing_locations(rr.visit(tree)), str(path), "exec")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if not (
                fullname.startswith(("sqlglot.optimizer.", "sqlglot.executor."))
                or fullname == "sqlglot.planner"
            ):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    def audit(event, args):
        if (
            event
            in ("subprocess.Popen", "os.system", "os.fork", "os.posix_spawn", "os.exec")
            and collector.mask
        ):
            collector.failed.update(
                i for i in range(len(nodes)) if collector.mask & (1 << i)
            )

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_class_subject = record
    from sqlglot import expressions as exp

    exp_module = exp
    baseline_symbols = {
        k: v
        for k, v in vars(exp).items()
        if type(v) is type and v.__module__.startswith("sqlglot.expressions.")
    }
    target_names = {k: str(id(v)) for k, v in baseline_symbols.items()}

    class WatchExports(ModuleType):
        def __setattr__(self, name, value):
            if name in baseline_symbols:
                sensitive[name] |= collector.mask or -1
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if name in baseline_symbols:
                sensitive[name] |= collector.mask or -1
            super().__delattr__(name)

    exp.__class__ = WatchExports
    futures.ProcessPoolExecutor = LocalExecutor
    sys.addaudithook(audit)
    status = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[collector]
    )
    if status not in (0, 1) or not collector.passed:
        raise RuntimeError(f"pytest failed: {status}")
    # Use symbolic MRO keys, so the retained profile has no process-specific addresses.
    exp_ids = {key: name for name, key in sorted(target_names.items(), reverse=True)}
    output_data = {
        "classes": {
            k: [exp_ids[i] for i in v if i in exp_ids] for k, v in class_mros.items()
        },
        "members": {name: exp_ids[i] for name, i in target_names.items()},
        "nodes": nodes,
        "complete": sorted(collector.passed - collector.failed),
        "sensitive": {
            k: format(v if v >= 0 else (1 << len(nodes)) - 1, "x")
            for k, v in sensitive.items()
        },
        "files": {
            p: {
                spot: {
                    k: ("*" if value < 0 else format(value, "x"))
                    for k, value in row.items()
                }
                for spot, row in sites.items()
            }
            for p, sites in locations.items()
        },
    }
    output.write_bytes(
        gzip.compress(
            json.dumps(output_data, sort_keys=True, separators=(",", ":")).encode(),
            mtime=0,
        )
    )


if __name__ == "__main__":
    build(
        *(Path(x).resolve() for x in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:]
    )
