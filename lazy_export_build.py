"""Measure pristine lazy-export bindings and tests that can observe their caching."""
import ast
import builtins
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
from coverage_build import Collector
from lazy_exports import cache_write


class Usage(Collector):
    def set_context(self, mask, nodeid=None):
        self.mask = mask


def build(root, catalog, output):
    os.chdir(root)
    sys.path.insert(0, str(root))
    package = "sqlglot.dialects"
    parent = importlib.import_module(package)
    # This resolver obtains the identically named object from the mapped child,
    # stores it in the package namespace, then returns it.
    lookup = ast.parse(Path(parent.__file__).read_bytes())
    getter = next(n for n in lookup.body if isinstance(n, ast.FunctionDef) and n.name == "__getattr__")
    branch = getter.body[1]
    assert ast.unparse(getter.body[0]) == "module_name = MODULE_BY_ATTRIBUTE.get(name)"
    assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "module_name"
    lock, value, write, result = branch.body
    assert isinstance(lock, ast.With) and ast.unparse(lock.items[0].context_expr) == "_import_lock"
    assert ast.unparse(lock.body[0]) == "module = importlib.import_module(f'sqlglot.dialects.{module_name}')"
    assert ast.unparse(value) == "attr = getattr(module, name)" and cache_write(write)
    assert ast.unparse(result) == "return attr"

    modules = parent.MODULE_BY_ATTRIBUTE
    assert all(type(k) is str and k and type(v) is str and v.isidentifier() for k, v in modules.items())
    exports = {k: package + "." + v for k, v in modules.items()}
    resident = {k: module for k, module in exports.items() if module in sys.modules}
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    usage = Usage(root, nodes)
    sensitive = 0
    def mark():
        nonlocal sensitive
        sensitive |= usage.mask

    class Watched(types.ModuleType):
        def __getattribute__(self, key):
            if key in {"__dict__", "__getattr__", "MODULE_BY_ATTRIBUTE", "_import_lock"}:
                mark()
            return types.ModuleType.__getattribute__(self, key)

        def __dir__(self):
            mark()
            return types.ModuleType.__dir__(self)

        def __setattr__(self, key, value):
            # Python also publishes lowercase submodules on their parent.
            if key in exports or key in {"__getattr__", "MODULE_BY_ATTRIBUTE", "_import_lock"}:
                mark()
            types.ModuleType.__setattr__(self, key, value)

        def __delattr__(self, key):
            mark()
            types.ModuleType.__delattr__(self, key)

    class Definition(types.ModuleType):
        def __setattr__(self, key, value):
            if key in resident:
                mark()
            super().__setattr__(key, value)

        def __delattr__(self, key):
            if key in resident:
                mark()
            super().__delattr__(key)

    class ImportMachinery(types.ModuleType):
        def __setattr__(self, key, value):
            if key == "import_module":
                mark()
            super().__setattr__(key, value)

        def __delattr__(self, key):
            if key == "import_module":
                mark()
            super().__delattr__(key)

    for target in set(resident.values()):
        child = sys.modules[target]
        if type(child) is not types.ModuleType:
            raise RuntimeError("unmodelled module class")
        child.__class__ = Definition
    parent.__class__ = Watched
    importlib.__class__ = ImportMachinery
    real_import = builtins.__import__
    real_import_module = importlib.import_module
    def observe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == package and fromlist:
            if importlib.import_module is not real_import_module:
                mark()
            namespace = types.ModuleType.__getattribute__(parent, "__dict__")
            for alias in fromlist:
                target = sys.modules.get(exports.get(alias, ""))
                if alias in exports and target is not None and alias in namespace:
                    if namespace[alias] is not vars(target).get(alias):
                        mark()
        return real_import(name, globals, locals, fromlist, level)

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system", "sys.settrace", "sys.setprofile"}:
            mark()
    sys.addaudithook(audit)
    builtins.__import__ = observe_import
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    rc = pytest.main(["-q", "--tb=no", "-p", "no:cacheprovider", "tests"], plugins=[usage])
    if rc not in (0, 1) or not usage.passed:
        raise RuntimeError(f"pytest status {rc}")
    clean = sorted(nodes[i] for i in usage.passed - usage.failed if not sensitive & (1 << i))
    payload = {"package": package, "exports": resident, "observed_clean": clean}
    output.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(),mtime=0))
    print(f"{len(clean)} tests without observed reads or mutation of the package cache")


if __name__ == "__main__":
    build(*(Path(x).resolve() for x in sys.argv[1:]))
