"""Attribute dialect use and dynamic imports on the unmodified test suite."""

import argparse
import builtins
from collections import defaultdict
import gzip
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from coverage_build import Collector, encode


class DialectUsage(Collector):
    def __init__(self, root, nodes):
        super().__init__(root, nodes)
        self.objects = {}
        self.last = {}
        self.users = defaultdict(int)
        self.modules = defaultdict(int)
        self.attrs = defaultdict(dict)
        self.all_dialects = 0
        self.subprocesses = 0

    def set_context(self, mask, nodeid=None):
        self.mask = mask

    def observe(self, cls, attr=None):
        mask = self.mask
        if not mask:
            return
        key = id(cls)
        if self.last.get(key) != mask:
            self.last[key] = mask
            self.objects[key] = cls
            self.users[key] |= mask
        if attr and (attr.isupper() or attr == "__dict__"):
            by_attr = self.attrs[key]
            by_attr[attr] = by_attr.get(attr, 0) | mask

    def note_module(self, name, fromlist=()):
        if name.startswith("sqlglot"):
            self.modules[name] |= self.mask
            if name == "sqlglot.dialects":
                for part in fromlist or ():
                    self.modules[name + "." + part.lower()] |= self.mask

    def install(self):
        from sqlglot import Dialect

        meta = type(Dialect)
        class_get = meta.__getattribute__
        obj_get = Dialect.__getattribute__
        real_import = builtins.__import__
        import_module = importlib.import_module
        run = subprocess.run

        def on_class(cls, key):
            if self.mask:
                self.observe(cls, key)
                if key == "classes":
                    self.all_dialects |= self.mask
            return class_get(cls, key)

        def on_instance(obj, key):
            if self.mask:
                self.observe(type(obj), key)
            return obj_get(obj, key)

        def follow_module(name, package=None):
            if self.mask and (
                name.startswith("sqlglot") or (name.startswith(".") and package)
            ):
                self.note_module(importlib.util.resolve_name(name, package))
            return import_module(name, package)

        def follow_import(name, globals=None, locals=None, fromlist=(), level=0):
            if self.mask:
                if level and globals and (package := globals.get("__package__")):
                    if package.startswith("sqlglot"):
                        self.note_module(
                            importlib.util.resolve_name("." * level + name, package),
                            fromlist,
                        )
                elif name.startswith("sqlglot"):
                    self.note_module(name, fromlist)
            return real_import(name, globals, locals, fromlist, level)

        def follow_python(args, *a, **kw):
            if (
                self.mask
                and isinstance(args, (list, tuple))
                and args
                and Path(args[0]).name.startswith("python")
            ):
                self.subprocesses |= self.mask
            return run(args, *a, **kw)

        for key in ("__getitem__", "get"):
            original = vars(meta)[key].__func__

            def observed(cls, *args, __original=original, **kw):
                result = __original(cls, *args, **kw)
                if isinstance(result, meta):
                    self.observe(result)
                return result

            setattr(meta, key, classmethod(observed))
        meta.__getattribute__ = on_class
        Dialect.__getattribute__ = on_instance
        importlib.import_module = follow_module
        builtins.__import__ = follow_import
        subprocess.run = follow_python

    def save(self, path):
        names = defaultdict(int)
        attrs = defaultdict(lambda: defaultdict(int))
        for key, cls in self.objects.items():
            recorded_attrs = False
            # Class creation and dynamic plugin lookups can visit metaclasses too.
            for base in type.__getattribute__(cls, "__mro__"):
                module = type.__getattribute__(base, "__module__")
                if module.startswith("sqlglot.dialects."):
                    dialect = module.split(".")[-1]
                    names[dialect] |= self.users[key]
                    if not recorded_attrs:
                        for attr, mask in self.attrs[key].items():
                            attrs[dialect][attr] |= mask
                        recorded_attrs = True
        missing = set(range(len(self.nodes))) - self.passed | self.failed
        missing.update(
            i for i in range(len(self.nodes)) if self.subprocesses & (1 << i)
        )
        data = {
            "attributes": {
                name: {k: encode(v) for k, v in rows.items()}
                for name, rows in attrs.items()
            },
            "nodes": self.nodes,
            "dialects": {k: encode(v) for k, v in names.items()},
            "all_dialects": encode(self.all_dialects),
            "incomplete": sorted(missing),
            "modules": {k: encode(v) for k, v in self.modules.items()},
        }
        with path.open("wb") as out:
            out.write(
                gzip.compress(
                    json.dumps(data, sort_keys=True, separators=(",", ":")).encode(),
                    mtime=0,
                )
            )
        print(
            f"{len(names)} dialects, {len(self.modules)} dynamically imported names, "
            f"{len(missing)} incomplete tests"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("catalog", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--only", nargs="+")
    args = p.parse_args()
    root = args.source.resolve()
    out = args.output.resolve()
    nodes = [row["id"] for row in json.loads(args.catalog.read_text())["tests"]]
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.chdir(root)
    sys.path.insert(0, str(root))
    recorder = DialectUsage(root, nodes)
    recorder.install()
    status = pytest.main(
        ["-q", "--tb=short", "-p", "no:cacheprovider", *(args.only or ["tests"])],
        plugins=[recorder],
    )
    if status not in (0, 1) or not recorder.passed:
        raise SystemExit(status or 1)
    recorder.save(out)


if __name__ == "__main__":
    main()
