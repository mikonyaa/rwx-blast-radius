"""Observe consumers of shared expression type metadata on the pinned test suite."""

import ast
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

from exception_profile import GuardProfile


def dtype(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        outer = node.value
        if isinstance(outer.value, ast.Name) and outer.value.id == "exp" and outer.attr == "DType":
            return node.attr
    return None


def expression(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp":
        return node.attr
    return None


def returns_value(record):
    if not isinstance(record, ast.Dict) or len(record.keys) != 1:
        return None
    key = record.keys[0]
    value = record.values[0]
    return value if isinstance(key, ast.Constant) and key.value == "returns" and dtype(value) else None


def declared_rows(tree):
    for stmt in tree.body:
        target = (stmt.target if isinstance(stmt, ast.AnnAssign) else stmt.targets[0]
                  if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 else None)
        if not isinstance(target, ast.Name) or target.id != "EXPRESSION_METADATA" or not isinstance(stmt.value, ast.Dict):
            continue
        for name, record in zip(stmt.value.keys, stmt.value.values):
            value = returns_value(record)
            if value and expression(name):
                yield value, {expression(name)}
            elif name is None and isinstance(record, ast.DictComp) and len(record.generators) == 1:
                generator = record.generators[0]
                if (not isinstance(record.key, ast.Name) or not isinstance(generator.target, ast.Name)
                        or record.key.id != generator.target.id or not isinstance(generator.iter, (ast.Set, ast.Tuple, ast.List))
                        or generator.ifs):
                    continue
                classes = {expression(node) for node in generator.iter.elts}
                value = returns_value(record.value)
                if value and classes and None not in classes:
                    yield value, classes


class MetadataReads(GuardProfile):
    filename = "metadata_reads.json.gz"

    def __init__(self, directory, tests):
        super().__init__(directory, tests)
        self.types = set(self.sites.pop("$types", {}).get("fields", ()))

    def candidates(self, path, syntax, updated, blocks):
        rows = self.sites.get(path)
        if not rows or len(blocks) != 1:
            return None
        line, removed, added = blocks[0]
        row = rows.get(str(line))
        if row is None or len(removed) != 1 or len(added) != 1:
            return None
        original = [value for value, _ in declared_rows(syntax.tree) if value.lineno == line]
        if len(original) != 1:
            return None
        tree = ast.parse(updated)
        for value, _ in declared_rows(tree):
            if value.lineno != line or dtype(value) not in self.types:
                continue
            value.attr = original[0].attr
            if ast.dump(tree) == ast.dump(syntax.tree):
                return self.unknown | {self.numbers[i] for i in row if self.numbers[i] is not None}
        return None


def build(source, catalog, dest, reverse=False):
    import importlib
    import os
    import pytest
    from coverage_build import Collector

    root = source.resolve()
    dest = dest.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_text())["tests"]]
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import Dialect, exp
    from sqlglot.executor.python import Python

    class Visits(Collector):
        def set_context(self, mask, nodeid=None):
            self.mask = mask

    tracker = Visits(root, nodes, reverse=reverse)
    records = defaultdict(set)
    masks = {}
    for source_file in (root/"sqlglot"/"typing").glob("*.py"):
        name = "sqlglot.typing" if source_file.stem == "__init__" else "sqlglot.typing." + source_file.stem
        module = importlib.import_module(name)
        table = module.EXPRESSION_METADATA
        for value, exprs in declared_rows(ast.parse(source_file.read_bytes())):
            site = str(source_file.relative_to(root)), value.lineno
            if any(getattr(exp, expr, None) not in table for expr in exprs):
                continue
            masks[site] = 0
            for expr in exprs:
                records[id(table[getattr(exp, expr)])].add(site)

    def observe(signals):
        for site in signals:
            masks[site] = masks[site] | tracker.mask if tracker.mask else -1

    class Lookup(dict):
        def __init__(self, table):
            super().__init__(table)
            self.signals = {key: records[id(value)] for key, value in table.items() if id(value) in records}
            self.every = {s for sites in self.signals.values() for s in sites}

        def get(self, key, default=None):
            observe(self.signals.get(key, ()))
            return super().get(key, default)

        def __getitem__(self, key):
            observe(self.signals.get(key, ()))
            return super().__getitem__(key)

        def items(self):
            observe(self.every)
            return super().items()

        def values(self):
            observe(self.every)
            return super().values()

        def copy(self):
            observe(self.every)
            return super().copy()

        def __eq__(self, other):
            observe(self.every)
            return super().__eq__(other)

        def __ne__(self, other):
            observe(self.every)
            return super().__ne__(other)

    dialects = {Python, *Dialect.classes.values()}
    classes = {cls for dialect in dialects for cls in dialect.__mro__}
    modules = {mod for name, mod in sys.modules.items() if name == "sqlglot" or name.startswith("sqlglot.")}
    replacements = {}
    for obj in classes | modules:
        table = vars(obj).get("EXPRESSION_METADATA")
        if type(table) is dict:
            if id(table) not in replacements:
                replacements[id(table)] = Lookup(table)
            setattr(obj, "EXPRESSION_METADATA", replacements[id(table)])

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"} and tracker.mask:
            tracker.failed.update(i for i in range(len(nodes)) if tracker.mask & (1 << i))
    sys.addaudithook(audit)
    result = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[tracker])
    if result not in (0, 1) or not tracker.passed:
        raise RuntimeError(f"pytest exited with {result}")
    files = defaultdict(dict)
    for (path, line), mask in masks.items():
        if mask >= 0:
            files[path][str(line)] = [i for i in range(len(nodes)) if mask & (1 << i)]
    files["$types"] = {"fields": sorted(exp.DType.__members__)}
    data = {"nodes": nodes, "complete": sorted(tracker.passed - tracker.failed), "files": files}
    dest.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        from callback_profile import merge
        merge(*(Path(arg) for arg in sys.argv[2:]))
    else:
        build(*(Path(arg) for arg in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:])
