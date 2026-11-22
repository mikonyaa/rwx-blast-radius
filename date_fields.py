"""Track string map lookups and the interval unit set derived from date-part aliases."""

import ast
from collections import defaultdict
import gzip
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

from exception_profile import GuardProfile
from semantic import name_of


ALIASES = "DATE_PART_MAPPING"
UNITS = "VALID_INTERVAL_UNITS"
SINGULARS = "TIME_PART_SINGULARS"
FIELDS = {ALIASES, UNITS, SINGULARS}


class DateFields(GuardProfile):
    filename = "date_fields.json.gz"

    def candidates(self, path, syntax, new, blocks):
        if path not in self.sites.get("sources", {}) or len(blocks) != 1:
            return None
        start, old_rows, new_rows = blocks[0]
        if len(old_rows) != 1 or len(new_rows) != 1:
            return None
        scope, stmt = syntax.contexts(start)
        if not isinstance(scope, ast.ClassDef):
            return None
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or name_of(stmt) not in {ALIASES, SINGULARS}:
            return None
        if not isinstance(stmt.value, ast.Dict):
            return None
        clues = [(k.value, v.value) for k, v in zip(stmt.value.keys, stmt.value.values)
                 if isinstance(k, ast.Constant) and type(k.value) is str and isinstance(v, ast.Constant)
                 and type(v.value) is str and v.lineno == start]
        if len(clues) != 1:
            return None
        key, before = clues[0]
        tree = ast.parse(new)
        after_values = [v for decl in ast.walk(tree) if type(decl) is type(stmt) and decl.lineno == stmt.lineno
                        and isinstance(decl.value, ast.Dict) for k, v in zip(decl.value.keys, decl.value.values)
                        if isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Constant)
                        and type(v.value) is str and v.lineno == start]
        if len(after_values) != 1:
            return None
        changed = after_values[0]
        after = changed.value
        changed.value = before
        if ast.dump(tree) != ast.dump(syntax.tree):
            return None
        field = name_of(stmt)
        useful = [(field, key), (field, None)]
        if field == ALIASES:
            useful += [(UNITS, before), (UNITS, after), (UNITS, None)]
        selected = self.unknown.copy()
        for name, key in useful:
            locations = self.sites["whole"] if key is None else self.sites["lookups"].get(name, {})
            masks = locations.get(name if key is None else key, [])
            if -1 in masks:
                return None
            selected.update(self.numbers[i] for i in masks if self.numbers[i] is not None)
        return selected


def derived(node, parents):
    """A dict copy preserves all individual keys, and a set union preserves membership."""
    parent = parents.get(node)
    if isinstance(parent, ast.Dict) and any(k is None and v is node for k, v in zip(parent.keys, parent.values)):
        decl = parents.get(parent)
        return isinstance(decl, (ast.Assign, ast.AnnAssign)) and name_of(decl) == node.attr
    if isinstance(parent, ast.Attribute) and parent.attr in {"values", "keys"} and node.attr == ALIASES:
        call = parents.get(parent)
        if not isinstance(call, ast.Call) or call.args or call.keywords:
            return False
        parent = parents.get(call)
    if not isinstance(parent, ast.Starred) or not isinstance(parents.get(parent), ast.Set):
        return False
    decl = parents.get(parents[parent])
    if not isinstance(decl, ast.Assign) or len(decl.targets) != 1:
        return False
    target = decl.targets[0]
    return (node.attr in {ALIASES, UNITS} and isinstance(target, (ast.Name, ast.Attribute))
            and (target.id if isinstance(target, ast.Name) else target.attr) == UNITS)


def instrument(tree):
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def recorded(field, arg, any_key=False):
        fn = ast.Name("_rwx_date_input", ast.Load())
        return ast.copy_location(ast.Call(fn, [ast.Constant(field), arg, ast.Constant(any_key)], []), arg)

    for node in list(ast.walk(tree)):
        if not isinstance(node, ast.Attribute) or node.attr not in FIELDS or not isinstance(node.ctx, ast.Load):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and call.args and not isinstance(call.args[0], ast.Starred):
                call.args[0] = recorded(node.attr, call.args[0])
                continue
        if (isinstance(parent, ast.Compare) and len(parent.ops) == 1
                and isinstance(parent.ops[0], (ast.In, ast.NotIn)) and parent.comparators[0] is node):
            parent.left = recorded(node.attr, parent.left)
            continue
        if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
            parent.slice = recorded(node.attr, parent.slice)
            continue
        if derived(node, parents):
            continue
        # Unknown calls, alias escaping, equality or iteration can inspect all values.
        for attr, obj in ast.iter_fields(parent):
            if obj is node:
                setattr(parent, attr, recorded(node.attr, node, True))
            elif isinstance(obj, list):
                for ix, sub in enumerate(obj):
                    if sub is node:
                        obj[ix] = recorded(node.attr, node, True)
    return ast.fix_missing_locations(tree)


def build(source, catalog, output, reverse):
    import builtins
    import concurrent.futures
    import pytest
    from coverage_build import Collector, LocalExecutor

    root = source.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    output = output.resolve()
    tracker = Collector(root, nodes, reverse=reverse)
    observations = defaultdict(dict)
    whole = {}
    sources = {}
    prefix = str(root / "sqlglot") + os.sep

    def record(field, key, wildcard):
        mask = tracker.mask
        # Keep initialization outside individual tests conservative.
        f = sys._getframe(1)
        while mask and f:
            if f.f_code.co_name == "<module>" and f.f_code.co_filename.startswith(prefix):
                mask = 0
                break
            f = f.f_back
        mask = mask or -1
        if wildcard or type(key) is not str:
            whole[field] = whole.get(field, 0) | mask
        else:
            groups = observations[field]
            groups[key] = groups.get(key, 0) | mask
        return key

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            filename = Path(self.path)
            tree = ast.parse(filename.read_bytes(), self.path)
            if any(isinstance(n, (ast.Assign, ast.AnnAssign)) and name_of(n) in {ALIASES, SINGULARS}
                   for n in ast.walk(tree)):
                sources[str(filename.relative_to(root))] = True
            return compile(instrument(tree), self.path, "exec")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".", 1)[0] not in {"sqlglot", "tests"}:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py") and str(spec.origin).startswith(str(root) + os.sep):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    def audit(evt, args):
        if evt in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"} and tracker.mask:
            tracker.failed.update(i for i in range(len(nodes)) if tracker.mask & (1 << i))

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_date_input = record
    concurrent.futures.ProcessPoolExecutor = LocalExecutor
    sys.addaudithook(audit)
    result = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[tracker])
    if result not in {0, 1} or not tracker.passed:
        raise RuntimeError(f"pytest exited with {result}")

    def indices(mask):
        return [-1] if mask < 0 else [i for i in range(len(nodes)) if mask & (1 << i)]
    records = {"sources": sources, "whole": {k: indices(v) for k, v in whole.items()},
               "lookups": {name: {k: indices(v) for k, v in group.items()} for name, group in observations.items()}}
    data = {"nodes": nodes, "complete": sorted(tracker.passed - tracker.failed), "files": records}
    output.write_bytes(gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0))


def merge(first, second, output):
    a, b = (json.loads(gzip.decompress(path.read_bytes())) for path in (first, second))
    if a["nodes"] != b["nodes"]:
        raise ValueError("Profiles refer to different catalogs")
    a["complete"] = sorted(set(a["complete"]) & set(b["complete"]))
    x, y = a["files"], b["files"]
    x["sources"] = {name: True for name in x["sources"] if name in y["sources"]}
    def combine(left, right):
        for key, indices in right.items():
            seen = set(left.get(key, ())) | set(indices)
            left[key] = [-1] if -1 in seen else sorted(seen)
    combine(x["whole"], y["whole"])
    for field, groups in y["lookups"].items():
        combine(x["lookups"].setdefault(field, {}), groups)
    output.write_bytes(gzip.compress(json.dumps(a, sort_keys=True, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        merge(*(Path(arg) for arg in sys.argv[2:5]))
    else:
        build(*(Path(arg) for arg in sys.argv[1:4]), "--reverse" in sys.argv[4:])
