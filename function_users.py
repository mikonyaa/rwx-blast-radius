"""Track callers and code-object inspection of functions whose local bindings change."""

import ast
import gzip
import json
import sys
from pathlib import Path


def plain_functions(tree):
    result = {}

    def add(prefix, declarations):
        for stmt in declarations:
            if (isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list and
                    not stmt.name.startswith("__")):
                result[prefix + stmt.name] = stmt

    add("", tree.body)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and not cls.decorator_list and not cls.keywords:
            add(cls.name + ".", cls.body)
    return result


def changed_function(old, new, blocks):
    starts = [start for start, _, _ in blocks]
    ends = [start + max(1, len(removed)) - 1 for start, removed, _ in blocks]
    if not starts:
        return None
    low, high = min(starts), max(ends)
    matches = [(key, s) for key, s in plain_functions(old).items()
               if s.lineno < s.body[0].lineno <= low and high <= s.end_lineno]
    if len(matches) != 1:
        return None
    key, before = matches[0]
    fresh = ast.parse(new)
    after = plain_functions(fresh).get(key)
    if after is None or after.lineno != before.lineno:
        return None
    if ast.get_docstring(before, clean=False) != ast.get_docstring(after, clean=False):
        return None
    # Conversion between ordinary functions, coroutines and generators can be
    # observed by callers without ever executing the function body.
    if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal))
           for fn in (before, after) for n in ast.walk(fn)):
        return None
    after.body = before.body
    return (key, before) if ast.dump(old) == ast.dump(fresh) else None


class FunctionUsers:
    def __init__(self, directory, tests):
        path = Path(directory) / "function_observers.json.gz"
        self.files = {}
        if not path.is_file():
            return
        profile = json.loads(gzip.decompress(path.read_bytes()))
        ids = {r["id"]: r["number"] for r in tests}
        self.numbers = [ids.get(node) for node in profile["nodes"]]
        self.unknown = set(ids.values()) - {self.numbers[i] for i in profile["complete"]}
        self.files = profile["files"]
        self.methods = profile.get("methods", False)

    def candidates(self, impact, path, new, blocks):
        if not impact.import_trace or path not in self.files or path not in impact.syntax:
            return None
        change = changed_function(impact.syntax[path].tree, new, blocks)
        if change is None:
            return None
        name, fn = change
        if "." in name and not self.methods:
            return None
        low, high = fn.body[0].lineno, fn.end_lineno
        init = impact.import_trace
        if (any(low <= line <= high for line in init.lazy_lines.get(path, ()))
                or impact.interval(init.lines.get(path, {}), low, high)):
            return None
        watchers = self.files[path].get(name, [])
        if -1 in watchers:
            return None
        mask = impact.interval(impact.trace.lines.get(path, {}), low, high)
        mask |= impact.trace.reads.get(path, 0)
        selected = impact.trace.test_numbers(mask) | impact.trace.incomplete | self.unknown
        selected.update(self.numbers[i] for i in watchers if self.numbers[i] is not None)
        return selected


def build(source, catalog, dest, reverse=False):
    import concurrent.futures
    import os
    from types import FunctionType
    import pytest
    from coverage_build import Collector, LocalExecutor

    source = source.resolve()
    nodes = [r['id'] for r in json.loads(catalog.read_bytes())['tests']]
    tracker = Collector(source, nodes, reverse=reverse)
    files = {}
    modules = {}
    for p in (source/'sqlglot').rglob('*.py'):
        names = plain_functions(ast.parse(p.read_bytes()))
        if not names:
            continue
        rel = p.relative_to(source).as_posix()
        name = rel[:-3].replace('/', '.').removesuffix('.__init__')
        modules[name] = (rel, set(names))
        files[rel] = {}

    def audit(event, args):
        if event == 'object.__getattr__' and len(args) >= 2 and args[1] == '__code__':
            fn = args[0]
            if type(fn) is FunctionType and (row := modules.get(fn.__module__)):
                path, names = row
                if fn.__qualname__ in names:
                    observers = files[path]
                    observers[fn.__qualname__] = observers.get(fn.__qualname__, 0) | (tracker.mask or -1)
        elif event in {'subprocess.Popen', 'os.fork', 'os.posix_spawn','os.system'} and tracker.mask:
            tracker.failed.update(i for i in range(len(nodes)) if tracker.mask & (1 << i))
    os.chdir(source)
    sys.path.insert(0,str(source))
    os.environ.update(SKIP_INTEGRATION='0',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    concurrent.futures.ProcessPoolExecutor = LocalExecutor
    sys.addaudithook(audit)
    result = pytest.main(['-q','-p','no:cacheprovider','--tb=no','tests'],plugins=[tracker])
    if result not in (0,1) or not tracker.passed:
        raise RuntimeError(f'Unusable profile, pytest exit {result}')
    files = {p: {fn: [-1] if mask < 0 else [i for i in range(len(nodes)) if mask >> i & 1]
                 for fn, mask in values.items()} for p, values in files.items()}
    data = {'files': files, 'methods': True, 'nodes': nodes,'complete': sorted(tracker.passed-tracker.failed)}
    dest.write_bytes(gzip.compress(json.dumps(data, separators=(',',':'),sort_keys=True).encode(),mtime=0))


def merge(first, second, dest):
    profiles = [json.loads(gzip.decompress(path.read_bytes())) for path in (first, second)]
    a, b = profiles
    if a["nodes"] != b["nodes"]:
        raise ValueError("Profiles belong to different catalogs")
    a["methods"] = bool(a.get("methods") and b.get("methods"))
    a["complete"] = sorted(set(a["complete"]) & set(b["complete"]))
    for path in list(a["files"]):
        if path not in b["files"]:
            del a["files"][path]
            continue
        for fn, seen in b["files"][path].items():
            observed = set(a["files"][path].get(fn, ())) | set(seen)
            a["files"][path][fn] = [-1] if -1 in observed else sorted(observed)
    dest.write_bytes(gzip.compress(json.dumps(a, sort_keys=True, separators=(",", ":")).encode(), mtime=0))


if __name__ == '__main__':
    paths = [Path(arg).resolve() for arg in sys.argv[1:4]]
    if "--merge" in sys.argv[4:]:
        merge(*paths)
    else:
        build(*paths, reverse="--reverse" in sys.argv[4:])
