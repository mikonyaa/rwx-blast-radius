"""Profile resolved literal defaults and reads of Python function defaults."""

import ast
from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path
import sys
import types

from exception_profile import GuardProfile


def defaults(node):
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    return [
        (param.arg, value)
        for param, value in [
            *zip(positional[len(positional) - len(args.defaults) :], args.defaults),
            *zip(args.kwonlyargs, args.kw_defaults),
        ]
        if isinstance(value, ast.Constant)
        and type(value.value) in (type(None), int, bool, str)
    ]


def sites(tree):
    """Top-level and class methods only. Nested functions can share source sites."""

    def functions(body):
        for i, node in enumerate(body):
            if isinstance(node, ast.ClassDef):
                yield from functions(node.body)
                continue
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = node.decorator_list
            if decorators:
                if len(decorators) != 1 or ast.unparse(decorators[0]) != "t.overload":
                    continue
                if not any(
                    isinstance(s, ast.FunctionDef)
                    and s.name == node.name
                    and not s.decorator_list
                    for s in body[i + 1 :]
                ):
                    continue
            if node.name.startswith("__"):
                continue
            for name, value in defaults(node):
                yield node, name, value

    yield from functions(tree.body)


def location(fn):
    return fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno


class DefaultProfile(GuardProfile):
    filename = "defaults.json.gz"

    def candidates(self, path, tree, new, blocks, import_trace):
        observed = self.sites.get(path)
        if not observed or len(blocks) != 1:
            return None
        start, removed, added = blocks[0]
        if len(removed) != 1 or len(added) != 1 or "=" not in removed[0]:
            return None
        matches = [(f, n, v) for f, n, v in sites(tree) if v.lineno == start]
        if not matches:
            return None
        updated = ast.parse(new)
        options = {(f.lineno, n): v for f, n, v in sites(updated)}
        for fn, name, old in matches:
            if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(fn)):
                continue
            fresh = options.get((fn.lineno, name))
            if fresh is None:
                continue
            if fresh.lineno != start or ast.dump(old) == ast.dump(fresh):
                continue
            value = fresh.value
            fresh.value = old.value
            identical = ast.dump(tree) == ast.dump(updated)
            fresh.value = value
            if not identical:
                return None
            row = observed.get(f"{location(fn)}:{name}")
            if row is None or row == "*":
                return None
            if not fn.decorator_list:
                # An initializer's use of a real function can be inherited later
                # by every test. Cold import observations outrank scoped traces.
                for table in (import_trace.lines, import_trace.lazy_lines):
                    lines = table.get(path, ())
                    if any(fn.body[0].lineno <= n <= fn.end_lineno for n in lines):
                        return None
            selected = {self.numbers[i] for i in row}
            selected.discard(None)
            return selected | self.unknown
        return None


def build(source, catalog, dest, reverse=False):
    import os
    import pytest
    from coverage_build import Collector

    root = source.resolve()
    dest = dest.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_text())["tests"]]
    collector = Collector(root, nodes, reverse=reverse)
    tracked = {}
    all_sites = defaultdict(dict)
    seen_modules = set()
    for filename in (root / "sqlglot").rglob("*.py"):
        rel = str(filename.relative_to(root))
        for fn, name, literal in sites(ast.parse(filename.read_bytes())):
            key = f"{location(fn)}:{name}"
            tracked.setdefault((str(filename), location(fn), fn.name), []).append(
                (name, literal.value, key)
            )
            all_sites[rel][key] = 0
    root_prefix = str(root) + os.sep
    visits = {}

    def record(rows, key, mask):
        rows[key] = (rows[key] | mask) if mask else -1

    @lru_cache(None)
    def target(code):
        attrs = tracked.get((code.co_filename, code.co_firstlineno, code.co_name))
        if not attrs:
            return None
        return all_sites[code.co_filename[len(root_prefix) :]], attrs

    def called(code, offset):
        if code.co_name == "<module>" and code.co_filename.startswith(root_prefix):
            seen_modules.add(code.co_filename[len(root_prefix) :])
        result = target(code)
        if not result:
            return sys.monitoring.DISABLE
        rows, attrs = result
        mask = collector.mask
        if not mask:
            for _, _, key in attrs:
                record(rows, key, mask)
            return sys.monitoring.DISABLE
        mask_count = visits.get(code)
        count = mask_count[1] + 1 if mask_count and mask_count[0] == mask else 1
        visits[code] = (mask, count)
        frame = sys._getframe(1)
        if frame.f_code is not code:
            count = 4
        local = frame.f_locals
        complete = True
        for name, default, key in attrs:
            if (
                count >= 4
                or name not in local
                or (type(local[name]) is type(default) and local[name] == default)
            ):
                record(rows, key, mask)
            if rows[key] & mask != mask:
                complete = False
        if complete:
            return sys.monitoring.DISABLE

    def audit(event, args):
        if event == "object.__getattr__" and args[1] in {
            "__defaults__",
            "__kwdefaults__",
        }:
            fn = args[0]
            if isinstance(fn, types.FunctionType):
                result = target(fn.__code__)
                if result:
                    rows, attrs = result
                    for _, _, key in attrs:
                        record(rows, key, collector.mask)
        elif (
            event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}
            and collector.mask
        ):
            mask = collector.mask
            collector.failed.update(i for i in range(len(nodes)) if mask & (1 << i))

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, "optional argument consumers")
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, called)
    sys.monitoring.set_events(tool, sys.monitoring.events.PY_START)
    sys.addaudithook(audit)
    try:
        status = pytest.main(
            ["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[collector]
        )
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.free_tool_id(tool)
    if status not in (0, 1) or not collector.passed:
        raise RuntimeError(f"pytest exited with {status}")
    files = {
        file: {
            key: ("*" if m < 0 else [i for i in range(m.bit_length()) if m >> i & 1])
            for key, m in rows.items()
        }
        for file, rows in all_sites.items()
        if file in seen_modules
    }
    data = {
        "nodes": nodes,
        "complete": sorted(collector.passed - collector.failed),
        "files": files,
    }
    dest.write_bytes(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
    )
    print(len(files), "modules;", len(data["complete"]), "fully observed tests")


def merge_profiles(first, second, dest):
    left, right = [json.loads(gzip.decompress(p.read_bytes())) for p in (first, second)]
    if left["nodes"] != right["nodes"]:
        raise ValueError("different test catalogs")
    files = {}
    for filename in left["files"].keys() & right["files"].keys():
        a, b = left["files"][filename], right["files"][filename]
        files[filename] = {
            key: "*"
            if a[key] == "*" or b[key] == "*"
            else sorted(set(a[key]) | set(b[key]))
            for key in a.keys() & b.keys()
        }
    data = {
        "nodes": left["nodes"],
        "files": files,
        "complete": sorted(set(left["complete"]) & set(right["complete"])),
    }
    dest.write_bytes(
        gzip.compress(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        merge_profiles(*(Path(arg) for arg in sys.argv[2:5]))
    else:
        build(
            *(Path(arg) for arg in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:]
        )
