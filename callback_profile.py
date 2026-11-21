"""Observe calls and bytecode readers of deferred callbacks on pristine SQLGlot."""

import ast
from collections import defaultdict
from functools import cache
import gzip
import json
from pathlib import Path
import sys
import types

from exception_profile import GuardProfile


class CallbackProfile(GuardProfile):
    filename = "callbacks.json.gz"

    def candidates(self, path, syntax, new, blocks):
        rows = self.sites.get(path)
        if not rows or len(blocks) != 1:
            return None
        first, removed, added = blocks[0]
        if not removed or not added:
            return None
        end = first + len(removed) - 1
        possible = [node for node in syntax.nodes.get(first, ())
                    if isinstance(node, ast.Lambda) and node.end_lineno >= end
                    and str(node.lineno) in rows]
        if not possible:
            return None
        updated = ast.parse(new)
        fresh = [node for node in ast.walk(updated) if isinstance(node, ast.Lambda)]
        for before in reversed(possible):
            for after in fresh:
                if before.lineno != after.lineno or before.col_offset != after.col_offset:
                    continue
                # Replacing precisely the deferred expression must reconstruct the
                # pristine module. Arguments, defaults, wrappers and eager code stay intact.
                previous = after.body
                after.body = before.body
                same = ast.dump(syntax.tree) == ast.dump(updated)
                after.body = previous
                if same:
                    ids = rows[str(before.lineno)]
                    return self.unknown | {self.numbers[i] for i in ids if self.numbers[i] is not None}
        return None


def build(source, catalog, dest, reverse=False):
    import os
    import pytest
    from coverage_build import Collector

    root = source.resolve()
    dest = dest.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_text())["tests"]]
    collector = Collector(root, nodes, reverse=reverse)
    seen_modules = set()
    file_paths = {}
    masks = defaultdict(dict)
    for source_file in (root / "sqlglot").rglob("*.py"):
        rel = str(source_file.relative_to(root))
        file_paths[str(source_file)] = rel
        for node in ast.walk(ast.parse(source_file.read_bytes())):
            if isinstance(node, ast.Lambda):
                masks[rel][node.lineno] = 0

    @cache
    def target(code):
        rel = file_paths.get(code.co_filename)
        if rel and code.co_name == "<lambda>" and code.co_firstlineno in masks[rel]:
            return rel, code.co_firstlineno
        return None

    def observe(site):
        rel, line = site
        if masks[rel][line] < 0:
            return
        mask = collector.mask
        if mask:
            # A callback evaluated during an import can create state consumed
            # by a different test hours later. Skip such callbacks entirely.
            frame = sys._getframe(1)
            while frame:
                code = frame.f_code
                if code.co_name == "<module>" and code.co_filename in file_paths:
                    mask = 0
                    break
                frame = frame.f_back
        masks[rel][line] = (masks[rel][line] | mask) if mask else -1

    def entered(code, offset):
        if code.co_name == "<module>" and code.co_filename in file_paths:
            seen_modules.add(file_paths[code.co_filename])
            sys.monitoring.restart_events()
        site = target(code)
        if site is not None:
            observe(site)
        return sys.monitoring.DISABLE

    inspecting = False

    def audit(event, args):
        nonlocal inspecting
        if event == "object.__getattr__" and args[1] == "__code__":
            fn = args[0]
            if not inspecting and isinstance(fn, types.FunctionType):
                inspecting = True
                try:
                    site = target(fn.__code__)
                    if site is not None:
                        observe(site)
                finally:
                    inspecting = False
        elif event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
            mask = collector.mask
            if mask:
                collector.failed.update(i for i in range(len(nodes)) if mask & (1 << i))

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool, "deferred callback consumers")
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, entered)
    sys.monitoring.set_events(tool, sys.monitoring.events.PY_START)
    sys.addaudithook(audit)
    try:
        status = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[collector])
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.free_tool_id(tool)
    if status not in (0, 1) or not collector.passed:
        raise RuntimeError(f"pytest exited with {status}")
    files = {rel: {str(line): [i for i in range(len(nodes)) if mask & (1 << i)]
                   for line, mask in rows.items() if mask >= 0}
             for rel, rows in masks.items() if rel in seen_modules}
    data = {"nodes": nodes, "complete": sorted(collector.passed - collector.failed), "files": files}
    dest.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))


def merge(first, second, destination):
    left, right = [json.loads(gzip.decompress(path.read_bytes())) for path in (first, second)]
    if left["nodes"] != right["nodes"]:
        raise ValueError("different test catalogs")
    files = {}
    for path in left["files"].keys() & right["files"].keys():
        a, b = left["files"][path], right["files"][path]
        files[path] = {key: sorted(set(a[key]) | set(b[key])) for key in a.keys() & b.keys()}
    data = {"nodes": left["nodes"], "files": files,
            "complete": sorted(set(left["complete"]) & set(right["complete"]))}
    destination.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        merge(*(Path(arg) for arg in sys.argv[2:]))
    else:
        build(*(Path(arg) for arg in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:])
