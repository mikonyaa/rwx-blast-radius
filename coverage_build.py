"""Build line and read coverage from pristine SQLGlot. Requires CPython 3.13 and the kit's dependencies."""

import argparse
import base64
from collections import defaultdict
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from functools import cache, wraps
import gzip
import json
from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


class LocalExecutor(Executor):
    """Exercise executor and optimizer workers synchronously so their coverage is attributed."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer:
            initializer(*ForkingPickler.loads(ForkingPickler.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        result = Future()
        try:
            fn, args, kwargs = ForkingPickler.loads(ForkingPickler.dumps((fn, args, kwargs)))
            value = fn(*args, **kwargs)
            result.set_result(ForkingPickler.loads(ForkingPickler.dumps(value)))
        except Exception as error:
            result.set_exception(error)
        return result


def encode(mask):
    return base64.b64encode(mask.to_bytes((mask.bit_length() + 7) // 8, "little")).decode()


def trace_in_fresh_python(source, output, command):
    """Run a standalone -c check before SQLGlot has been imported."""
    lines = defaultdict(set)
    prefix = str(Path(source).resolve()) + os.sep

    def line_event(code, line):
        if code.co_filename.startswith(prefix):
            lines[code.co_filename[len(prefix):]].add(line)
        return sys.monitoring.DISABLE

    tool = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tool, "child coverage")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line_event)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)
    sys.path.insert(0, os.getcwd())
    sys.argv[:] = ["-c"]
    try:
        exec(compile(command, "<string>", "exec"), {"__name__": "__main__"})
    finally:
        sys.monitoring.set_events(tool, 0)
        Path(output).write_text(json.dumps({k: sorted(v) for k, v in lines.items()}))


class Collector:
    def __init__(self, root, nodes, semantic=None, reverse=False):
        self.reverse = reverse
        self.root = str(root) + os.sep
        self.nodes = nodes
        self.indices = {node: i for i, node in enumerate(nodes)}
        self.semantic = semantic
        self.lines = defaultdict(dict)
        self.reads = {}
        self.passed = set()
        self.failed = set()
        self.mask = 0

    @cache
    def filename(self, path):
        if path.startswith(self.root):
            return path[len(self.root):]
        return None

    def on_line(self, code, line):
        filename = self.filename(code.co_filename)
        if filename and self.mask:
            rows = self.lines[filename]
            rows[line] = rows.get(line, 0) | self.mask
        return sys.monitoring.DISABLE

    def on_audit(self, event, args):
        if event != "open" or not self.mask:
            return
        path, mode, _ = args
        if not isinstance(path, str) or not isinstance(mode, str) or not mode.startswith("r"):
            return
        # Code imports are tracked through lines, not a read attributed to the first importer.
        if path.endswith((".pyc", ".pyo")):
            return
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        filename = self.filename(path)
        if filename is None:
            return
        if filename.endswith(".py") and sys._getframe(1).f_code.co_name == "get_data":
            return
        self.reads[filename] = self.reads.get(filename, 0) | self.mask

    def set_context(self, mask, nodeid=None):
        self.mask = mask
        if self.semantic is not None:
            if nodeid is None and mask:
                nodeid = tuple(node for i, node in enumerate(self.nodes) if mask & (1 << i))
            self.semantic.active = nodeid
        sys.monitoring.restart_events()

    @contextmanager
    def context(self, mask, nodeid=None):
        old_mask = self.mask
        old_id = self.semantic.active if self.semantic is not None else None
        self.set_context(mask, nodeid)
        try:
            yield
        finally:
            self.set_context(old_mask, old_id)

    def pytest_collection_modifyitems(self, items):
        items[:] = [item for item in items if item.nodeid in self.indices]
        if self.reverse:
            items.reverse()
        class_masks = defaultdict(int)
        for item in items:
            if item.cls:
                class_masks[item.cls] |= 1 << self.indices[item.nodeid]
            if hasattr(item.module, "ProcessPoolExecutor"):
                item.module.ProcessPoolExecutor = LocalExecutor
        # setUpClass runs only for the first test but its effects are required by every class member.
        for cls, mask in class_masks.items():
            for method in ("setUpClass", "tearDownClass"):
                if method not in vars(cls):
                    continue
                bound = getattr(cls, method)
                setattr(cls, method, classmethod(self.fixture_wrapper(bound, mask)))

    def fixture_wrapper(self, fn, mask):
        @wraps(fn)
        def run(cls):
            with self.context(mask):
                return fn()
        return run

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        idx = self.indices[item.nodeid]
        with self.context(1 << idx, item.nodeid):
            yield

    def pytest_runtest_logreport(self, report):
        if report.nodeid in self.indices:
            idx = self.indices[report.nodeid]
            if report.failed:
                self.failed.add(idx)
            if report.when == "call" and report.passed:
                self.passed.add(idx)

    def write(self, output):
        payload = {
            "nodes": self.nodes,
            "lines": {
                name: {str(n): encode(mask) for n, mask in sorted(rows.items())}
                for name, rows in sorted(self.lines.items())
            },
            "reads": {name: encode(mask) for name, mask in sorted(self.reads.items())},
            "incomplete": sorted(set(range(len(self.nodes))) - self.passed | self.failed),
        }
        with output.open("wb") as file:
            with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
                compressed.write(json.dumps(payload, separators=(",", ":")).encode())
        print(f"wrote {len(payload['lines'])} files, {len(payload['reads'])} reads, "
              f"{len(payload['incomplete'])} incomplete tests", file=sys.stderr)

    def with_child_coverage(self, run):
        def wrapped(args, *pos, **kwargs):
            if (not self.mask or not isinstance(args, (list, tuple)) or len(args) != 3
                    or Path(args[0]).name != "python" or args[1] != "-c"):
                return run(args, *pos, **kwargs)
            with tempfile.TemporaryDirectory() as directory:
                result_file = Path(directory) / "lines.json"
                child = [sys.executable, __file__, "--child", self.root, str(result_file), args[2]]
                result = run(child, *pos, **kwargs)
                if result_file.exists():
                    for name, rows in json.loads(result_file.read_text()).items():
                        dst = self.lines[name]
                        for line in rows:
                            dst[line] = dst.get(line, 0) | self.mask
                return result
        return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--only", nargs="+", help="restrict a development profile to pytest paths or nodes")
    parser.add_argument("--semantic", type=Path, help="add optional import-time config instrumentation")
    parser.add_argument("--reverse", action="store_true", help="collect in reverse order to vary lazy cache ownership")
    args = parser.parse_args()
    root = args.source.resolve()
    nodes = [row["id"] for row in json.loads(args.catalog.read_text())["tests"]]
    output = args.output.resolve()
    semantic_output = args.semantic.resolve() if args.semantic else None
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sys.path.insert(0, str(root))
    os.chdir(root)
    semantic = None
    if semantic_output:
        import semantic
        semantic.install()
    collector = Collector(root, nodes, semantic, args.reverse)
    subprocess.run = collector.with_child_coverage(subprocess.run)
    tool = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tool, "pinned coverage")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, collector.on_line)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)
    sys.addaudithook(collector.on_audit)
    try:
        status = pytest.main(["-q", "--disable-warnings", "--tb=short", "-p", "no:cacheprovider",
                              *(args.only or ["tests"])], plugins=[collector])
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.free_tool_id(tool)
    if not collector.passed or status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        raise SystemExit(f"could not collect usable coverage, pytest status {status}")
    collector.write(output)
    if semantic_output:
        semantic.save(semantic_output)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        trace_in_fresh_python(*sys.argv[2:])
    else:
        main()
