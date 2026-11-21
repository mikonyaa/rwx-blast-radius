"""Run short candidate suites against an isolated copy-on-write source tree."""

import itertools
import os
from pathlib import Path
import select
import signal
import sys
import tempfile
import time
import unittest

import query_probe


def _overlay(base, target, paths):
    """Mirror only modified directories. Unchanged subtrees stay read-only links."""
    edits = {}
    for path, content in paths.items():
        head, separator, rest = path.partition("/")
        if not head or head in (".", ".."):
            raise ValueError("expected a repository-relative path")
        if separator:
            edits.setdefault(head, {})[rest] = content
        else:
            edits[head] = content

    names = set(edits)
    if base.is_dir():
        names.update(os.listdir(base))
    for name in names:
        src = base / name
        dst = target / name
        if name not in edits:
            if name != ".git":
                dst.symlink_to(src)
        elif isinstance(edits[name], dict):
            dst.mkdir()
            _overlay(src, dst, edits[name])
        elif edits[name] is not None:
            content = edits[name]
            dst.write_bytes(content if isinstance(content, bytes) else content.encode())
            # Ignore timestamp-validation pyc files from the unchanged source.
            if src.is_file():
                stamp = int(src.stat().st_mtime) + 4
                os.utime(dst, (stamp, stamp))


class _Result(unittest.TestResult):
    def __init__(self, numbers):
        super().__init__()
        self.numbers = numbers
        self.passed = []
        self.broken_fixture = False

    def addSuccess(self, test):
        self.passed.append(self.numbers[test.id()])

    def addError(self, test, err):
        if test.id() not in self.numbers:
            self.broken_fixture = True
        super().addError(test, err)


def _run_cases(records, notify):
    # Grouping honors unittest's class/module setup and teardown. Report a pass
    # only once the entire module has completed, including its teardown.
    records = sorted(records, key=lambda record: (record["file"], record["number"]))
    for _, group in itertools.groupby(records, key=lambda record: record["file"]):
        cases = []
        numbers = {}
        for record in group:
            filename, cls, method = record["id"].split("::")
            name = ".".join([filename[:-3].replace("/", "."), cls, method])
            cases.append(name)
            numbers[name] = record["number"]
        suite = unittest.defaultTestLoader.loadTestsFromNames(cases)
        result = _Result(numbers)
        result.failfast = True
        suite.run(result)
        if not result.broken_fixture:
            for number in result.passed:
                os.write(notify, f"{number}\n".encode())
        if not result.wasSuccessful():
            break


def _probe_child(view, records, shards, notify, cpu_limit):
    os.setsid()
    signal.setitimer(signal.ITIMER_PROF, cpu_limit)
    null = os.open(os.devnull, os.O_WRONLY)
    for fd in (1, 2):
        os.dup2(null, fd)
    os.close(null)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["SKIP_INTEGRATION"] = "0"
    os.chdir(view)
    sys.path.insert(0, str(view))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(view), os.environ.get("PYTHONPATH", "")])
    # Never silently run pristine modules imported earlier by the main selector.
    if "sqlglot" in sys.modules or "tests" in sys.modules:
        return
    if shards:
        for number in query_probe.probe_rows(shards):
            os.write(notify, f"{number}\n".encode())
    if records:
        _run_cases(records, notify)


def _confirmed_passes(view, records, shards, limit, cpu_limit):
    reader, writer = os.pipe()
    pid = os.fork()
    if not pid:
        os.close(reader)
        try:
            _probe_child(view, records, shards, writer, cpu_limit)
        finally:
            os._exit(0)

    os.close(writer)
    deadline = time.monotonic() + limit
    buffer = bytearray()
    finished = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([reader], [], [], remaining)[0]:
                break
            block = os.read(reader, 8192)
            if not block:
                finished = True
                break
            buffer.extend(block)
    finally:
        if not finished:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os.waitpid(pid, 0)
        os.close(reader)
    return {int(number) for number in buffer.splitlines() if number.isdigit()}


def refine(source_root, replacements, candidates, tests, budget=None):
    """Remove tests witnessed passing on this patch; uncertainty keeps the test.

    tests is a sequence of the official numbered records (or a number→record
    mapping). The automatic gate reserves probes for short Python changes where
    eliminating the final worker can earn its fixed 2-second startup cost.
    """
    chosen = set(candidates)
    if not chosen or not replacements or budget == 0:
        return chosen
    if any(content is None for content in replacements.values()):
        return chosen
    if not all(path.endswith(".py") for path in replacements):
        return chosen
    try:
        shards = query_probe.plan(source_root, replacements, chosen, tests)
        # An untouched query set is already proved by pristine per-query coverage.
        chosen.difference_update(number for number, _, indices in shards if not indices)
        specialized = {number for number, _, _ in shards}
        shards = [shard for shard in shards if shard[2]]
        records = [tests[number] for number in chosen if number not in specialized]
        seconds = sum(record["seconds"] for record in records)
        if budget is None:
            # Initializing DuckDB and the fixture worker pool costs more than
            # these small selections usually save. Their TPC shards are handled
            # above and count separately from ordinary testcase probes.
            costly_setup = any(r["file"] in ("tests/test_executor.py", "tests/test_optimizer.py")
                               for r in records)
            if costly_setup or seconds > 0.85 or len(records) > 120:
                records = []
            # The judge charges CPU. A batch competing for a time slice on an
            # emulator should not abandon a fast check merely because of wall time.
            cpu_limit = 2.6 if shards else 1.0
            budget = 5.0 if shards else 3.0
        else:
            cpu_limit = budget
        if not records and not shards:
            return chosen
        with tempfile.TemporaryDirectory(prefix="rwx-probe-") as folder:
            view = Path(folder)
            _overlay(Path(source_root).resolve(), view, replacements)
            return chosen - _confirmed_passes(view, records, shards, budget, cpu_limit)
    except (OSError, TypeError, KeyError, ValueError):
        return chosen
