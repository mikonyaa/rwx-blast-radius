"""Run short candidate suites against an isolated copy-on-write source tree."""

from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import select
import signal
import sys
import tempfile
import time
import unittest

import query_probe
from packing import pack


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
        self.passed = set()
        self.subfailures = set()
        self.broken_fixture = False
        self.failfast = True

    def stopTest(self, test):
        super().stopTest(test)
        # End an assertion-heavy failing case immediately, but still check independent cases.
        if not self.broken_fixture:
            self.shouldStop = False

    def addSuccess(self, test):
        if test.id() not in self.subfailures:
            self.passed.add(self.numbers[test.id()])

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.subfailures.add(test.id())
            self.passed.discard(self.numbers[test.id()])
        super().addSubTest(test, subtest, err)

    def addError(self, test, err):
        if test.id() not in self.numbers:
            self.broken_fixture = True
        super().addError(test, err)


def _run_cases(records, notify):
    # Grouping honors unittest's class/module setup and teardown. Report a pass
    # only once the entire module has completed, including its teardown.
    grouped = {}
    for record in records:
        grouped.setdefault(record["file"], []).append(record)
    # Complete one module including its teardown before moving to the next.
    # Spend a limited child budget first where a whole-module pass frees most load.
    for _, group in sorted(grouped.items(), key=lambda pair: (-sum(r["seconds"] for r in pair[1]), pair[0])):
        cases = []
        numbers = {}
        for record in sorted(group, key=lambda row: row["number"]):
            filename, cls, method = record["id"].split("::")
            name = ".".join([filename[:-3].replace("/", "."), cls, method])
            cases.append(name)
            numbers[name] = record["number"]
        suite = unittest.defaultTestLoader.loadTestsFromNames(cases)
        result = _Result(numbers)
        suite.run(result)
        if not result.broken_fixture:
            for number in result.passed:
                os.write(notify, f"{number}\n".encode())
        if result.broken_fixture:
            # Another module may otherwise inherit incompletely restored process state.
            break


def _child(view, fn, args, notify, cpu_limit, base):
    os.setsid()
    signal.setitimer(signal.ITIMER_PROF, cpu_limit)
    null = os.open(os.devnull, os.O_WRONLY)
    for fd in (1, 2):
        os.dup2(null, fd)
    os.close(null)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["SKIP_INTEGRATION"] = "0"
    if base is not None:
        base = Path(base).resolve()
        sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != base]
    os.chdir(view)
    sys.path.insert(0, str(view))
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    inherited = [p for p in inherited if p and (base is None or Path(p).resolve() != base)]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(view), *inherited])
    # Never silently run pristine modules imported earlier by the main selector.
    if "sqlglot" in sys.modules or "tests" in sys.modules:
        return
    fn(*args, notify)


def _run_selected(records, shards, notify):
    if shards:
        for number in query_probe.probe_rows(shards):
            os.write(notify, f"{number}\n".encode())
    if records:
        _run_cases(records, notify)


def isolated_numbers(view, fn, args, limit, cpu_limit, base=None):
    reader, writer = os.pipe()
    pid = os.fork()
    if not pid:
        os.close(reader)
        try:
            _child(view, fn, args, writer, cpu_limit, base)
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
        # EOF also follows a hard CPU timeout in the group leader. A spawned
        # pool process need not inherit the pipe and may still be running.
        # Terminate the group before reaping the leader so its PID cannot be reused.
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            if not finished:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os.waitpid(pid, 0)
        os.close(reader)
    return {int(number) for number in buffer.split(b"\n")[:-1] if number.isdigit()}


def _schedule_cost(numbers, seconds):
    workers = pack(numbers, seconds)
    return 2 * len(workers) + max((sum(seconds[n] for n in w) for w in workers), default=0)


@lru_cache(maxsize=1)
def _timings():
    path = Path(__file__).with_name("maps") / "probe_timings.json.gz"
    if not path.is_file():
        return {}
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _expand_plan(records, baseline, chosen, specialized, tests):
    """Consider extra modules when replay CPU costs less than the projected CI saving."""
    if len(records) == len(baseline) or len(chosen) > 250:
        return baseline, 0
    times = _timings()
    groups = {}
    existing = {r["number"] for r in baseline}
    for row in records:
        measured = times.get(row["file"], {}).get("tests", {})
        if row["number"] not in existing and row["id"] in measured:
            groups.setdefault(row["file"], []).append(row)
    if not groups:
        return baseline, 0
    durations = {n: tests[n]["seconds"] for n in chosen}
    leftover = chosen - existing - specialized
    before = _schedule_cost(leftover, durations)
    already_imported = {r["file"] for r in baseline}
    options = list(groups.values())
    if len(groups) > 1:
        options.append([r for items in groups.values() for r in items])
    best = None
    for rows in options:
        if len(rows) > 150:
            continue
        modules = {r["file"] for r in rows} - already_imported
        # Modules share the main SQLGlot import. Estimate its cold cost once.
        cold = max((times[m]["import"] for m in modules), default=0)
        fixtures = sum(times[m]["setup"] for m in modules)
        estimate = cold + fixtures + sum(times[r["file"]]["tests"][r["id"]] for r in rows)
        if estimate > 1.0:
            continue
        gain = before - _schedule_cost(leftover - {r["number"] for r in rows}, durations)
        payoff = gain - 3.0 * estimate - 0.3
        if payoff > 0 and (best is None or payoff > best[0]):
            best = payoff, rows, estimate
    if best is None:
        return baseline, 0
    return baseline + best[1], best[2]


def refine(source_root, replacements, candidates, tests, budget=None, effective=None):
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
    try:
        planned_paths = replacements if effective is None else effective
        python_only = all(path.endswith(".py") for path in planned_paths)
        if python_only:
            shards = query_probe.plan(source_root, planned_paths, chosen, tests)
        elif len(planned_paths) == 1 and next(iter(planned_paths)).endswith(".sql"):
            shards = query_probe.fixture_plan(source_root, planned_paths, chosen, tests)
            if not shards:
                return chosen
        else:
            return chosen
        # An untouched query set is already proved by pristine per-query coverage.
        chosen.difference_update(number for number, _, indices in shards if not indices)
        specialized = {number for number, _, _ in shards}
        shards = [shard for shard in shards if shard[2]]
        records = [tests[number] for number in chosen if number not in specialized] if python_only else []
        if budget is None:
            # Initializing DuckDB and the fixture worker pool costs more than
            # these small selections usually save. Their TPC shards are handled
            # above and count separately from ordinary testcase probes.
            possible = records
            records = [r for r in records if r["file"] not in (
                "tests/test_executor.py", "tests/test_optimizer.py")]
            if sum(r["seconds"] for r in records) > 0.85 or len(records) > 120:
                records = []
            records, extra_cpu = _expand_plan(possible, records, chosen, specialized, tests)

            # Even confirming every planned test can leave the same bottleneck:
            # another long test still needs its worker. Only spend CPU on partial
            # probes when a successful result could pay for the work.
            planned = {r["number"] for r in records} | {n for n, _, _ in shards}
            if planned and chosen - planned:
                durations = {n: tests[n]["seconds"] for n in chosen}
                saving = _schedule_cost(chosen, durations) - _schedule_cost(chosen - planned, durations)
                if not extra_cpu and saving <= 0.15 + sum(r["seconds"] for r in records):
                    return chosen
            # The judge charges CPU. A batch competing for a time slice on an
            # emulator should not abandon a fast check merely because of wall time.
            cpu_limit = 2.6 if shards else 1.0 + extra_cpu
            budget = 5.0 if shards else 3.0
        else:
            cpu_limit = budget
        if not records and not shards:
            return chosen
        with tempfile.TemporaryDirectory(prefix="rwx-probe-") as folder:
            view = Path(folder)
            _overlay(Path(source_root).resolve(), view, replacements)
            return chosen - isolated_numbers(view, _run_selected, (records, shards), budget, cpu_limit, source_root)
    except (OSError, TypeError, KeyError, ValueError):
        return chosen
