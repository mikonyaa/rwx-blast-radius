"""Extend pristine module-load profiles through real Python multiprocessing spawn workers."""
from collections import defaultdict
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def instrument():
    """Called by sitecustomize in the test interpreter and each of its spawn workers."""
    from multiprocessing.process import BaseProcess

    directory = Path(os.environ["RWX_IMPORT_REPORT"])
    root = os.environ["RWX_IMPORT_SOURCE"]
    prefix = root + "/sqlglot/"
    report = (directory / f"{os.getpid()}.events").open("a", buffering=1)
    paths = set()
    report.write("start\n")

    def emit(event, args):
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.fork",
                     "os.forkpty", "os.posix_spawn"):
            report.write("external\n")
        if event == "exec":
            name = getattr(args[0], "co_filename", None)
        elif event == "open":
            name = args[0]
        else:
            return
        if isinstance(name, bytes):
            name = os.fsdecode(name)
        if not isinstance(name, str) or not name.endswith((".pyc", ".py")):
            return
        name = os.path.abspath(name)
        if not name.startswith(prefix):
            return
        if name.endswith(".pyc"):
            try:
                name = importlib.util.source_from_cache(name)
            except ValueError:
                name = name[:-1]
        if name not in paths:
            paths.add(name)
            report.write("read " + os.path.relpath(name, root) + "\n")

    start = BaseProcess.start
    def observed(proc, *args, **kwargs):
        outcome = start(proc, *args, **kwargs)
        report.write(f"worker {proc.pid}\n")
        return outcome
    BaseProcess.start = observed
    if any(name == "sqlglot" or name.startswith("sqlglot.") for name in sys.modules):
        report.write("external\n")
    sys.addaudithook(emit)


def measure(source, target, expected):
    with tempfile.TemporaryDirectory(prefix="rwx-worker-imports-") as folder:
        report = Path(folder)
        (report / "sitecustomize.py").write_text("from profile_child_imports import instrument\ninstrument()\n")
        env = dict(os.environ, RWX_IMPORT_REPORT=folder, RWX_IMPORT_SOURCE=str(source),
                   SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHON_CPU_COUNT="2",
                   PYTHONPATH=os.pathsep.join((folder, str(Path(__file__).resolve().parent), str(source))))
        result = subprocess.run([sys.executable, __file__, "--run", target, str(report / "result.json")],
                                cwd=source, env=env, timeout=240, stdout=subprocess.DEVNULL)
        dest = report / "result.json"
        if result.returncode or not dest.is_file():
            return None
        outcome = json.loads(dest.read_text())
        starts = set()
        workers = {outcome["pid"]}
        reads = set()
        for events in report.glob("*.events"):
            rows = events.read_text().splitlines()
            if "external" in rows or rows.count("start") != 1:
                return None
            starts.add(int(events.stem))
            workers.update(int(line[7:]) for line in rows if line.startswith("worker "))
            reads.update(line[5:] for line in rows if line.startswith("read "))
        if not workers.issubset(starts) or not expected.issubset(outcome["passed"]):
            return None
        return {"read": sorted(reads), "passed": sorted(outcome["passed"])}


def run(target, path):
    import pytest
    class Reports:
        def __init__(self):
            self.passed = set()
            self.valid = True
        def pytest_runtest_logreport(self, report):
            if report.failed or report.skipped:
                self.valid = False
            elif report.when == "call" and report.passed:
                self.passed.add(report.nodeid)
    results = Reports()
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short", target], plugins=[results])
    if status == 0 and results.valid:
        path.write_text(json.dumps({"pid": os.getpid(), "passed": sorted(results.passed)}))


def build(source, catalog, base, output):
    files = defaultdict(set)
    for row in json.loads(catalog.read_bytes())["tests"]:
        files[row["file"]].add(row["id"])
    profile = json.loads(gzip.decompress(base.read_bytes()))
    for name in sorted(files.keys() - profile["files"].keys()):
        try:
            data = measure(source, name, files[name])
        except subprocess.TimeoutExpired:
            data = None
        if data:
            profile["files"][name] = data
            print("Verified", name, len(data["passed"]), "tests", len(data["read"]), "modules", flush=True)
        else:
            print("Kept fallback for", name, flush=True)
    raw = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    output.write_bytes(gzip.compress(raw,mtime=0))


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run(sys.argv[2],Path(sys.argv[3]))
    else:
        build(*(Path(v).resolve() for v in sys.argv[1:]))
