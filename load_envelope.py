"""Observe which SQLGlot source files a pristine test module loads in a cold process."""
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def data_field_change(owner, before, after):
    """Narrow only isolated default-value edits on plain dataclasses."""
    if owner.bases or owner.keywords or len(owner.decorator_list) != 1:
        return False
    decorator = owner.decorator_list[0]
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if not isinstance(decorator, ast.Name) or decorator.id != "dataclass":
        return False
    if not isinstance(after, ast.AnnAssign) or before.value is None or after.value is None:
        return False
    if ast.dump(before.target) != ast.dump(after.target) or ast.dump(before.annotation) != ast.dump(after.annotation):
        return False
    if isinstance(before.value, ast.Constant) and isinstance(after.value, ast.Constant):
        return type(before.value.value) is type(after.value.value)

    def simple_factory(expr):
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "field":
            return False
        if expr.args or len(expr.keywords) != 1:
            return False
        kw = expr.keywords[0]
        return kw.arg == "default_factory" and isinstance(kw.value, ast.Name) and kw.value.id in {
            "dict", "list", "set", "tuple", "str", "int", "float", "bool", "bytes",
        }

    return simple_factory(before.value) and simple_factory(after.value)


class LoadEnvelope:
    def __init__(self, directory, catalog):
        self.unknown = set()
        self.by_path = defaultdict(list)
        path = Path(directory) / "load_envelope.json.gz"
        if not path.is_file():
            return
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        if data.get("version") != 1:
            return
        by_file = defaultdict(set)
        for record in catalog:
            by_file[record["file"]].add((record["id"], record["number"]))
        for filename, rows in by_file.items():
            record = data["files"].get(filename)
            observed = set(record["passed"]) if record else set()
            verified = {number for node, number in rows if node in observed}
            self.unknown.update(number for node, number in rows if node not in observed)
            if record:
                for module in record["read"]:
                    self.by_path[module].append(verified)

    @lru_cache(maxsize=128)
    def users(self, path):
        """None means no complete profile observed this path."""
        if path not in self.by_path:
            return None
        return self.unknown.union(*self.by_path[path])


def observe(root, filename, report_path):
    os.chdir(root)
    sys.path.insert(0, str(root))
    prefix = str(root / "sqlglot") + os.sep
    observed = set()
    external = False

    def audit(event, args):
        nonlocal external
        if event in {"subprocess.Popen", "os.fork", "os.forkpty", "os.posix_spawn",
                     "os.system", "os.exec", "os.spawn"}:
            external = True
        if event == "exec":
            path = getattr(args[0], "co_filename", None)
        elif event == "open":
            path = args[0]
        else:
            return
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str) or not path.endswith((".py", ".pyc")):
            return
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if path.startswith(prefix):
            if path.endswith(".pyc"):
                import importlib.util
                try:
                    path = importlib.util.source_from_cache(path)
                except ValueError:
                    path = path[:-1]
            observed.add("sqlglot/" + path[len(prefix):])

    sys.addaudithook(audit)
    # multiprocessing's spawn backend can call the C fork_exec primitive directly
    # without emitting subprocess.Popen or os.fork on Python 3.13.
    from multiprocessing.process import BaseProcess
    original_start = BaseProcess.start

    def track_start(process):
        nonlocal external
        external = True
        return original_start(process)

    BaseProcess.start = track_start
    import pytest

    class Reports:
        def __init__(self):
            self.passed = set()
            self.incomplete = False

        def pytest_runtest_logreport(self, report):
            if report.failed or report.skipped:
                self.incomplete = True
            elif report.when == "call" and report.passed:
                self.passed.add(report.nodeid)

    reporter = Reports()
    result = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", filename], plugins=[reporter])
    if result == 0 and not external and not reporter.incomplete:
        report_path.write_text(json.dumps({"passed": sorted(reporter.passed), "read": sorted(observed)}))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=80)
    args = parser.parse_args()
    source = args.source.resolve()
    files = defaultdict(set)
    for record in json.loads(args.catalog.read_bytes())["tests"]:
        files[record["file"]].add(record["id"])
    script = Path(__file__).resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source),
               PYTHON_CPU_COUNT="2", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               SKIP_INTEGRATION="0", PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])

    def run(filename):
        with tempfile.TemporaryDirectory(prefix="rwx-load-") as folder:
            report = Path(folder) / "result.json"
            try:
                process = subprocess.run([sys.executable, str(script), "--observe", str(source),
                                          filename, str(report)], cwd=source, env=env, timeout=args.timeout,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return filename, None
            if process.returncode or not report.is_file():
                return filename, None
            data = json.loads(report.read_text())
            if not files[filename].issubset(data["passed"]):
                return filename, None
            return filename, data

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(pool.map(run, sorted(files)))
    complete = {key: val for key, val in results.items() if val is not None}
    payload = json.dumps({"version": 1, "files": complete}, sort_keys=True, separators=(",", ":"))
    args.output.write_bytes(gzip.compress(payload.encode(), mtime=0))
    print(f"Completed {len(complete)} of {len(files)} independent test files", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--observe"]:
        observe(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    else:
        main()
