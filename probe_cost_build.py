"""Measure pristine unittest CPU to estimate the cost of replaying test suites."""

import argparse
import concurrent.futures
import gzip
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import unittest


def cpu_time():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def measure(root, catalog, module, output):
    rows = [row for row in json.loads(catalog.read_bytes())["tests"]
            if row["file"] == module and row["seconds"] <= 4.0]
    os.chdir(root)
    sys.path.insert(0, str(root))
    names = {
        row["id"].replace(".py::", ".").replace("::", ".").replace("/", "."): row["id"]
        for row in rows
    }
    durations = {}
    successes = set()
    local_test_cpu = 0.0
    submissions = 0
    original_submit = concurrent.futures.ProcessPoolExecutor.submit

    def observed_submit(pool, *args, **kwargs):
        nonlocal submissions
        submissions += 1
        return original_submit(pool, *args, **kwargs)

    class Timing(unittest.TestResult):
        def startTest(self, test):
            super().startTest(test)
            self.start_cpu = cpu_time()
            self.start_local = time.process_time()
            self.start_wall = time.monotonic()
            self.start_submissions = submissions

        def addSuccess(self, test):
            successes.add(test.id())
            super().addSuccess(test)

        def stopTest(self, test):
            nonlocal local_test_cpu
            local_test_cpu += time.process_time() - self.start_local
            elapsed = cpu_time() - self.start_cpu
            # The OS reports persistent workers' CPU only after pool shutdown.
            # Use the observed wait for a case that handed work to the pool.
            if self.start_submissions != submissions:
                elapsed = max(elapsed, time.monotonic() - self.start_wall)
            durations[test.id()] = max(0.00003, elapsed)
            super().stopTest(test)

    concurrent.futures.ProcessPoolExecutor.submit = observed_submit
    try:
        start = cpu_time()
        suite = unittest.defaultTestLoader.loadTestsFromNames(list(names))
        import_cpu = cpu_time() - start
        result = Timing()
        start = time.process_time()
        suite.run(result)
        setup_cpu = max(0, time.process_time() - start - local_test_cpu)
    finally:
        concurrent.futures.ProcessPoolExecutor.submit = original_submit

    payload = {
        "import": round(import_cpu, 5),
        "setup": round(setup_cpu, 5),
        "tests": {names[name]: round(durations[name], 5) for name in successes & names.keys()},
    }
    output.write_text(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Pristine pinned SQLGlot checkout")
    parser.add_argument("catalog", type=Path, help="Public tests.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--module", help="Measure one module and write uncompressed JSON")
    args = parser.parse_args()
    root, catalog, output = args.root.resolve(), args.catalog.resolve(), args.output.resolve()
    if args.module:
        measure(root, catalog, args.module, output)
        return

    rows = json.loads(catalog.read_bytes())["tests"]
    modules = sorted({row["file"] for row in rows})
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       SKIP_INTEGRATION="0", PYTHON_CPU_COUNT="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root))
    costs = {}
    with tempfile.TemporaryDirectory() as directory:
        for index, module in enumerate(modules):
            result_file = Path(directory) / f"module-{index}.json"
            command = [sys.executable, __file__, str(root), str(catalog), str(result_file),
                       "--module", module]
            try:
                process = subprocess.run(command, env=environment, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, timeout=240)
            except subprocess.TimeoutExpired:
                continue
            if process.returncode == 0 and result_file.is_file():
                costs[module] = json.loads(result_file.read_bytes())
    payload = json.dumps(costs, separators=(",", ":"), sort_keys=True).encode()
    output.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
