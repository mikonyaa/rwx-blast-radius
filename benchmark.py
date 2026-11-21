"""Build the submission and score three restricted sample runs, including container CPU."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

RUN_FLAGS = [
    "--rm", "--platform", "linux/amd64", "--network", "none", "--cpus", "2",
    "--memory", "4096m", "--memory-swap", "4096m", "--pids-limit", "256",
    "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges", "--user", "65534:65534", "--env", "HOME=/tmp",
]
# Read cgroup usage at program shutdown. This includes children and the interpreter's
# startup within this cgroup, with a small unmeasured interval before container exit.
CPU_WRAPPER = """from pathlib import Path
import runpy, sys
sys.argv = ['/app/selector.py', *sys.argv[1:]]
sys.path.insert(0, '/app')
runpy.run_path(sys.argv[0], run_name='__main__')
stats = dict(row.split() for row in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
print('measured_cpu_usec=' + stats['usage_usec'], file=sys.stderr)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kit", type=Path, help="unpacked official starter directory")
    parser.add_argument("--image", default="rwx-local:entry")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--without-probes", action="store_true", help="benchmark coverage selection alone")
    args = parser.parse_args()
    kit = args.kit.resolve()
    here = Path(__file__).resolve().parent
    if not args.skip_build:
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", args.image, str(here)], check=True)
    mounts = []
    for name in ("commits", "tests.json"):
        mounts.extend(["-v", f"{kit / 'sample' / name}:/data/{name}:ro"])
    extra = ["--without-probes"] if args.without_probes else []
    scores = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.runs):
            run = subprocess.run(["docker", "run", *RUN_FLAGS, *mounts, args.image, "python", "-c", CPU_WRAPPER, *extra],
                                 capture_output=True, timeout=182, check=True)
            messages = run.stderr.decode().splitlines()
            usec = int(next(line.split("=")[1] for line in messages if line.startswith("measured_cpu_usec=")))
            for line in messages:
                if not line.startswith("measured_cpu_usec="):
                    print(line, file=sys.stderr)
            predictions = Path(tmp) / "result.jsonl"
            predictions.write_bytes(run.stdout)
            result = subprocess.check_output([sys.executable, str(kit / "score.py"), str(kit / "sample"),
                                             str(kit / "sample/answers.json"), str(predictions),
                                             "--cpu-seconds", str(usec / 1e6)], text=True)
            print(f"Run {i+1} (CPU {usec / 1e6:.3f} s)\n{result}", flush=True)
            total = next(line for line in result.splitlines() if line.startswith("Total:"))
            scores.append(Decimal(total.split()[1]))
    median = statistics.median(scores)
    print(f"Median sample total: {median:.3f} s, rounded {median.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)} s")


if __name__ == "__main__":
    main()
