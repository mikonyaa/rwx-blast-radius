# RWX Blast Radius

A predictive test selector for SQLGlot at `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. The entry reads the contest test catalog and one unified diff per independent change. It prints one JSON object per change.

## Run the official sample

Download the [organizers' starter archive](https://www.joinbuilders.org/buildathon/tracks/blast-radius/files/blast-radius-sample.tar.gz) and extract it into an empty directory. From this repository run:

```sh
python3 benchmark.py /absolute/path/to/unpacked-starter
```

This builds the pinned `linux/amd64` image, starts three fresh containers with the contest resource limits, and calls the official scorer on each result. The benchmark reads CPU usage from cgroup v2; on Apple Silicon the container runs under emulation and timings will differ from the judging machine. Sample answers stay on the host. The container mounts only the catalog and the patch directory.

## How selection works

1. Apply each patch to its own view of the pinned files. Locate the changed old-side statements and inspect both versions of the syntax where needed.
2. Select tests that exercised those code paths or read a changed fixture. Account for class setup, SQLGlot's process-pool serialization, and subprocess imports. For short-circuit edits, look at the earlier operands that determined whether execution reached the changed operand.
3. For supported class declarations, use measured access to the baseline attribute, expression schema, or parser dispatch entry. For a single comparison-operator change, evaluate the old and new operator on observed categories of operand values. Coverage maps contain baseline observations. They use pytest node IDs so changes to numeric catalog ordering are handled.
4. When economical, execute candidate tests in a fresh child against the patched source, and drop confirmed passes. The parent preserves candidates when validation is inconclusive or its time budget expires. An unsupported change can fall back to the full suite.
5. Compare worker counts from one to eight. For each promising count, place longest tests first on the least loaded worker and refine small assignments with a bounded search. The objective includes the 2-second cost of each worker.

## Contents

- `selector.py`, `impact.py` and `packing.py` implement the contest entry point, selection and scheduling.
- `probe.py` and `query_probe.py` supply budgeted runtime verification.
- `coverage_build.py`, `semantic.py` and `operand_build.py` collect baseline observations. `comparison.py` interprets operand profiles.
- `maps/` contains the baseline profiles used by the entry. The profile scripts take a clean SQLGlot tree and a public test catalog. The source tarball and dependencies in the Dockerfile are pinned.

Unseen changes can exercise behavior absent from a baseline trace. Misses and CPU time on the organizers' hidden runs determine the official score. The practice leaderboard and final leaderboard are independent evaluations.
