"""Select and schedule SQLGlot tests for a batch of independent patches."""
import argparse
import json
from pathlib import Path
import sys
import time

from collection_guard import CollectionGuard
from impact import Impact
from packing import pack
from padding import Padding
from probe import refine
import optimizer_replay
from token_boundary import refine_patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--source", type=Path, default=Path("/opt/sqlglot"))
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--without-probes", action="store_true")
    args = parser.parse_args()

    tests = json.loads((args.data / "tests.json").read_bytes())["tests"]
    by_number = {test["number"]: test for test in tests}
    seconds = {number: row["seconds"] for number, row in by_number.items()}
    all_tests = set(by_number)
    patches = sorted((args.data / "commits").glob("*.patch"))
    start = time.monotonic()
    # Leave enough time to emit the rest of the batch if a patch is unusually expensive.
    limit = 168
    try:
        impact = Impact(args.source, tests, args.coverage)
        padding = Padding(impact)
        guard = CollectionGuard(impact)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Using full suite: could not load baseline maps ({exc})", file=sys.stderr)
        impact = None

    for index, path in enumerate(patches):
        chosen = all_tests
        patch = ""
        time_left = start + limit - time.monotonic()
        if impact is not None and time_left > max(2, (len(patches) - index) * 0.05):
            try:
                patch = path.read_text()
                analysis = impact.analyze(patch)
                chosen = analysis.tests
                if not args.without_probes and time_left > 6:
                    chosen = refine_patch(impact, patch, analysis, chosen, by_number)
                    chosen = refine(args.source, analysis.paths, chosen, by_number)
                    if start + limit - time.monotonic() > 10:
                        chosen = optimizer_replay.refine(args.source, analysis.paths, chosen, by_number)
                if time_left > 9:
                    chosen = guard.augment(patch, analysis, chosen)
            except (OSError, ValueError, SyntaxError, KeyError, IndexError, RecursionError) as exc:
                print(f"Using full suite for {path.stem}: {type(exc).__name__}", file=sys.stderr)
                chosen = all_tests
        workers = pack(chosen, seconds)
        if impact is not None and len(chosen) != len(all_tests):
            workers = padding.extend(workers, patch)
        print(json.dumps({"commit": path.stem, "workers": workers}, separators=(",", ":")))


if __name__ == "__main__":
    main()
