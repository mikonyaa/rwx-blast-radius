"""Capture each pristine SQLGlot module's eager imports in a fresh interpreter."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def qualified(path):
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def observed(source, name):
    sys.path.insert(0, str(source))
    importlib.import_module(name)
    files = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not isinstance(path, str):
            continue
        path = Path(path)
        if path.is_relative_to(source):
            files.add(str(path.relative_to(source)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    source = args.source.resolve()
    files = sorted(p.relative_to(source) for p in (source / "sqlglot").rglob("*.py"))

    def observe(path):
        name = qualified(path)
        process = subprocess.run([sys.executable, __file__, "--observe", str(source), name],
                                 capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return name, json.loads(process.stdout) if process.returncode == 0 else None

    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        closures = dict(workers.map(observe, files))
    unresolved = sorted(name for name, deps in closures.items() if deps is None)
    target = args.output.resolve()
    data = json.dumps({"closures": closures, "unresolved": unresolved}, separators=(",", ":"), sort_keys=True)
    target.write_bytes(gzip.compress(data.encode(), mtime=0))
    print(f"{len(closures)} modules, {len(unresolved)} standalone imports unsupported")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--observe"]:
        print(json.dumps(observed(Path(sys.argv[2]), sys.argv[3])))
    else:
        main()
