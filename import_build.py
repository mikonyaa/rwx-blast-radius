"""Profile imports of each test module separately on the pinned, unmodified checkout."""

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def observe(source, module):
    os.chdir(source)
    sys.path.insert(0, source)
    prefix = source + os.sep
    calls = defaultdict(set)

    def observe_line(code, line):
        # Class and module bodies execute at import. Their users are represented by
        # the separate import map. Functions can also compute constants at import.
        if code.co_flags & inspect.CO_OPTIMIZED and code.co_filename.startswith(prefix):
            calls[code.co_filename[len(prefix):]].add(line)
        return sys.monitoring.DISABLE

    tool = sys.monitoring.COVERAGE_ID
    if module is None:
        from sqlglot import Dialect
    sys.monitoring.use_tool_id(tool, "import-time test dependencies")
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, observe_line)
    sys.monitoring.set_events(tool, sys.monitoring.events.LINE)
    if module is None:
        # Loading all dialects evaluates their registries once, before normal consumers.
        _ = Dialect.classes
    else:
        importlib.import_module(module)
    sys.monitoring.set_events(tool, 0)
    files = set()
    for loaded in tuple(sys.modules.values()):
        file = getattr(loaded, "__file__", None)
        if isinstance(file, str) and file.startswith(prefix):
            files.add(file[len(prefix):])
    print(json.dumps({"imports": sorted(files), "lines": {k: sorted(v) for k, v in calls.items()}}))


def encode(mask):
    return base64.b64encode(mask.to_bytes((mask.bit_length()+7)//8, "little")).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    source = str(args.source.resolve())
    tests = json.loads(args.catalog.read_text())["tests"]
    nodes = [r["id"] for r in tests]
    groups = defaultdict(int)
    for i, record in enumerate(tests):
        groups[record["file"]] |= 1 << i
    imports = defaultdict(int)
    lines = defaultdict(dict)

    def load(path):
        module = path.removesuffix(".py").replace("/", ".")
        out = subprocess.run([sys.executable, __file__, "--observe", source, module],
                             check=True, capture_output=True, text=True)
        return path, json.loads(out.stdout)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for path, result in pool.map(load, sorted(groups)):
            mask = groups[path]
            for name in result["imports"]:
                imports[name] |= mask
            for name, source_lines in result["lines"].items():
                dst = lines[name]
                for line in source_lines:
                    dst[line] = dst.get(line, 0) | mask
    lazy = subprocess.run([sys.executable, __file__, "--dialects", source],
                          check=True, capture_output=True, text=True)
    payload = {"nodes": nodes, "lazy_lines": json.loads(lazy.stdout)["lines"],
               "imports": {k: encode(v) for k, v in imports.items()},
               "lines": {k: {str(n): encode(v) for n,v in rows.items()} for k,rows in lines.items()}}
    with args.output.open("wb") as output:
        output.write(gzip.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),mtime=0))
    print(f"{len(groups)} test modules, {len(imports)} imports, {sum(map(len, lines.values()))} function lines")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--observe"]:
        observe(*sys.argv[2:])
    elif sys.argv[1:2] == ["--dialects"]:
        observe(sys.argv[2], None)
    else:
        main()
