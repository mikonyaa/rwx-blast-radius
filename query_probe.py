"""Split the expensive multiprocessing executor tests at their existing fixture rows.

Build-time traces come from the pinned, passing test suite. Each bit is one query
actually executed by the original test. The same SQL, IPC serialization and
row comparisons run again for every shard reached by a changed statement.
"""

import ast
from collections import defaultdict
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import time
from multiprocessing.reduction import ForkingPickler


TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tpch",
}
HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("RWX_QUERY_CACHE", HERE / "query-cache"))


def clone(obj):
    return ForkingPickler.loads(ForkingPickler.dumps(obj))


def run_row(module, case, tag, index):
    """Replicate the parse, submit, future.result and assertion in _mp_execute."""
    meta, sql, _ = getattr(case, tag + "_sqls")[index]
    function, args = clone((module.mp_execute, (module.parse_one(sql), meta)))
    table = clone(function(*args))
    if table is not None:
        case.subtestHelper(index, table, tpch=(tag == "tpch"))


def cache_tables(source, destination):
    """Optional Docker build step: avoid repeatedly parsing immutable fixture CSVs."""
    sys.path.insert(0, str(Path(source).resolve()))
    import tests.test_executor as module

    module.TestExecutor.setUpClass()
    try:
        for tag in ("tpch", "tpcds"):
            directory = Path(destination) / tag
            directory.mkdir(parents=True, exist_ok=True)
            for name, table in getattr(module.TestExecutor, tag + "_tables").items():
                with (directory / name).open("wb") as handle:
                    pickle.dump(table, handle, protocol=5)
    finally:
        module.TestExecutor.tearDownClass()


class Tables(dict):
    def __init__(self, root):
        self.root = root

    def __missing__(self, key):
        with (self.root / key).open("rb") as handle:
            table = pickle.load(handle)
        self[key] = table
        return table


def _connection(module, tag, schema):
    folder = getattr(module, "DIR_" + tag.upper())
    conn = module.duckdb.connect(config={"threads": 1})
    for name, columns in schema.items():
        conn.execute(f"CREATE VIEW {name} AS SELECT * FROM READ_CSV("
                     f"'{folder}{name}.csv.gz', delim='|', header=True, columns={columns})")
    return conn


def probe_rows(results):
    from sqlglot import parse_one
    from tests.helpers import load_sql_fixture_pairs

    passed = []
    for number, tag, indices in results:
        folder_name = "tpc-h" if tag == "tpch" else "tpc-ds"
        sqls = list(load_sql_fixture_pairs(f"optimizer/{folder_name}/{folder_name}.sql"))
        # mp_execute returns immediately for these queries. They still cross the
        # process boundary, so their expression serialization must succeed.
        if all(not sqls[i][0].get("execute") for i in indices):
            try:
                for i in indices:
                    meta, sql, _ = sqls[i]
                    clone((parse_one(sql), meta))
            except Exception:
                continue
            passed.append(number)
            continue

        import tests.test_executor as module
        case = module.TestExecutor("test_execute_" + tag)
        schema = getattr(module, tag.upper() + "_SCHEMA")
        setattr(case, tag + "_sqls", sqls)
        conn = None
        try:
            if not (CACHE / tag).is_dir():
                continue
            conn = _connection(module, tag, schema)
            setattr(case, tag + "_conn", conn)
            case.cache = {}
            module.initializer(schema, Tables(CACHE / tag))
            for index in indices:
                run_row(module, case, tag, index)
        except Exception:
            continue
        finally:
            if conn is not None:
                conn.close()
        passed.append(number)
    return passed


def build_map(source, destination):
    root = Path(source).resolve()
    sys.path.insert(0, str(root))
    prefix = str(root / "sqlglot") + "/"
    maps = {tag: defaultdict(dict) for tag in ("setup", "tpch", "tpcds")}
    times = {}
    tag = "setup"
    bit = 1

    def line_event(code, line):
        if code.co_filename.startswith(prefix):
            relative = "sqlglot/" + code.co_filename[len(prefix):]
            lines = maps[tag][relative]
            lines[line] = lines.get(line, 0) | bit
        return sys.monitoring.DISABLE

    tracker = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tracker, "fixture query coverage")
    sys.monitoring.register_callback(tracker, sys.monitoring.events.LINE, line_event)
    sys.monitoring.set_events(tracker, sys.monitoring.events.LINE)
    try:
        import tests.test_executor as module

        module.TestExecutor.setUpClass()
        case = module.TestExecutor()
        for group in ("tpch", "tpcds"):
            tag, bit = "setup", 1
            sys.monitoring.restart_events()
            schema = getattr(module, group.upper() + "_SCHEMA")
            tables = getattr(case, group + "_tables")
            module.initializer(*clone((schema, tables)))
            times[group] = []
            for index, _ in enumerate(getattr(case, group + "_sqls")):
                tag, bit = group, 1 << index
                sys.monitoring.restart_events()
                before = time.process_time()
                run_row(module, case, group, index)
                times[group].append(round(time.process_time() - before, 4))
        tag, bit = "setup", 1
        sys.monitoring.restart_events()
        module.TestExecutor.tearDownClass()
    finally:
        sys.monitoring.set_events(tracker, 0)
        sys.monitoring.free_tool_id(tracker)
    with gzip.open(destination, "wt") as handle:
        json.dump({"lines": maps, "times": times}, handle, separators=(",", ":"))


@lru_cache(maxsize=1)
def _map():
    path = HERE / "maps" / "queries.json.gz"
    imports = path.with_name("imports.json.gz")
    if path.is_file() and imports.is_file():
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        with gzip.open(imports, "rt") as handle:
            startup = json.load(handle)
        # A function executed as a dialect module loads can initialize state
        # subsequently shared by several queries. Never slice such an edit.
        setup = data["lines"]["setup"]
        for table in (startup["lines"], startup["lazy_lines"]):
            for file, rows in table.items():
                setup.setdefault(file, {}).update((str(line), 1) for line in rows)
        return data


def _changed_lines(base, replacements):
    # Use this optimization only for same-line edits inside ordinary function
    # bodies. Definition-time changes, control-flow insertions, modified tests
    # and fixtures use the full candidate list.
    lines = []
    for path, content in replacements.items():
        if not path.startswith("sqlglot/") or not path.endswith(".py") or not isinstance(content, str):
            return None
        before = (Path(base) / path).read_text()
        old, new = before.splitlines(), content.splitlines()
        if len(old) != len(new):
            return None
        differences = {i for i, (a, b) in enumerate(zip(old, new), 1) if a != b}
        if not differences:
            continue
        for i in differences:
            a, b = old[i - 1], new[i - 1]
            if len(a) - len(a.lstrip()) != len(b) - len(b.lstrip()):
                return None
        anchors = None
        for source in (before, content):
            sites = {}
            tree = ast.parse(source)
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(fn):
                    if isinstance(node, (ast.If, ast.While)):
                        value = node.test
                    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)):
                        value = node.value
                    else:
                        continue
                    if value is None:
                        continue
                    for line in differences:
                        if fn.lineno < node.lineno <= line <= value.end_lineno:
                            sites[line] = (fn.name, node.lineno, value.end_lineno)
            if set(sites) != differences:
                return None
            if anchors is None:
                anchors = sites
            elif sites != anchors:
                return None
        for _, start, end in anchors.values():
            lines.extend((path, str(line)) for line in range(start, end + 1))
    return lines


def plan(source, replacements, candidates, tests):
    eligible = [(n, TESTS[tests[n]["id"]]) for n in candidates if tests[n]["id"] in TESTS]
    if not eligible or not (data := _map()):
        return []
    changed = _changed_lines(source, replacements)
    if not changed:
        return []
    maps = data["lines"]
    if any(line in maps["setup"].get(path, {}) for path, line in changed):
        return []
    plans = []
    for number, tag in eligible:
        mask = 0
        for path, line in changed:
            mask |= maps[tag].get(path, {}).get(line, 0)
        indices = [i for i in range(mask.bit_length()) if mask & (1 << i)]
        estimate = sum(data["times"][tag][i] for i in indices)
        if estimate < min(3.0, 0.35 * tests[number]["seconds"]):
            plans.append((number, tag, indices))
    return plans


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cache", "map"))
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    if args.mode == "cache":
        cache_tables(args.source, args.destination)
    else:
        build_map(args.source, args.destination)
