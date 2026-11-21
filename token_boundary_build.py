"""Measure pristine tokenizer calls, their configuration, visited source lines and results."""

from collections import defaultdict
import gzip
import hashlib
import inspect
import marshal
import os
from pathlib import Path
import pickle
import sys
import time

import pytest
from coverage_build import Collector
from token_boundary import signature


def build(root, catalog, destination):
    import json

    root = root.resolve()
    destination = destination.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    sys.path.insert(0, str(root))
    os.chdir(root)
    # Monitoring must be installed before importing SQLGlot. Shared calls from import
    # must be replayed to account for initialized state that can reach multiple tests.
    target = str(root / "sqlglot/tokenizer_core.py")
    tool = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tool, "tokenizer boundary")
    active = False
    unsafe = defaultdict(int)
    unknown = 0
    hits = 0

    trace = Collector(root, nodes)
    full = (1 << len(nodes)) - 1

    def line(code, number):
        nonlocal hits
        if active:
            hits |= 1 << number
        else:
            unsafe[number] |= trace.mask or full
        return sys.monitoring.DISABLE

    def start(code, offset):
        if code.co_filename == target and code.co_qualname.startswith("TokenizerCore."):
            if code.co_name not in ("__init__", "reset", "tokenize"):
                sys.monitoring.set_local_events(tool, code, sys.monitoring.events.LINE)
        return sys.monitoring.DISABLE

    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, start)
    sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, line)
    sys.monitoring.set_events(tool, sys.monitoring.events.PY_START)
    from sqlglot.tokenizer_core import TokenizerCore

    # All scanner code objects now exist. Local events avoid instrumenting unrelated
    # Python code every time a fresh query restarts the monitoring callbacks.
    for name, method in vars(TokenizerCore).items():
        if isinstance(method, property):
            method = method.fget
        code = getattr(method, "__code__", None)
        if code is not None and name not in ("__init__", "reset", "tokenize"):
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.LINE)
    sys.monitoring.set_events(tool, 0)
    original = TokenizerCore.tokenize
    names = list(inspect.signature(TokenizerCore.__init__).parameters)[1:]
    configs = []
    config_lookup = {}
    identity_cache = {}

    def config(core):
        args = {key: getattr(core, key) for key in names}
        key = tuple(id(v) for v in args.values())
        existing = identity_cache.get(key)
        if existing and existing[1] == args:
            return existing[2]
        blob = pickle.dumps(args, protocol=5)
        idx = config_lookup.get(blob)
        if idx is None:
            idx = len(configs)
            configs.append(blob)
            config_lookup[blob] = idx
        if len(identity_cache) < 1024:
            identity_cache[key] = args, pickle.loads(blob), idx
        return idx

    cache = {}
    rows = []
    tests = defaultdict(set)
    shared = set()

    def observed(core, sql):
        nonlocal active, hits, unknown
        if active or type(sql) is not str or type(core) is not TokenizerCore:
            unknown |= trace.mask or full
            return original(core, sql)
        ci = config(core)
        key = (ci, sql)
        idx = cache.get(key)
        first = idx is None
        if first:
            idx = len(rows)
            cache[key] = idx
            rows.append(None)
        frame = sys._getframe(1)
        import_time = not trace.mask
        # Module globals and default arguments can retain values produced while
        # performing an otherwise per-test lazy import.
        while not import_time and frame is not None:
            if frame.f_code.co_name == "<module>":
                import_time = frame.f_globals.get("__name__") != "__main__"
                break
            frame = frame.f_back
        if import_time:
            shared.add(idx)
        elif trace.mask:
            tests[trace.mask].add(idx)
        active = True
        hits = 0
        sys.monitoring.restart_events()
        start = time.process_time()
        digest = None
        try:
            out = original(core, sql)
        except BaseException:
            digest = None
            raise
        else:
            digest = signature(core, sql, out) if len(sql) <= 65536 else None
            return out
        finally:
            elapsed = max(1, round(1e6 * (time.process_time() - start)))
            if first:
                rows[idx] = [
                    ci,
                    sql if digest is not None else None,
                    digest,
                    hits,
                    elapsed,
                ]
            else:
                row = rows[idx]
                if row[2] != digest:
                    row[1:3] = [None, None]
                row[3] |= hits
                row[4] = min(row[4], elapsed)
            active = False
            sys.monitoring.restart_events()

    TokenizerCore.tokenize = observed

    def external():
        nonlocal unknown
        if trace.mask:
            unknown |= trace.mask
            return
        frame = sys._getframe(1)
        while frame:
            path = frame.f_code.co_filename
            if path.startswith(str(root / "tests") + os.sep):
                fname = str(Path(path).relative_to(root))
                unknown |= sum(
                    1 << i
                    for i, node in enumerate(nodes)
                    if node.split("::", 1)[0] == fname
                )
                return
            frame = frame.f_back
        unknown |= full

    from multiprocessing.process import BaseProcess

    spawn = BaseProcess.start

    def start_process(process):
        external()
        return spawn(process)

    BaseProcess.start = start_process

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
            external()

    sys.addaudithook(audit)
    status = pytest.main(
        ["-q", "--disable-warnings", "--tb=short", "-p", "no:cacheprovider", "tests"],
        plugins=[trace],
    )
    sys.monitoring.set_events(tool, 0)
    sys.monitoring.free_tool_id(tool)
    if status not in (0, 1) or not trace.passed:
        raise RuntimeError(status)
    for i in set(range(len(nodes))) - trace.passed | trace.failed:
        unknown |= 1 << i
    per_test = [set() for _ in nodes]
    for mask, ids in tests.items():
        while mask:
            bit = mask & -mask
            per_test[bit.bit_length() - 1].update(ids)
            mask ^= bit
    raw = dict(
        source_hash=hashlib.sha256(Path(target).read_bytes()).digest(),
        nodes=nodes,
        configs=configs,
        rows=rows,
        shared=sorted(shared),
        unsafe=dict(unsafe),
        unknown=unknown,
        tests=[sorted(xs) for xs in per_test],
    )
    destination.write_bytes(gzip.compress(marshal.dumps(raw, 4), mtime=0))
    print(
        "configs",
        len(configs),
        "rows",
        len(rows),
        "test-call links",
        sum(map(len, per_test)),
        "shared",
        len(shared),
        "unreplayable",
        sum(row[2] is None for row in rows),
        "size",
        destination.stat().st_size,
    )


if __name__ == "__main__":
    build(*(Path(s) for s in sys.argv[1:]))
