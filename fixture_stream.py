"""Track when a test can only see a fixture through a pinned SQL loader."""
from collections import defaultdict
from functools import lru_cache
import gzip
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


@lru_cache(maxsize=2)
def profile(directory):
    path = directory / "fixture_streams.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.is_file() else {}


def stable_users(root, directory, path, old, new, by_id):
    data = profile(directory)
    modes = data.get("files", {}).get(path)
    if not modes or modes.get("unattributed"):
        return set()
    from fixture_impact import helper
    module = helper(root)
    equal, changed = set(), set(data.get("uncertain", ()))
    for mode, nodes in modes.items():
        if mode == "direct":
            changed.update(nodes)
            continue
        reader = getattr(module, mode)
        def stream(content):
            with patch.object(module, "open", lambda *a, **kw: io.StringIO(content), create=True):
                return list(reader(path))
        try:
            same = stream(old) == stream(new)
        except (ValueError, IndexError, UnicodeError, TypeError):
            same = False
        (equal if same else changed).update(nodes)
    return {by_id[n] for n in equal - changed if n in by_id}


def build(source, catalog, output):
    import concurrent.futures
    import pytest
    from coverage_build import Collector, LocalExecutor
    source, output = source.resolve(), output.resolve()
    nodes = [t["id"] for t in json.loads(catalog.read_bytes())["tests"]]
    collector = Collector(source, nodes)
    reads = defaultdict(dict)
    os.chdir(source)
    sys.path.insert(0, str(source))
    prefix = str(source / "tests" / "fixtures") + os.sep
    helper_file = str(source / "tests" / "helpers.py")
    loaders = {"load_sql_fixtures", "load_sql_fixture_pairs"}

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"} and collector.mask:
            collector.failed.update(i for i in range(len(nodes)) if collector.mask & (1 << i))
        if event != "open" or not isinstance(args[0], str) or not args[0].endswith(".sql"):
            return
        if not isinstance(args[1], str) or not args[1].startswith("r"):
            return
        filename = os.path.abspath(args[0])
        if not filename.startswith(prefix):
            return
        origin = sys._getframe(1).f_code
        mode = origin.co_name if origin.co_filename == helper_file and origin.co_name in loaders else "direct"
        if not collector.mask:
            mode = "unattributed"
        entry = reads[filename[len(str(source)) + 1:]]
        entry[mode] = entry.get(mode, 0) | (collector.mask or -1)

    concurrent.futures.ProcessPoolExecutor = LocalExecutor
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sys.addaudithook(audit)
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[collector])
    if status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        raise RuntimeError(f"Profiler exited with status {status}")
    files = {p: {mode: ([n for i, n in enumerate(nodes) if mask & (1 << i)] if mask >= 0 else ["*"])
                 for mode, mask in entry.items()} for p, entry in reads.items()}
    uncertain = sorted(set(nodes) - {nodes[i] for i in collector.passed - collector.failed})
    data = {"files": files, "uncertain": uncertain}
    output.write_bytes(gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    build(*(Path(arg) for arg in sys.argv[1:4]))
