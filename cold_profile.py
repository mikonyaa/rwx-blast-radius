"""Read cold import and source access profiles for individual expensive tests."""
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def build(root, catalog, output, jobs=3):
    root = Path(root).resolve()
    records = json.loads(Path(catalog).read_bytes())["tests"]
    # Limit offline collection to tests likely to determine the worker peak.
    nodes = [r["id"] for r in records if r["seconds"] >= 0.25]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root), PYTHON_CPU_COUNT="2",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", SKIP_INTEGRATION="0",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    observer = str(Path(__file__).resolve().with_name("load_envelope.py"))

    def run(node):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            try:
                proc = subprocess.run([sys.executable, observer, "--observe", str(root), node, str(report)],
                                      cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      timeout=180)
            except subprocess.TimeoutExpired:
                return node, None
            if proc.returncode or not report.is_file():
                return node, None
            data = json.loads(report.read_bytes())
            return node, data["read"] if node in data["passed"] else None
    with ThreadPoolExecutor(jobs) as pool:
        results = {n: r for n, r in pool.map(run, nodes) if r}
    encoded = json.dumps(results, separators=(",", ":"), sort_keys=True).encode()
    Path(output).write_bytes(gzip.compress(encoded, mtime=0))
    print(f"Profiled {len(results)} of {len(nodes)} candidate tests", flush=True)


class ColdCoverage:
    """Independent single-test evidence supplements broad import-time candidates."""
    def __init__(self, directory, catalog):
        self.profiles = {}
        path = Path(directory) / "cold_profile.json.gz"
        if path.is_file():
            by_id = {t["id"]: t["number"] for t in catalog}
            for node, data in json.loads(gzip.decompress(path.read_bytes())).items():
                if node in by_id:
                    self.profiles[by_id[node]] = set(data)

    def filter(self, impact, path, old, new, blocks, selected):
        if not self.profiles.keys() & selected or len(blocks) != 1 or not path.startswith("sqlglot/"):
            return selected
        start, removed, added = blocks[0]
        if len(removed) != 1 or len(added) != 1 or old == new:
            return selected
        # Cold source absence is useful for conservative declaration fallbacks.
        # Required runtime consumers still include both warm suite observations
        # and every module known to import the changed file at collection.
        absent = {n for n in self.profiles.keys() & selected if path not in self.profiles[n]}
        if not absent:
            return selected
        warm = impact.interval(impact.trace.lines.get(path, {}), 1, float("inf"))
        absent -= impact.trace.test_numbers(warm | impact.trace.reads.get(path, 0))
        absent -= impact.trace.incomplete
        if impact.import_trace:
            absent -= impact.import_trace.test_numbers(impact.import_trace.imports.get(path, 0))
        if not absent:
            return selected

        original = impact.syntax[path].tree if path in impact.syntax else ast.parse(old)
        try:
            updated = ast.parse(new)
        except (ValueError, SyntaxError):
            return selected
        # Only a single replaced literal in a declarative assignment is safe here.
        # This excludes new imports, changes to bases/keys that install methods,
        # executable initializer calls, and writes to other objects or modules.
        owners = [original] + [c for c in ast.walk(original) if isinstance(c, ast.ClassDef)]
        for owner in owners:
            for stmt in owner.body:
                if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not stmt.lineno <= start <= stmt.end_lineno:
                    continue
                if isinstance(stmt, ast.Assign):
                    if len(stmt.targets) != 1:
                        continue
                    target = stmt.targets[0]
                else:
                    target = stmt.target
                if not isinstance(target, ast.Name) or not target.id.isupper() or not stmt.value:
                    continue
                allowed = (ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Dict, ast.List, ast.Tuple, ast.Set)
                if any(not isinstance(n, allowed) for n in ast.walk(stmt.value)):
                    continue
                old_literals = [n for n in ast.walk(stmt.value) if isinstance(n, ast.Constant) and n.lineno == start]
                for value in old_literals:
                    replacement = next((n for n in ast.walk(updated) if isinstance(n, ast.Constant)
                                        and n.lineno == start and value.col_offset == n.col_offset), None)
                    if replacement is None or type(replacement.value) is not type(value.value) or replacement.value == value.value:
                        continue
                    changed_value = replacement.value
                    replacement.value = value.value
                    same = ast.dump(original) == ast.dump(updated)
                    replacement.value = changed_value
                    if same:
                        return selected - absent
        return selected


if __name__ == "__main__":
    build(*sys.argv[1:])
