"""Measure reads of plain module-level configuration on the pristine test suite."""
import ast
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys


def binding(stmt):
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign):
        target = stmt.target
    else:
        return None
    if not isinstance(target, ast.Name) or not target.id.isupper() or stmt.value is None:
        return None
    if isinstance(stmt.value, ast.Constant) and type(stmt.value.value) in (bool, int, float, str, bytes):
        return target.id


class GlobalReads:
    def __init__(self, directory, tests):
        path = Path(directory) / "global_reads.json.gz"
        self.files = {}
        if path.is_file():
            data = json.loads(gzip.decompress(path.read_bytes()))
            self.files = data["files"]
            by_id = {r["id"]: r["number"] for r in tests}
            self.by_index = [by_id.get(node) for node in data["nodes"]]
            self.unknown = set(by_id.values()) - {self.by_index[i] for i in data["complete"]}

    def match(self, path, tree, new, blocks):
        entries = self.files.get(path)
        if not entries or len(blocks) != 1:
            return None
        line, removed, added = blocks[0]
        if len(removed) != 1 or len(added) != 1 or not removed[0].startswith(tuple("_ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
            return None
        old = next((s for s in tree.body if s.lineno == line), None)
        name = binding(old)
        if name not in entries:
            return None
        fresh = ast.parse(new)
        updated = next((s for s in fresh.body if s.lineno == line), None)
        if binding(updated) != name or type(old.value.value) is not type(updated.value.value):
            return None
        updated.value = old.value
        if ast.dump(fresh) != ast.dump(tree):
            return None
        return self.unknown | {self.by_index[i] for i in entries[name] if self.by_index[i] is not None}


def build(root, catalog, output):
    import builtins
    import importlib.abc
    import importlib.machinery
    import os
    from types import ModuleType
    import pytest
    from coverage_build import Collector

    root, output = Path(root).resolve(), Path(output).resolve()
    nodes = [r["id"] for r in json.loads(Path(catalog).read_bytes())["tests"]]
    collector = Collector(root, nodes)
    observations = defaultdict(dict)
    stack = []
    classes = {}

    def mark(row, key):
        # Values consumed during import can be cached and then used by any test.
        row[key] = -1 if stack or not collector.mask else row.get(key, 0) | collector.mask

    def observed(file, name, value):
        mark(observations[file], name)
        return value

    def tracked_module(file, names):
        # Normal global lookups are covered by the AST instrumentation. This
        # intercepts external getattr, from-imports and module dictionary access.
        class ProfiledModule(ModuleType):
            def __getattribute__(self, name):
                if name in names:
                    mark(observations[file], name)
                elif name == "__dict__":
                    frame = sys._getframe(1)
                    if frame.f_globals.get("__name__", "") not in {
                            "_frozen_importlib", "_frozen_importlib_external", "importlib._bootstrap",
                            "importlib._bootstrap_external"}:
                        mark(observations[file], "*")
                return super().__getattribute__(name)
        return ProfiledModule

    class Loader(importlib.machinery.SourceFileLoader):
        def exec_module(self, module):
            path = Path(self.path)
            file = str(path.relative_to(root))
            tree = ast.parse(path.read_bytes(), self.path)
            names = {key for stmt in tree.body if (key := binding(stmt))}
            # Dynamic references to the defining module's globals or direct
            # execution of generated Python cannot be attributed to a binding.
            if any(isinstance(n, ast.Name) and n.id in ("eval", "exec", "globals", "locals")
                   or isinstance(n, ast.Attribute) and n.attr in ("__globals__", "f_globals")
                   for n in ast.walk(tree)):
                names.clear()

            class Instrument(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id not in names or not isinstance(node.ctx, ast.Load):
                        return node
                    hit = ast.Call(ast.Name("_rwx_global_read", ast.Load()),
                                   [ast.Constant(file), ast.Constant(node.id), node], [])
                    return ast.copy_location(hit, node)

            stack.append(file)
            try:
                if not names:
                    super().exec_module(module)
                    return
                observations[file].update({k: observations[file].get(k, 0) for k in names})
                if file not in classes:
                    classes[file] = tracked_module(file, names)
                module.__class__ = classes[file]
                code = compile(ast.fix_missing_locations(Instrument().visit(tree)), self.path, "exec")
                exec(code, ModuleType.__getattribute__(module, "__dict__"))
            finally:
                stack.pop()

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith(".py"):
                spec.loader = Loader(fullname, spec.origin)
            return spec

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"} and collector.mask:
            collector.failed.update(i for i in range(len(nodes)) if collector.mask & (1 << i))

    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, Finder())
    builtins._rwx_global_read = observed
    sys.addaudithook(audit)
    status = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", "tests"], plugins=[collector])
    if not collector.passed or status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        raise RuntimeError(f"Could not profile baseline: pytest exited {status}")
    result = {}
    for file, row in observations.items():
        eager = row.pop("*", 0)
        entries = {name: [i for i in range(len(nodes)) if (mask | eager) & (1 << i)]
                   for name, mask in row.items() if mask >= 0 and eager >= 0}
        if entries:
            result[file] = entries
    complete = sorted(collector.passed - collector.failed)
    data = {"nodes": nodes, "complete": complete, "files": result}
    output.write_bytes(gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    print(f"{len(complete)} tests, {sum(map(len, result.values()))} traced bindings")


if __name__ == "__main__":
    build(*sys.argv[1:])
