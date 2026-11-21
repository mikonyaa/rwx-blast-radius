"""Resolve import-time test consumers using pristine module closures and runtime lookups."""
import ast
import base64
from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path


def decode(value):
    return int.from_bytes(base64.b64decode(value), "little")


class ModuleImpact:
    def __init__(self, directory, tests):
        self.by_path = {}
        self.unknown = set()
        self.numbers = []
        directory = Path(directory)
        closures_path = directory / "module_imports.json.gz"
        uses_path = directory / "dialects.json.gz"
        if not closures_path.is_file() or not uses_path.is_file():
            return
        with gzip.open(closures_path, "rt") as handle:
            closures = json.load(handle)["closures"]
        with gzip.open(uses_path, "rt") as handle:
            data = json.load(handle)
        ids = {record["id"]: record["number"] for record in tests}
        self.numbers = [ids.get(node) for node in data["nodes"]]
        self.unknown = {self.numbers[i] for i in data["incomplete"]} - {None}
        self.unknown.update(set(ids.values()) - set(self.numbers))

        demand = defaultdict(int)
        for module, value in data["modules"].items():
            demand[module] |= decode(value)
        all_dialects = decode(data["all_dialects"])
        dialect_all = set()
        for module, files in closures.items():
            if module.count(".") == 2 and module.startswith("sqlglot.dialects.") and files:
                dialect_all.update(files)
        for dialect, value in data["dialects"].items():
            demand["sqlglot.dialects." + dialect.lower()] |= decode(value)

        # Limit this shortcut to the dialect component families we instrument.
        # Other packages and optional imports keep the full-suite fallback.
        families = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")
        covered = {p for p in dialect_all if p.startswith(families)}
        consumers = {path: all_dialects for path in covered}
        unresolved = 0
        for module, mask in demand.items():
            # A from-import can identify an attribute; fall back to the containing
            # loaded module if that name is not itself an importable module.
            while module and module not in closures:
                module = module.rpartition(".")[0]
            if not module:
                continue
            files = closures[module]
            if files is None:
                unresolved |= mask
                continue
            for file in files:
                if file in consumers:
                    consumers[file] |= mask
        self.by_path = {path: mask | unresolved for path, mask in consumers.items()}

    @lru_cache(maxsize=256)
    def users(self, path):
        if path not in self.by_path:
            return None
        mask = self.by_path[path]
        tests = self.unknown.copy()
        while mask:
            bit = mask & -mask
            number = self.numbers[bit.bit_length() - 1]
            if number is not None:
                tests.add(number)
            mask ^= bit
        return tests


@lru_cache(maxsize=128)
def eager_exports(source_root, package):
    """Recognize fixed direct exports from child modules in an ancestor __init__."""
    init = Path(source_root).joinpath(*package.split("."), "__init__.py")
    if not init.is_file():
        return {}
    tree = ast.parse(init.read_bytes())
    blocked = set()
    exports = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and not statement.level and statement.module:
            for alias in statement.names:
                target = alias.asname or alias.name
                if target in exports:
                    blocked.add(target)
                exports[target] = (statement.module, alias.name)
        else:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    blocked.add(node.id)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    blocked.add(node.name)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    blocked.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return {name: value for name, value in exports.items() if name not in blocked}


def same_imports(root, old, new, read_at_import):
    """A function can use the defining child in place of its package's eager re-export."""
    if not isinstance(old, ast.ImportFrom) or not isinstance(new, ast.ImportFrom):
        return False
    if old.level or new.level or not old.module or not new.module or len(old.names) != len(new.names):
        return False
    if read_at_import or not new.module.startswith(old.module + "."):
        return False
    exports = eager_exports(root, old.module)
    for a, b in zip(old.names, new.names):
        if (a.asname or a.name) != (b.asname or b.name):
            return False
        if exports.get(a.name) != (new.module, b.name):
            return False
    return True
