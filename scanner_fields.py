"""Trace scalar scanner options through Tokenizer._init_core into concrete scanner slots."""

import ast
from collections import defaultdict
from functools import cache
import gzip
import json
from pathlib import Path
import sys

from exception_profile import GuardProfile
from semantic import name_of


class ScannerFields(GuardProfile):
    filename = "scanner_fields.json.gz"

    def candidates(self, path, syntax, updated, blocks):
        rows = self.sites.get(path)
        if not rows or len(blocks) != 1:
            return None
        line, removed, added = blocks[0]
        if str(line) not in rows or len(removed) != 1 or len(added) != 1:
            return None
        _, old = syntax.contexts(line)
        if not isinstance(old, (ast.Assign, ast.AnnAssign)):
            return None
        tree = ast.parse(updated)
        for new in ast.walk(tree):
            if type(new) is not type(old) or new.lineno != line or name_of(new) != name_of(old):
                continue
            if not isinstance(new.value, ast.Constant) or not isinstance(old.value, ast.Constant):
                return None
            if type(new.value.value) is not type(old.value.value):
                return None
            new.value = old.value
            if ast.dump(tree) == ast.dump(syntax.tree):
                return self.unknown | {self.numbers[i] for i in rows[str(line)] if self.numbers[i] is not None}
        return None


def build(source, catalog, dest, reverse=False):
    import gc
    import os
    import pytest
    from coverage_build import Collector
    from semantic import import_expressions

    root = source.resolve()
    dest = dest.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_text())["tests"]]
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import Dialect
    from sqlglot.executor.python import Python
    from sqlglot import tokens
    from sqlglot.tokenizer_core import TokenizerCore

    class Visits(Collector):
        def set_context(self, mask, nodeid=None):
            self.mask = mask

    tracker = Visits(root, nodes, reverse=reverse)
    tree = ast.parse((root / "sqlglot/tokens.py").read_bytes())
    if any(isinstance(obj, TokenizerCore) for obj in gc.get_objects()):
        raise RuntimeError("a tokenizer core was already initialized before tracking")
    constructor = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_init_core")
    construct = constructor.body[0].value
    assert isinstance(construct, ast.Call) and ast.unparse(construct.func) == "TokenizerCore"
    bindings = {}
    forbidden = set()
    for kw in construct.keywords:
        if not isinstance(kw.value, ast.Attribute) or not kw.arg:
            continue
        owner = ast.unparse(kw.value.value)
        if owner in {"self", "self.dialect"} and kw.arg in TokenizerCore.__slots__:
            bindings[kw.value.attr] = (kw.arg, owner)

    declarations = {}
    dialects = {Python, *Dialect.classes.values()}
    owners = {base for d in dialects
              for cls in (d, d.tokenizer_class, d.jsonpath_tokenizer_class) for base in cls.__mro__}
    for cls in owners:
        module = sys.modules[cls.__module__]
        filename = getattr(module, "__file__", None)
        if not filename or not Path(filename).resolve().is_relative_to(root / "sqlglot"):
            continue
        path = Path(filename).resolve().relative_to(root)
        body = ast.parse((root/path).read_bytes()).body
        for part in cls.__qualname__.split("."):
            declaration = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == part), None)
            if declaration is None:
                break
            body = declaration.body
        else:
            for stmt in body:
                name = name_of(stmt)
                if name in bindings and isinstance(stmt.value, ast.Constant) and type(stmt.value.value) in (bool, str, int):
                    if type(vars(cls).get(name)) is type(stmt.value.value) and vars(cls)[name] == stmt.value.value:
                        declarations[cls, name] = (str(path), stmt.lineno)

    # Skip symbols used in eager metadata outside the constructor. A copied class
    # value could be consumed without reading either the class or the scanner.
    for folder in ("sqlglot", "tests"):
        for path in (root/folder).rglob("*.py"):
            tree = ast.parse(path.read_bytes())
            for function in ast.walk(tree):
                if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name in {
                    "__init__", "__new__", "__init_subclass__"
                }:
                    forbidden.update(node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute))
            for expression in import_expressions(tree):
                if isinstance(expression, ast.Attribute):
                    forbidden.add(expression.attr)
                elif isinstance(expression, ast.Name) and isinstance(expression.ctx, ast.Load):
                    forbidden.add(expression.id)
    declarations = {key: row for key, row in declarations.items() if key[1] not in forbidden}
    fields = {name: binding for name, binding in bindings.items() if any(a == name for _, a in declarations)}
    masks = {row: 0 for row in declarations.values()}
    build_code = tokens.Tokenizer._init_core.__code__

    def hit(signal):
        if signal is not None:
            if tracker.mask:
                masks[signal] |= tracker.mask
            else:
                masks[signal] = -1

    @cache
    def sources(tokenizer_type, dialect_type):
        origins = {}
        for name, (field, owner) in fields.items():
            cls = tokenizer_type if owner == "self" else dialect_type
            for base in cls.__mro__:
                if name in vars(base):
                    origins[field] = declarations.get((base, name))
                    break
        return origins

    class ProfiledCore(TokenizerCore):
        __slots__ = ("_observed_fields",)

        def __init__(self, *args, **kwargs):
            caller = sys._getframe(1)
            if caller.f_code is build_code:
                tokenizer = caller.f_locals["self"]
                self._observed_fields = sources(type(tokenizer), type(tokenizer.dialect))
            else:
                self._observed_fields = {}
            super().__init__(*args, **kwargs)

    class Slot:
        def __init__(self, name):
            self.name = name
            self.raw = vars(TokenizerCore)[name]

        def __get__(self, obj, owner):
            if obj is not None:
                hit(obj._observed_fields.get(self.name))
            return self.raw.__get__(obj, owner)

        def __set__(self, obj, value):
            self.raw.__set__(obj, value)

    class Scalar:
        def __init__(self, literal, signal):
            self.literal = literal
            self.signal = signal

        def __get__(self, obj, owner):
            if sys._getframe(1).f_code is not build_code:
                hit(self.signal)
            return self.literal

    for name, _ in fields.values():
        setattr(ProfiledCore, name, Slot(name))
    for (cls, name), signal in declarations.items():
        setattr(cls, name, Scalar(vars(cls)[name], signal))
    tokens.TokenizerCore = ProfiledCore

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"} and tracker.mask:
            tracker.failed.update(i for i in range(len(nodes)) if tracker.mask & (1 << i))
    sys.addaudithook(audit)
    result = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=no", "tests"], plugins=[tracker])
    if result not in (0, 1) or not tracker.passed:
        raise RuntimeError(f"pytest exited with {result}")
    files = defaultdict(dict)
    for (path, line), mask in masks.items():
        if mask < 0:
            continue
        files[path][str(line)] = [i for i in range(len(nodes)) if mask & (1 << i)]
    data = {"nodes": nodes, "complete": sorted(tracker.passed - tracker.failed), "files": files}
    dest.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True).encode(), mtime=0))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        from callback_profile import merge
        merge(*(Path(arg) for arg in sys.argv[2:]))
    else:
        build(*(Path(arg) for arg in sys.argv[1:4]), reverse="--reverse" in sys.argv[4:])
