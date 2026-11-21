"""Track observable consumers of tokenizer byte-string escape configuration."""

import ast
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from probe import _overlay, isolated_numbers


def assignment_value(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0], node.value
    if isinstance(node, ast.AnnAssign):
        return node.target, node.value
    return None, None


def compared_guard(old, new):
    """Permit comparison flips in the otherwise identical class initialization guard."""
    restored = False
    a = list(ast.walk(old))
    b = list(ast.walk(new))
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if not isinstance(left, ast.Compare) or not isinstance(right, ast.Compare):
            continue
        if (
            len(left.ops) != 1
            or len(right.ops) != 1
            or type(left.ops[0]) is type(right.ops[0])
        ):
            continue
        if type(left.ops[0]) not in (ast.Eq, ast.NotEq) or type(right.ops[0]) not in (
            ast.Eq,
            ast.NotEq,
        ):
            return False
        right.ops = left.ops
        restored = True
    return restored and ast.dump(old) == ast.dump(new)


def eligible(path, old, new, origins):
    if path not in origins:
        return False
    updated = ast.parse(new, filename=path)

    def classes(tree):
        result = {}

        def visit(body, prefix):
            for stmt in body:
                if isinstance(stmt, ast.ClassDef):
                    owner = prefix + stmt.name
                    result[owner] = stmt
                    visit(stmt.body, owner + ".")

        visit(tree.body, "")
        return result

    original = classes(old)
    changed = False
    for name, cls in classes(updated).items():
        if name not in original or name not in origins[path]:
            continue
        previous = original[name]
        if len(previous.body) != len(cls.body):
            continue
        for a, b in zip(previous.body, cls.body):
            if (
                name == "_TokenizerBase"
                and isinstance(a, ast.FunctionDef)
                and isinstance(b, ast.FunctionDef)
                and a.name == b.name == "__init_subclass__"
                and len(a.body) == len(b.body)
            ):
                for was, now in zip(a.body, b.body):
                    if not (
                        isinstance(was, ast.If)
                        and isinstance(now, ast.If)
                        and len(was.body) == 1
                        and not was.orelse
                    ):
                        continue
                    target, _ = assignment_value(was.body[0])
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "BYTE_STRING_ESCAPES"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "cls"
                        and compared_guard(was.test, now.test)
                    ):
                        changed = True
                continue
            target, before = assignment_value(a)
            ntarget, after = assignment_value(b)
            if not (
                isinstance(target, ast.Name)
                and target.id == "BYTE_STRING_ESCAPES"
                and isinstance(ntarget, ast.Name)
                and ntarget.id == target.id
                and isinstance(before, ast.List)
                and isinstance(after, ast.List)
                and ast.dump(before) != ast.dump(after)
            ):
                continue
            if all(
                isinstance(x, ast.Constant) and type(x.value) is str
                for x in [*before.elts, *after.elts]
            ):
                b.value = before
                changed = True
    return changed and ast.dump(old) == ast.dump(updated)


def fingerprint(notify):
    from sqlglot import Dialect
    from sqlglot.executor.python import (
        Python,
    )  # registers the executor's private dialect

    # The dialect metaclass can also derive maps used by ordinary strings or identifiers.
    # Compare those maps including insertion order before accepting sparse byte-only reads.
    rows = {}
    for key, dialect in sorted({**Dialect.classes, "executor.Python": Python}.items()):
        rows[key] = [
            list(getattr(dialect, attr).items())
            for attr in (
                "UNESCAPED_SEQUENCES",
                "ESCAPED_SEQUENCES",
                "IDENTIFIER_ESCAPED_SEQUENCES",
            )
        ]
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).digest()
    os.write(notify, (str(int.from_bytes(digest, "big")) + "\n").encode())


@lru_cache(maxsize=2)
def pristine(root):
    return isolated_numbers(root, fingerprint, (), 4.0, 1.1)


class ByteConfig:
    def __init__(self, directory, tests):
        self.origins = {}
        path = Path(directory) / "token_config.json.gz"
        if path.is_file():
            data = json.loads(gzip.decompress(path.read_bytes()))
            self.origins = data["origins"]
            unaffected = set(data["unaffected"])
            self.consumers = {x["number"] for x in tests if x["id"] not in unaffected}

    def candidates(self, root, path, syntax, new):
        if not eligible(path, syntax, new, self.origins):
            return None
        baseline = pristine(root)
        if not baseline:
            return None
        with tempfile.TemporaryDirectory(prefix="rwx-token-config-") as tmp:
            view = Path(tmp)
            _overlay(root, view, {path: new})
            changed = isolated_numbers(view, fingerprint, (), 4.0, 1.1, base=root)
        return self.consumers.copy() if changed == baseline else None


def build(root, catalog, dest):
    import pytest
    from coverage_build import Collector

    root = root.resolve()
    dest = dest.resolve()
    nodes = [x["id"] for x in json.loads(catalog.read_bytes())["tests"]]
    sys.path.insert(0, str(root))
    os.chdir(root)
    from sqlglot import Dialect
    from sqlglot.executor.python import (
        Python,
    )  # registers the executor's private dialect
    from sqlglot.tokens import Tokenizer, _TokenizerBase
    from sqlglot.tokenizer_core import TokenizerCore

    dialects = [*Dialect.classes.values(), Python]
    classes = {
        base
        for d in dialects
        for t in (d.tokenizer_class, d.jsonpath_tokenizer_class)
        for base in t.__mro__
    }
    origins = defaultdict(list)
    for cls in classes:
        if cls.__module__.startswith("sqlglot."):
            path = str(Path(sys.modules[cls.__module__].__file__).relative_to(root))
            origins[path].append(cls.__qualname__)

    class Records(Collector):
        def set_context(self, mask, nodeid=None):
            self.mask = mask

    trace = Records(root, nodes)
    observed = 0
    whole_files = set()
    unknown = False

    def note():
        nonlocal observed, unknown
        if trace.mask:
            observed |= trace.mask
            return
        frame = sys._getframe(1)
        while frame:
            filename = frame.f_code.co_filename
            if filename.startswith(str(root / "tests") + "/"):
                whole_files.add(str(Path(filename).relative_to(root)))
                return
            frame = frame.f_back
        unknown = True

    class ConfigValue:
        def __init__(self, value, ignore):
            self.value, self.ignore = value, ignore

        def __get__(self, instance, cls):
            if sys._getframe(1).f_code not in self.ignore:
                note()
            return self.value

    raw = vars(TokenizerCore)["byte_string_escapes"]

    class CoreValue:
        def __get__(self, obj, cls):
            note()
            return raw.__get__(obj, cls)

        def __set__(self, obj, value):
            raw.__set__(obj, value)

    TokenizerCore.byte_string_escapes = CoreValue()
    for cls in classes:
        for attr in ("BYTE_STRING_ESCAPES", "_BYTE_STRING_ESCAPES"):
            if attr in vars(cls):
                setattr(
                    cls,
                    attr,
                    ConfigValue(vars(cls)[attr], {Tokenizer._init_core.__code__}),
                )
    for cls in {base for d in dialects for base in d.__mro__}:
        attr = "BYTE_STRINGS_SUPPORT_ESCAPED_SEQUENCES"
        if attr in vars(cls):
            setattr(cls, attr, ConfigValue(vars(cls)[attr], set()))

    # Guard class creation after the pristine built-in dialect inventory. Custom
    # classes can derive additional state from the changed escape configuration.
    tool = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(tool, "custom tokenizer construction")

    def created(code, offset):
        note()

    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, created)
    for code in (
        _TokenizerBase.__init_subclass__.__func__.__code__,
        type(Dialect).__new__.__code__,
    ):
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)

    def audit(event, args):
        if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
            note()

    sys.addaudithook(audit)
    status = pytest.main(
        ["-q", "--disable-warnings", "--tb=short", "-p", "no:cacheprovider", "tests"],
        plugins=[trace],
    )
    sys.monitoring.free_tool_id(tool)
    if status not in (0, 1) or not trace.passed:
        raise RuntimeError(status)
    unaffected = [
        node
        for i, node in enumerate(nodes)
        if i in trace.passed
        and i not in trace.failed
        and not observed & (1 << i)
        and node.split("::")[0] not in whole_files
        and not unknown
    ]
    data = {
        "origins": {p: sorted(v) for p, v in origins.items()},
        "unaffected": unaffected,
    }
    dest.write_bytes(
        gzip.compress(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
    )
    print(
        "unaffected",
        len(unaffected),
        "full files",
        sorted(whole_files),
        "unknown before test",
        unknown,
    )


if __name__ == "__main__":
    build(*(Path(s) for s in sys.argv[1:]))
