"""Collect pristine scanner table value lookups per numbered test."""

import argparse
import ast
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import sys

import pytest
from coverage_build import Collector, encode


class KeywordTrace(Collector):
    def __init__(self, root, nodes):
        super().__init__(root, nodes)
        self.unknown = 0
        self.words = defaultdict(dict)
        self.blocked = set()
        self.punctuation = set()
        self.uses = defaultdict(dict)
        self.origins = defaultdict(list)

    def set_context(self, mask, nodeid=None):
        self.mask = mask

    def install(self):
        from sqlglot import Dialect
        from sqlglot.jsonpath import JSONPathTokenizer
        from sqlglot.tokens import Tokenizer, TokenType

        defaults = set()
        for dialect in Dialect.classes.values():
            for cls in (dialect.tokenizer_class, dialect.jsonpath_tokenizer_class):
                defaults.update(cls.__mro__)
        root = Path(self.root)
        for filename in (root / "sqlglot").rglob("*.py"):
            for node in ast.walk(ast.parse(filename.read_bytes())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pop"
                ):
                    subject = node.func.value
                    if isinstance(subject, ast.Name) and subject.id == "KEYWORDS":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            self.blocked.add(node.args[0].value)
        for cls in defaults:
            if issubclass(cls, Tokenizer):
                self.punctuation.update(cls.SINGLE_TOKENS)
                self.blocked.add(cls.HINT_START)
            if not cls.__module__.startswith("sqlglot."):
                continue
            path = (
                Path(sys.modules[cls.__module__].__file__).relative_to(root).as_posix()
            )
            if path != "sqlglot/jsonpath.py" and not path.startswith(
                "sqlglot/dialects/"
            ):
                continue
            family = "jsonpath" if issubclass(cls, JSONPathTokenizer) else "sql"
            for name in ("KEYWORDS", "SINGLE_TOKENS"):
                if name in vars(cls):
                    self.origins[path].append([cls.__qualname__, name, family])

        trace = self

        class Observed(dict):
            def __init__(self, original, family, name):
                super().__init__(original)
                self.context = trace.uses[family + ":" + name]
                self.tokens = {
                    k: str(k) + ":" + v.name
                    for k, v in original.items()
                    if isinstance(v, TokenType)
                }
                self.last = {}
                self.word = trace.words[family] if name == "KEYWORDS" else None
                self.word_last = {}

            def word_read(self, key):
                mask = trace.mask
                if (
                    mask
                    and self.word is not None
                    and self.word_last.get(key) != mask
                    and isinstance(key, str)
                    and len(key) <= 40
                    and key.isascii()
                    and key.isidentifier()
                ):
                    self.word_last[key] = mask
                    self.word[key] = self.word.get(key, 0) | mask

            def __contains__(self, key):
                self.word_read(key)
                return super().__contains__(key)

            def mark(self, key):
                mask = trace.mask
                if not mask or self.last.get(key) == mask:
                    return
                name = self.tokens.get(key)
                if name is not None:
                    self.last[key] = mask
                    self.context[name] = self.context.get(name, 0) | mask

            def __getitem__(self, key):
                self.mark(key)
                self.word_read(key)
                return super().__getitem__(key)

            def get(self, key, default=None):
                self.mark(key)
                self.word_read(key)
                return super().get(key, default)

        real = Tokenizer._init_core
        tables = {}

        def observed(tokenizer):
            core = real(tokenizer)
            family = "jsonpath" if isinstance(tokenizer, JSONPathTokenizer) else "sql"
            for attr in ("keywords", "single_tokens"):
                original = getattr(core, attr)
                key = id(original), family, attr
                if key not in tables:
                    tables[key] = original, Observed(original, family, attr.upper())
                setattr(core, attr, tables[key][1])
            return core

        Tokenizer._init_core = observed

        def audit(event, args):
            if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
                trace.unknown |= trace.mask

        sys.addaudithook(audit)
        return list(TokenType.__members__)

    def save(self, path, enums):
        incomplete = set(range(len(self.nodes))) - self.passed | self.failed
        unknown = self.unknown | sum(1 << i for i in incomplete)
        raw = {
            "nodes": self.nodes,
            "unknown": encode(unknown),
            "enums": enums,
            "blocked": sorted(self.blocked),
            "punctuation": sorted(self.punctuation),
            "words": {
                family: {word: encode(mask) for word, mask in by_word.items()}
                for family, by_word in self.words.items()
            },
            "origins": self.origins,
            "uses": {
                name: {key: encode(mask) for key, mask in rows.items()}
                for name, rows in self.uses.items()
            },
        }
        path.write_bytes(
            gzip.compress(
                json.dumps(raw, sort_keys=True, separators=(",", ":")).encode(), mtime=0
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    output = args.output.resolve()
    nodes = [x["id"] for x in json.loads(args.catalog.read_bytes())["tests"]]
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sys.path.insert(0, str(root))
    os.chdir(root)
    trace = KeywordTrace(root, nodes)
    enums = trace.install()
    status = pytest.main(
        ["-q", "--disable-warnings", "--tb=short", "-p", "no:cacheprovider", "tests"],
        plugins=[trace],
    )
    if not trace.passed or status not in (0, 1):
        raise RuntimeError(f"Trace failed: {status}")
    trace.save(output, enums)
    print(f"Observed {len(trace.passed)} passing tests; {len(trace.failed)} failed.")


if __name__ == "__main__":
    main()
