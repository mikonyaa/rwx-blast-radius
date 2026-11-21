"""Use pristine expression observations for isolated edits to runtime values."""

import ast
from functools import lru_cache
import gzip
import json
from pathlib import Path


def position(node, kind):
    return f"{node.lineno}:{node.col_offset}:{kind}"


def scalar(node):
    return isinstance(node, ast.Constant) and type(node.value) in (
        type(None),
        bool,
        int,
        str,
        bytes,
        float,
    )


def changed(tree, updated):
    """Recognize an isolated AST edit and identify the exact original expression."""
    result = []

    def scan(a, b, parent=None, field=None):
        if type(a) is not type(b):
            if type(a).__base__ is type(b).__base__ and isinstance(
                a, (ast.cmpop, ast.operator, ast.unaryop)
            ):
                if isinstance(parent, ast.Compare) and len(parent.ops) == 1:
                    result.append((parent, "compare", a, b))
                    return True
                if isinstance(parent, ast.UnaryOp):
                    result.append((parent, "unary", a, b))
                    return True
                if isinstance(parent, ast.BinOp):
                    result.append((parent, "binary", a, b))
                    return True
            return False
        if isinstance(parent, ast.Attribute) and field == "attr" and a != b:
            result.append((parent, "attr", a, b))
            return True
        if isinstance(a, list):
            return len(a) == len(b) and all(
                scan(x, y, parent, field) for x, y in zip(a, b)
            )
        if isinstance(a, ast.Constant):
            if (
                type(a.value) is type(b.value)
                and a.value == b.value
                and a.kind == b.kind
            ):
                return True
            if scalar(a) and scalar(b) and a.kind == b.kind:
                result.append((a, "literal", a, b))
                return True
            return False
        if isinstance(a, ast.AST):
            return all(
                scan(getattr(a, key), getattr(b, key), a, key) for key in a._fields
            )
        return a == b

    return result[0] if scan(tree, updated) and len(result) == 1 else None


class ExpressionImpact:
    def __init__(self, folder, catalog):
        self.catalog = {r["id"]: r["number"] for r in catalog}
        self.rows = {}
        path = Path(folder) / "expression_hits.json.gz"
        if not path.exists():
            return
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        self.numbers = [self.catalog.get(node) for node in data["nodes"]]
        self.unknown = set(self.catalog.values()) - {
            self.numbers[i] for i in data["complete"]
        }
        self.rows = data["rows"]

    def consumers(self, path, tree, source, blocks, during_import):
        rows = self.rows.get(path)
        if not rows or during_import or len(blocks) != 1:
            return None
        line, removed, added = blocks[0]
        if len(removed) != 1 or len(added) != 1:
            return None
        if line not in self.line_index(path):
            return None
        edit = changed(tree, ast.parse(source))
        if edit is None:
            return None
        node, kind, old, new = edit
        state = rows.get(position(node, kind))
        if state is None or state == "*":
            return None
        ids = self.unknown.copy()
        mask = int(state, 16)
        while mask:
            bit = mask & -mask
            n = self.numbers[bit.bit_length() - 1]
            if n is not None:
                ids.add(n)
            mask ^= bit
        return ids

    @lru_cache(maxsize=None)
    def line_index(self, path):
        return {int(s.split(":", 1)[0]) for s in self.rows[path]}
