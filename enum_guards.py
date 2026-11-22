"""Narrow token membership guard edits using the actual tokens observed on the pinned suite."""

import ast
from functools import lru_cache
import gzip
import json


def member(node):
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "TokenType"
    ):
        return node.attr
    return None


def guard(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    rhs = node.comparators[0]
    op = node.ops[0]
    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(
        rhs, (ast.Set, ast.Tuple, ast.List)
    ):
        names = [member(v) for v in rhs.elts]
    elif isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
        names = [member(rhs)]
    else:
        return None
    if names and all(names):
        return names
    return None


def site(node):
    return f"{node.lineno}:{node.col_offset}"


class EnumGuards:
    def __init__(self, folder, catalog):
        self.filename = folder / "enum_guards.json.gz"
        self.ids = {row["id"]: row["number"] for row in catalog}

    @lru_cache(maxsize=1)
    def profile(self):
        with gzip.open(self.filename, "rt") as stream:
            data = json.load(stream)
        indices = [self.ids.get(name) for name in data["nodes"]]
        unknown = set(self.ids.values()) - {indices[i] for i in data["complete"]}
        return data, indices, unknown

    def candidates(self, path, syntax, source, blocks):
        if len(blocks) != 1 or not self.filename.is_file():
            return None
        at, old, new = blocks[0]
        if (
            len(old) != 1
            or len(new) != 1
            or "TokenType." not in old[0]
            or "TokenType." not in new[0]
        ):
            return None
        data, indices, unknown = self.profile()
        rows = data["files"].get(path)
        if not rows:
            return None
        updated = ast.parse(source)
        sites = {
            site(n): n
            for n in ast.walk(updated)
            if isinstance(n, ast.Compare) and n.lineno <= at <= n.end_lineno
        }
        for original in ast.walk(syntax.tree):
            if (
                not isinstance(original, ast.Compare)
                or not original.lineno <= at <= original.end_lineno
            ):
                continue
            names = guard(original)
            if not names:
                continue
            key = site(original)
            other = sites.get(key)
            observed = rows.get(key)
            members = guard(other)
            if (
                observed is None
                or not members
                or type(original.ops[0]) is not type(other.ops[0])
            ):
                continue
            if not (set(names) | set(members)).issubset(data["members"]) or len(
                members
            ) != len(names):
                continue
            rhs = other.comparators
            other.comparators = original.comparators
            valid = ast.dump(syntax.tree) == ast.dump(updated)
            other.comparators = rhs
            if not valid:
                continue
            changed = set(names) ^ set(members)
            mask = int(observed.get("?", "0"), 16)
            for key in changed:
                mask |= int(observed.get(key, "0"), 16)
            if mask < 0:
                return None
            impacted = set()
            while mask:
                bit = mask & -mask
                n = indices[bit.bit_length() - 1]
                if n is not None:
                    impacted.add(n)
                mask ^= bit
            return impacted, unknown
        return None
