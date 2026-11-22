"""Observe the subject of expression-class checks, independent of future edits."""

import ast
from functools import lru_cache
import gzip
import json


def members(node):
    values = node.elts if isinstance(node, ast.Tuple) else [node]
    if not values or not all(
        isinstance(v, ast.Attribute)
        and isinstance(v.value, ast.Name)
        and v.value.id == "exp"
        for v in values
    ):
        return ()
    return tuple(v.attr for v in values)


def calls(tree):
    return (
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "isinstance"
        and len(n.args) == 2
        and not n.keywords
        and members(n.args[1])
    )


def site(node):
    return f"{node.lineno}:{node.col_offset}"


class ClassSubjects:
    def __init__(self, directory, tests):
        self.profile_path = directory / "class_subjects.json.gz"
        self.catalog = {r["id"]: r["number"] for r in tests}

    @lru_cache(maxsize=1)
    def load(self):
        if not self.profile_path.exists():
            return {}, {}, {}, {}, set(self.catalog.values())
        data = json.loads(gzip.decompress(self.profile_path.read_bytes()))
        self.numbers = [self.catalog.get(n) for n in data["nodes"]]
        unknown = set(self.catalog.values()) - {
            self.numbers[i] for i in data["complete"]
        }
        return (
            data["classes"],
            data["members"],
            data["files"],
            data["sensitive"],
            unknown,
        )

    def candidates(self, path, syntax, new, blocks):
        if len(blocks) != 1:
            return None
        start, removed, added = blocks[0]
        if (
            len(removed) != 1
            or len(added) != 1
            or "exp." not in removed[0]
            or "exp." not in added[0]
        ):
            return None
        classes, targets, files, sensitive, unknown = self.load()
        if path not in files:
            return None
        checks = [
            n
            for n in calls(syntax.tree)
            if n.args[1].lineno <= start <= n.args[1].end_lineno
        ]
        if not checks:
            return None
        tree = ast.parse(new)
        updated = {site(n): n for n in calls(tree)}
        for n in checks:
            pos = site(n)
            m = updated.get(pos)
            if m is None or pos not in files[path]:
                continue
            left, right = members(n.args[1]), members(m.args[1])
            if left == right or not all(name in targets for name in left + right):
                continue
            saved = m.args[1]
            m.args[1] = n.args[1]
            same = ast.dump(tree) == ast.dump(syntax.tree)
            m.args[1] = saved
            if not same:
                continue
            old_types = {targets[k] for k in left}
            new_types = {targets[k] for k in right}
            selected = set(unknown)
            mask = int(sensitive.get("*", "0"), 16)
            for name in set(left + right):
                mask |= int(sensitive.get(name, "0"), 16)
            for subject, observed in files[path][pos].items():
                if subject in classes:
                    chain = classes[subject]
                    if bool(old_types.intersection(chain)) == bool(
                        new_types.intersection(chain)
                    ):
                        continue
                if observed == "*":
                    return None
                mask |= int(observed, 16)
            while mask:
                bit = mask & -mask
                index = bit.bit_length() - 1
                if self.numbers[index] is not None:
                    selected.add(self.numbers[index])
                mask ^= bit
            return selected
        return None
