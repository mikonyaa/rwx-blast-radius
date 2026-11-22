"""Observe exact dictionary keys at SQLGlot expression metadata and argument lookups."""
import ast
import gzip
import json
from functools import lru_cache


def site_key(call):
    return f"{call.lineno}:{call.col_offset}"


def lookup(call, path):
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and 1 <= len(call.args) <= 2 and not call.keywords
            and isinstance(call.args[0], ast.Constant) and type(call.args[0].value) is str):
        return False
    if call.func.attr == "meta_get":
        return True
    return (call.func.attr == "get" and path.startswith("sqlglot/optimizer/")
            and isinstance(call.func.value, ast.Attribute) and call.func.value.attr in ("args", "meta"))


@lru_cache(maxsize=1)
def profile(filename):
    if filename.is_file():
        with gzip.open(filename, "rt") as stream:
            return json.load(stream)
    return {}


class KeyReads:
    def __init__(self, directory, tests):
        self.filename = directory / "key_reads.json.gz"
        self.catalog = {t["id"]: t["number"] for t in tests}
        self.by_index = None

    def candidates(self, path, syntax, new, blocks):
        if not self.filename.is_file() or len(blocks) != 1:
            return None
        start, removed, added = blocks[0]
        if len(removed) != 1 or len(added) != 1 or not any(tag in removed[0] for tag in ("meta_get(", ".get(")):
            return None
        baseline = profile(self.filename)
        rows = baseline.get("files", {}).get(path, {})
        if not rows:
            return None
        if self.by_index is None:
            self.by_index = [self.catalog.get(node) for node in baseline["nodes"]]
            self.unknown = set(self.catalog.values()) - {self.by_index[i] for i in baseline["complete"]}
        calls = [call for call in ast.walk(syntax.tree) if lookup(call, path) and
                 call.args[0].lineno == start and site_key(call) in rows]
        if not calls:
            return None
        tree = ast.parse(new)
        updated = {site_key(call): call for call in ast.walk(tree) if lookup(call, path)}
        for call in calls:
            pos = site_key(call)
            other = updated.get(pos)
            if other is None:
                continue
            a, b = call.args[0], other.args[0]
            if a.value == b.value:
                continue
            other.args[0] = a
            matches = ast.dump(syntax.tree) == ast.dump(tree)
            other.args[0] = b
            if not matches:
                continue
            states = rows[pos]
            if states is None:
                return None
            mask = int(states.get("", "0"), 16)
            for key in [a.value, b.value]:
                mask |= int(states.get("k" + key, "0"), 16)
            selected = set()
            while mask:
                bit = mask & -mask
                n = self.by_index[bit.bit_length() - 1]
                if n is not None:
                    selected.add(n)
                mask ^= bit
            return selected, self.unknown
        return None
