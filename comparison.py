"""Interpret changed comparison operators against baseline operand observations."""
import ast
import gzip
import json
import operator
from pathlib import Path

ORDER = {"Eq": operator.eq, "NotEq": operator.ne, "Lt": operator.lt, "LtE": operator.le,
         "Gt": operator.gt, "GtE": operator.ge}
NAMES = {"Eq", "NotEq", "Is", "IsNot", "Lt", "LtE", "Gt", "GtE"}


def truth(op, state):
    # An observation stores identity, equality, and ordering (-1, 0, 1, or unknown).
    ident, equal, order = state
    if op in {"Is", "IsNot"}:
        return bool(ident) == (op == "Is")
    if op in {"Eq", "NotEq"}:
        return None if equal == 2 else bool(equal) == (op == "Eq")
    if order == 2:
        return None
    return {"Lt": order < 0, "LtE": order <= 0, "Gt": order > 0, "GtE": order >= 0}[op]


def changed_operator(before, after):
    """Return the old location and operators if this is the sole AST difference."""
    differences = []
    def scan(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, list):
            return len(a) == len(b) and all(scan(u, v) for u, v in zip(a, b))
        if not isinstance(a, ast.AST):
            return a == b
        if isinstance(a, ast.Compare) and len(a.ops) == len(b.ops) == 1:
            x, y = type(a.ops[0]).__name__, type(b.ops[0]).__name__
            if x != y:
                if x not in NAMES or y not in NAMES:
                    return False
                rhs = a.comparators[0]
                none_rhs = isinstance(rhs, ast.Constant) and rhs.value is None
                differences.append((f"{a.lineno}:{a.col_offset}", x, y, none_rhs, None))
                return scan(a.left, b.left) and scan(a.comparators, b.comparators)
            ar, br = a.comparators[0], b.comparators[0]
            if (x == y and x in ORDER and isinstance(ar, ast.Constant) and isinstance(br, ast.Constant)
                    and type(ar.value) is type(br.value) and type(ar.value) in (str, int)
                    and ar.value != br.value):
                differences.append((f"{a.lineno}:{a.col_offset}", x, y, False, (ar.value, br.value)))
                return scan(a.left, b.left)
        return all(scan(getattr(a, f), getattr(b, f)) for f in a._fields)
    try:
        if scan(before, after) and len(differences) == 1:
            return differences[0]
    except RecursionError:
        return None


class Comparison:
    def __init__(self, maps_path, tests):
        location = Path(maps_path)
        path = (location if location.is_dir() else location.parent) / "operands.json.gz"
        self.files = {}
        self.scalars = {}
        self.unknown = set()
        if path.is_file():
            with gzip.open(path, "rt") as source:
                data = json.load(source)
            self.files = data["files"]
            self.scalars = data.get("scalars", {})
            by_id = {test["id"]: test["number"] for test in tests}
            self.by_index = [by_id.get(name) for name in data["ids"]]
            profiled = {self.by_index[i] for i in data["complete"]}
            self.unknown = set(by_id.values()) - profiled

    def filter(self, path, old, new):
        sites = self.files.get(path)
        if not sites:
            return None
        before = ast.parse(old) if isinstance(old, str) else old
        after = ast.parse(new) if isinstance(new, str) else new
        change = changed_operator(before, after)
        if change is None:
            return None
        pos, old_op, new_op, none_rhs, values = change
        if values is not None:
            groups = self.scalars.get(path, {}).get(pos)
            if groups is None:
                return None
            first, second = values
            retain = self.unknown.copy()
            for key, indices in groups.items():
                if key != "?":
                    left = int(key[1:]) if key.startswith("i") else key[1:]
                    try:
                        if ORDER[old_op](left, first) == ORDER[new_op](left, second):
                            continue
                    except TypeError:
                        pass
                if -1 in indices:
                    return None
                retain.update(self.by_index[i] for i in indices)
            retain.discard(None)
            return retain
        if pos not in sites:
            return None
        retain = self.unknown.copy()
        for encoding, indices in sites[pos].items():
            state = tuple(int(s) for s in encoding.split(","))
            if none_rhs and state[0]:
                state = (1, 1, 2)  # Both operands are the singleton None.
            a, b = truth(old_op, state), truth(new_op, state)
            if a is None or b is None or a != b:
                if -1 in indices:  # Outside a test call. Every test may see this state.
                    return None
                retain.update(self.by_index[i] for i in indices)
        retain.discard(None)
        return retain
