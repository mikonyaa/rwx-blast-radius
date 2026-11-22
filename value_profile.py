"""Project isolated operator and literal changes onto observed, side-effect-free operands."""

import ast
from functools import lru_cache
import gzip
import json
import operator
import boolean_values
from pathlib import Path

COMPARISONS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Lt": operator.lt,
    "LtE": operator.le,
    "Gt": operator.gt,
    "GtE": operator.ge,
    "Is": operator.is_,
    "IsNot": operator.is_not,
    "In": lambda a, b: a in b,
    "NotIn": lambda a, b: a not in b,
}
ARITHMETIC = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mult": operator.mul,
    "BitAnd": operator.and_,
    "BitOr": operator.or_,
    "BitXor": operator.xor,
    "FloorDiv": operator.floordiv,
    "Mod": operator.mod,
    "LShift": operator.lshift,
    "RShift": operator.rshift,
}
INDICES = tuple(range(-4, 5))
UNKNOWN = object()


def pos(node, kind):
    return f"{node.lineno}:{node.col_offset}:{kind}"


def literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (
        type(None),
        bool,
        int,
        str,
        bytes,
    ):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        return -node.operand.value if type(node.operand.value) is int else UNKNOWN
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [literal(x) for x in node.elts]
        if any(x is UNKNOWN for x in values):
            return UNKNOWN
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.List):
            return values
        try:
            return set(values)
        except TypeError:
            return UNKNOWN
    return UNKNOWN


def encode(value):
    if value is None:
        return "n"
    if type(value) is bool:
        return "t" if value else "f"
    if type(value) is int and value.bit_length() <= 31:
        return "i" + str(value)
    if type(value) is str and len(value) <= 64:
        return "s" + value
    if type(value) is bytes and len(value) <= 32:
        return "b" + value.hex()
    return "?"


def decode(value):
    if value == "n":
        return None
    if value in ("t", "f"):
        return value == "t"
    if value[:1] == "i":
        return int(value[1:])
    if value[:1] == "s":
        return value[1:]
    if value[:1] == "b":
        return bytes.fromhex(value[1:])
    return UNKNOWN


def numeric(op, x, y):
    if type(x) not in (int, bool) or type(y) not in (int, bool):
        return UNKNOWN
    if (
        x.bit_length() > 31
        or y.bit_length() > 31
        or (op in ("LShift", "RShift") and not 0 <= y <= 31)
    ):
        return UNKNOWN
    try:
        return ARITHMETIC[op](x, y)
    except (OverflowError, ZeroDivisionError, KeyError, ValueError):
        return UNKNOWN


def arithmetic_equal(op, x, y, newop, newx, newy):
    a, b = numeric(op, x, y), numeric(newop, newx, newy)
    # The rest of the function can observe object identity or type. Limit the
    # proof to values interned by the contest's pinned CPython runtime.
    return (
        a is not UNKNOWN
        and b is not UNKNOWN
        and a is b
        and (type(a) is bool or type(a) is int and -5 <= a <= 256)
    )


def pair_relation(x, y):
    identity = x is y
    if x is None and y is None:
        return "110"
    if type(x) in (str, int, bool, bytes) and type(y) is type(x):
        return (
            f"{int(identity)}{int(x == y)}{('l' if x < y else 'g' if x > y else 'e')}"
        )
    if type(x) in (bool, int) and type(y) in (bool, int):
        return (
            f"{int(identity)}{int(x == y)}{('l' if x < y else 'g' if x > y else 'e')}"
        )
    return f"{int(identity)}??"


def relational_truth(op, state):
    i, e, order = state
    if op in ("Is", "IsNot"):
        return (i == "1") == (op == "Is")
    if op in ("Eq", "NotEq") and e != "?":
        return (e == "1") == (op == "Eq")
    if order in "lge":
        return {
            "Lt": order == "l",
            "LtE": order != "g",
            "Gt": order == "g",
            "GtE": order != "l",
        }.get(op, UNKNOWN)
    return UNKNOWN


class ValueProfile:
    def __init__(self, folder, catalog):
        path = Path(folder) / "expression_values.json.gz"
        self.rows = {}
        if not path.exists():
            return
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        ids = {r["id"]: r["number"] for r in catalog}
        self.numbers = [ids.get(s) for s in data["nodes"]]
        self.unknown = set(ids.values()) - {self.numbers[i] for i in data["complete"]}
        self.rows = data["rows"]

    @lru_cache(maxsize=None)
    def lines(self, path):
        return {int(key.split(":", 1)[0]) for key in self.rows[path]}

    def match(self, path, tree, source, line):
        if path not in self.rows or line not in self.lines(path):
            return None
        fresh = ast.parse(source)
        guard = boolean_values.guard_edits(tree, fresh, line)
        if guard is not None:
            test, old, new = guard
            states = self.rows[path].get(pos(test, "guard"))
            if states is not None:
                return self.selected(
                    states, lambda state: boolean_values.unchanged(old, new, state)
                )
        updated = {
            (x.lineno, x.col_offset, type(x)): x
            for x in ast.walk(fresh)
            if isinstance(x, (ast.Compare, ast.BinOp, ast.Subscript))
        }
        for a in ast.walk(tree):
            if (
                not isinstance(a, (ast.Compare, ast.BinOp, ast.Subscript))
                or a.lineno != line
            ):
                continue
            b = updated.get((a.lineno, a.col_offset, type(a)))
            if b is None:
                continue
            check = self.local(a, b)
            if check is None:
                continue
            kind, changed, key, invariant = check
            groups = self.rows[path].get(pos(a, kind))
            if groups is None:
                continue
            oldfield = getattr(b, changed)
            setattr(b, changed, getattr(a, changed))
            valid = ast.dump(tree) == ast.dump(fresh)
            setattr(b, changed, oldfield)
            if not valid:
                continue
            return self.selected(groups, invariant)
        return None

    def selected(self, groups, invariant):
        mask = 0
        for state, observed in groups.items():
            if state != "?" and invariant(state):
                continue
            if observed == "*":
                return None
            mask |= int(observed, 16)
        selected = self.unknown.copy()
        while mask:
            bit = mask & -mask
            number = self.numbers[bit.bit_length() - 1]
            if number is not None:
                selected.add(number)
            mask ^= bit
        return selected

    def local(self, a, b):
        if isinstance(a, ast.Subscript):
            x, y = literal(a.slice), literal(b.slice)
            if (
                type(x) is int
                and type(y) is int
                and x != y
                and y in INDICES
                and isinstance(a.ctx, ast.Load)
            ):
                return (
                    "sub",
                    "slice",
                    y,
                    lambda state: int(state, 16) & (1 << INDICES.index(y)),
                )
            return None
        if isinstance(a, ast.Compare):
            if len(a.ops) != 1 or len(b.ops) != 1:
                return None
            op, newop = type(a.ops[0]).__name__, type(b.ops[0]).__name__
            if op != newop:

                def unchanged(state):
                    x, y = relational_truth(op, state), relational_truth(newop, state)
                    return x is not UNKNOWN and x is y

                return "rel", "ops", newop, unchanged
            sides = (
                ("left", a.left, b.left),
                ("comparators", a.comparators[0], b.comparators[0]),
            )
            for field, old, new in sides:
                x, y = literal(old), literal(new)
                if (
                    x is UNKNOWN
                    or y is UNKNOWN
                    or ast.dump(old) == ast.dump(new)
                    or op in ("Is", "IsNot")
                ):
                    continue

                def unchanged(state):
                    atom = decode(state)
                    if atom is UNKNOWN:
                        return False
                    try:
                        first = (
                            COMPARISONS[op](x, atom)
                            if field == "left"
                            else COMPARISONS[op](atom, x)
                        )
                        second = (
                            COMPARISONS[op](y, atom)
                            if field == "left"
                            else COMPARISONS[op](atom, y)
                        )
                    except (ValueError, TypeError):
                        return False
                    return first is second

                return ("rhs" if field == "left" else "lhs"), field, op, unchanged
            return None
        op, newop = type(a.op).__name__, type(b.op).__name__
        if op not in ARITHMETIC or newop not in ARITHMETIC:
            return None
        x, y = literal(a.left), literal(a.right)
        ux, uy = literal(b.left), literal(b.right)
        if op != newop:

            def unchanged(state):
                left, right = state.split(":", 1)
                first, second = decode(left), decode(right)
                return arithmetic_equal(op, first, second, newop, first, second)

            return "bin", "op", newop, unchanged
        if (
            type(x) in (int, bool)
            and type(ux) in (int, bool)
            and ast.dump(a.left) != ast.dump(b.left)
        ):

            def unchanged(state):
                left, right = state.split(":", 1)
                return arithmetic_equal(op, x, decode(right), op, ux, decode(right))

            return "bin", "left", ux, unchanged
        if (
            type(y) in (int, bool)
            and type(uy) in (int, bool)
            and ast.dump(a.right) != ast.dump(b.right)
        ):

            def unchanged(state):
                left, right = state.split(":", 1)
                return arithmetic_equal(op, decode(left), y, op, decode(left), uy)

            return "bin", "right", uy, unchanged
        return None
