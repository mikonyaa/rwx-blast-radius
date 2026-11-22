"""Compare short-circuit decisions while preserving the order of untrusted evaluations."""

import ast
from functools import lru_cache
import itertools
from types import GetSetDescriptorType, MemberDescriptorType


def describe(test):
    leaves = []

    def build(node):
        if isinstance(node, ast.BoolOp):
            return (type(node.op).__name__, *(build(n) for n in node.values))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ("Not", build(node.operand))
        leaves.append(node)
        return len(leaves) - 1

    form = build(test)
    if isinstance(form, int) or len(leaves) < 2 or len(leaves) > 8:
        return None
    return form, leaves


UNKNOWN = object()


@lru_cache(maxsize=2048)
def protocol(cls):
    # Read class dictionaries through type itself, including for custom metaclasses.
    ancestry = type.__getattribute__(cls, "__mro__")
    dictionaries = [type.__getattribute__(base, "__dict__") for base in ancestry]
    plain_truth = not any("__bool__" in d or "__len__" in d for d in dictionaries)
    plain_attrs = (
        next(
            (d["__getattribute__"] for d in dictionaries if "__getattribute__" in d),
            None,
        )
        is object.__getattribute__
    )
    return plain_truth, plain_attrs


def truth(value):
    if value is UNKNOWN:
        return UNKNOWN
    cls = type(value)
    if cls in (
        type(None),
        int,
        bool,
        str,
        bytes,
        float,
        list,
        tuple,
        dict,
        set,
        frozenset,
    ):
        return bool(value)
    if type(cls) is type and protocol(cls)[0]:
        return True
    return UNKNOWN


def read(node, frame):
    if isinstance(node, ast.Name):
        # Only a local or cell proven bound at this point.
        return frame.f_locals.get(node.id, UNKNOWN)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute):
        value = read(node.value, frame)
        cls = type(value)
        if value is UNKNOWN or type(cls) is not type or not protocol(cls)[1]:
            return UNKNOWN
        for base in cls.__mro__:
            desc = vars(base).get(node.attr, UNKNOWN)
            if desc is not UNKNOWN:
                if type(desc) is MemberDescriptorType:
                    try:
                        return desc.__get__(value, cls)
                    except AttributeError:
                        return UNKNOWN
                return UNKNOWN
        # Avoid invoking even a user-defined __dict__ property.
        for base in cls.__mro__:
            desc = vars(base).get("__dict__")
            if desc is not None:
                if type(desc) is GetSetDescriptorType:
                    data = desc.__get__(value, cls)
                    return (
                        data.get(node.attr, UNKNOWN) if type(data) is dict else UNKNOWN
                    )
                return UNKNOWN
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Is, ast.IsNot))
    ):
        a, b = read(node.left, frame), read(node.comparators[0], frame)
        if a is not UNKNOWN and b is not UNKNOWN:
            return (a is b) == isinstance(node.ops[0], ast.Is)
    return UNKNOWN


def snapshot(leaves, frame):
    result = []
    for leaf in leaves:
        state = truth(read(leaf, frame))
        if state is UNKNOWN:
            # An uninspected expression can change state used by any later leaf.
            result.extend("?" for _ in range(len(leaves) - len(result)))
            break
        result.append("t" if state else "f")
    return "".join(result)


def evaluate(form, values, opaque):
    effects = []

    def run(expr):
        if type(expr) is int:
            if expr in opaque:
                effects.append(expr)
            return values[expr]
        tag, *args = expr
        if tag == "Not":
            return not run(args[0])
        if tag == "And":
            return all(run(x) for x in args)
        return any(run(x) for x in args)

    return run(form), effects


@lru_cache(maxsize=4096)
def unchanged(old, new, state):
    opaque = [i for i, c in enumerate(state) if c == "?"]
    values = [c == "t" for c in state]
    for choices in itertools.product((False, True), repeat=len(opaque)):
        for i, value in zip(opaque, choices):
            values[i] = value
        if evaluate(old, values, opaque) != evaluate(new, values, opaque):
            return False
    return True


def guard_edits(old, new, line):
    kinds = (ast.If, ast.IfExp, ast.While)
    other = {
        (n.lineno, n.col_offset, type(n)): n
        for n in ast.walk(new)
        if isinstance(n, kinds)
    }
    for a in ast.walk(old):
        if not isinstance(a, kinds) or not a.test.lineno <= line <= a.test.end_lineno:
            continue
        b = other.get((a.lineno, a.col_offset, type(a)))
        if b is None:
            continue
        left, right = describe(a.test), describe(b.test)
        if not left or not right or left[0] == right[0]:
            continue
        if [ast.dump(n) for n in left[1]] != [ast.dump(n) for n in right[1]]:
            continue
        preserved = b.test
        b.test = a.test
        exact = ast.dump(old) == ast.dump(new)
        b.test = preserved
        if exact:
            return a.test, left[0], right[0]
    return None
