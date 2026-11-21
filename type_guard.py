"""Profile which members of pristine isinstance tuples match their subject."""
import ast
from functools import lru_cache
import builtins

from exception_profile import GuardProfile, shadowed


def tags(expr):
    entries = expr.elts if isinstance(expr, ast.Tuple) else [expr]
    result = []
    for node in entries:
        if isinstance(node, ast.Name):
            result.append(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp":
            result.append("exp." + node.attr)
        else:
            return []
    return result


def checks(tree):
    return (n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "isinstance"
            and len(n.args) == 2 and not n.keywords and tags(n.args[1]))


def instrument(tree, rows, sites):
    for check in list(checks(tree)):
        if not isinstance(check.args[1], ast.Tuple) or len(check.args[1].elts) < 2:
            continue
        pos = f"{check.lineno}:{check.col_offset}"
        check.args = [ast.Constant(len(sites)), check.func, *check.args]
        check.func = ast.Name("_rwx_type_guard", ast.Load())
        sites.append(rows.setdefault(pos, {}))


def stable_names(tree, removed):
    """Dropping type expressions must not suppress a user-defined lookup."""
    local = {name for name in removed if "." not in name}
    if any(not isinstance(getattr(builtins, name, None), type) for name in local) or shadowed(tree, local):
        return False
    if not any(name.startswith("exp.") for name in removed):
        return True
    imports = [stmt for stmt in tree.body if isinstance(stmt, ast.ImportFrom)
               and stmt.module == "sqlglot" and not stmt.level
               and any(a.name == "exp" and a.asname is None for a in stmt.names)]
    if len(imports) != 1:
        return False
    rest = ast.Module(body=[stmt for stmt in tree.body if stmt is not imports[0]], type_ignores=[])
    return not shadowed(rest, {"exp"})


def recorder(sites, collector):
    @lru_cache(maxsize=8192)
    def ordinary_matches(cls, members):
        if any(type(t) is not type for t in members):
            return None
        # Python also consults obj.__class__ when an ordinary class match fails.
        # Proxies, virtual subclasses and custom metaclasses are inconclusive.
        if type(cls) is not type:
            return None
        builtin = cls.__module__ == "builtins" and getattr(builtins, cls.__name__, None) is cls
        if not builtin and cls.__getattribute__ is not object.__getattribute__:
            return None
        descriptor = next(base.__dict__["__class__"] for base in cls.__mro__ if "__class__" in base.__dict__)
        if descriptor is not object.__dict__["__class__"]:
            return None
        return sum(1 << i for i, t in enumerate(members) if t in cls.__mro__)

    def record(index, predicate, value, members):
        # Record uncertainty before propagating errors such as an invalid tuple.
        state = "?"
        try:
            result = predicate(value, members)
            if predicate is isinstance and all(type(cls) is type for cls in members):
                matched = ordinary_matches(type(value), members)
                if matched is not None and bool(matched) == result:
                    state = str(matched)
            return result
        finally:
            row = sites[index]
            row[state] = (row.get(state, 0) | collector.mask) if collector.mask else -1
    return record


class TypeGuardProfile(GuardProfile):
    filename = "type_guards.json.gz"

    def filter(self, filename, old_tree, new_source, blocks, during_import):
        sites = self.sites.get(filename)
        if not sites or len(blocks) != 1:
            return None
        start, removed, added = blocks[0]
        if len(removed) != 1 or len(added) != 1 or "isinstance" not in removed[0]:
            return None
        new_tree = ast.parse(new_source)
        fresh = {(c.lineno, c.col_offset): c for c in checks(new_tree)}
        for original in checks(old_tree):
            spot = original.lineno, original.col_offset
            updated = fresh.get(spot)
            if updated is None or not original.lineno <= start <= original.end_lineno:
                continue
            before, after = tags(original.args[1]), tags(updated.args[1])
            if len(set(before)) != len(before) or not after or not set(after) < set(before):
                continue
            if not stable_names(old_tree, set(before) - set(after)):
                continue
            # Only remove members. A tuple's element order may affect custom
            # class checks, which the instrumented baseline treats as unknown.
            if after != [tag for tag in before if tag in after]:
                continue
            states = sites.get(f"{spot[0]}:{spot[1]}")
            if states is None:
                continue
            preserved = updated.args[1]
            updated.args[1] = original.args[1]
            only_change = ast.dump(old_tree) == ast.dump(new_tree)
            updated.args[1] = preserved
            if not only_change:
                continue
            keep_bits = sum(1 << i for i,t in enumerate(before) if t in after)
            retain = self.unknown.copy()
            for key, indices in states.items():
                if key == "?" or (int(key) and not int(key) & keep_bits):
                    if during_import or -1 in indices:
                        return None
                    retain.update(self.numbers[i] for i in indices)
            retain.discard(None)
            return retain
        return None
