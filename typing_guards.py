"""Recognize import-only edits under a module's runtime-false typing guard."""
import ast


def guard_name(node):
    if isinstance(node, ast.Name):
        return node.id, True
    if isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING" and isinstance(node.value, ast.Name):
        return node.value.id, False
    return None


def unchanged_binding(body, until, name, direct):
    bound = False
    for stmt in body[:until]:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                if a.name == "*":
                    bound = False
                elif (a.asname or a.name.partition(".")[0]) == name:
                    bound = (
                        direct and isinstance(stmt, ast.ImportFrom) and not stmt.level
                        and stmt.module == "typing" and a.name == "TYPE_CHECKING"
                    ) or (not direct and isinstance(stmt, ast.Import) and a.name == "typing")
            continue
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            # Do not reason through arbitrary initializers, star imports or
            # mutations that may rebind the typing module or its flag.
            return False
    return bound


def unchanged_runtime_imports(tree, new_source, blocks, covered):
    if not blocks or not any("import " in line for _, old, new in blocks for line in (*old, *new)):
        return False
    first = min(start for start, _, _ in blocks)
    last = max(start + max(1, len(old)) - 1 for start, old, _ in blocks)
    for i, before in enumerate(tree.body):
        if (not isinstance(before, ast.If) or before.orelse or before.lineno >= first
                or before.end_lineno < last):
            continue
        name = guard_name(before.test)
        if not name or not unchanged_binding(tree.body, i, *name):
            continue
        if any(covered.get(n) for n in range(before.body[0].lineno, before.end_lineno + 1)):
            # True on a recorded path: ordinary analysis must include its effects.
            continue
        after_tree = ast.parse(new_source)
        if len(after_tree.body) != len(tree.body):
            continue
        after = after_tree.body[i]
        if not isinstance(after, ast.If) or after.orelse or ast.dump(before.test) != ast.dump(after.test):
            continue
        def statements(body):
            return [ast.dump(s) for s in body if not isinstance(s, (ast.Import, ast.ImportFrom))]

        if statements(before.body) != statements(after.body):
            continue
        # Reuse the whole original guard in the disposable tree. Equality
        # includes all statements, future imports, bindings and annotations
        # outside that exact block.
        after_tree.body[i] = before
        return ast.dump(tree) == ast.dump(after_tree)
    return False
