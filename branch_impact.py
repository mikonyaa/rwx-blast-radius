"""Project a short-circuit guard extension onto its affected original branch."""
import ast


def changed_branch(tree, new_source, blocks):
    if len(blocks) != 1:
        return None
    start, removed, added = blocks[0]
    if len(removed) != 1 or len(added) != 1:
        return None
    old_line, new_line = removed[0], added[0]
    old_indent = old_line[:len(old_line) - len(old_line.lstrip())]
    new_indent = new_line[:len(new_line) - len(new_line.lstrip())]
    if (not old_line.lstrip().startswith(("if ", "elif "))
            or not (" and " in new_line or " or " in new_line)
            or old_indent != new_indent):
        return None

    new_tree = ast.parse(new_source)
    def header(root):
        return next((node for node in ast.walk(root) if isinstance(node, ast.If)
                     and node.lineno == start and node.test.end_lineno == start), None)

    old = header(tree)
    new = header(new_tree)
    if not old or not new or not isinstance(new.test, ast.BoolOp):
        return None
    op = type(new.test.op)
    prefix = old.test.values if isinstance(old.test, ast.BoolOp) and type(old.test.op) is op else [old.test]
    if len(new.test.values) <= len(prefix):
        return None
    if any(ast.dump(a) != ast.dump(b) for a, b in zip(prefix, new.test.values)):
        return None
    tail = new.test.values[len(prefix):]
    # A walrus or a nested scope can change local/closure bindings even on
    # paths that did not evaluate the second operand.
    for value in tail:
        for part in ast.walk(value):
            if isinstance(part, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await, ast.Lambda,
                                 ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp)):
                return None
            if isinstance(part, ast.Name) and part.id in {"__class__", "super"}:
                return None
    # The one edited physical line must consist solely of the condition.
    # This also excludes inline suites and split or otherwise rewritten bodies.
    if not old_line.strip().endswith(":") or not new_line.strip().endswith(":"):
        return None

    # A && B can diverge only after A was true. For A || B, require a
    # separately observable else suite as a witness that A was false.
    body = old.body if op is ast.And else old.orelse
    if not body:
        return None
    witness = body[0]
    if (not isinstance(witness, (ast.Assign, ast.AugAssign, ast.Return, ast.Raise))
            or witness.lineno <= start or witness.lineno != witness.end_lineno):
        return None
    # Only executable function bodies. Module/class conditions can initialize
    # values before the coverage context for their future consumers starts.
    scope = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body[0].lineno <= start <= node.end_lineno:
                if scope is None or node.end_lineno - node.lineno < scope.end_lineno - scope.lineno:
                    scope = node
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    # A newly captured enclosing local can change this function's closure
    # even when the extra operand is short-circuited.
    enclosing = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.lineno < scope.lineno <= node.end_lineno for node in ast.walk(tree))
    if enclosing:
        existing = {node.id for node in ast.walk(old.test) if isinstance(node, ast.Name)}
        if any(isinstance(node, ast.Name) and node.id not in existing
               for value in tail for node in ast.walk(value)):
            return None
    return witness.lineno
