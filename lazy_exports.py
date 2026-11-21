"""Track observable cache effects of SQLGlot's package-level lazy re-exports."""
import ast
import gzip
import json
from pathlib import Path


def cache_write(node):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    return ast.unparse(node.targets[0]) == "globals()[name]" and ast.unparse(node.value) == "attr"


def plain_guard(expr):
    # Export keys validated at build time are nonempty built-in strings.
    if isinstance(expr, ast.Compare):
        return (isinstance(expr.left, ast.Name) and expr.left.id == "name"
                and len(expr.ops) == len(expr.comparators) == 1
                and isinstance(expr.ops[0], (ast.Eq, ast.NotEq))
                and isinstance(expr.comparators[0], ast.Constant)
                and type(expr.comparators[0].value) is str)
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute) or expr.args or expr.keywords:
        return False
    return expr.func.attr in {"islower", "isupper", "isalpha", "isascii"} and ast.unparse(expr.func.value) in {
        "name", "name[0]", "name[-1]",
    }


def cache_change(old, new):
    # The patched function must return exactly the same resolved value. Only
    # removing a cache write or putting a total predicate around it is allowed.
    before = next((fn for fn in old.body if isinstance(fn, ast.FunctionDef) and fn.name == "__getattr__"), None)
    after = next((fn for fn in new.body if isinstance(fn, ast.FunctionDef) and fn.name == "__getattr__"), None)
    if not before or not after:
        return False
    branches = [n for n in ast.walk(before) if isinstance(n, ast.If)
                and any(cache_write(s) for s in n.body)]
    if len(branches) != 1:
        return False
    original = branches[0]
    slot = next(i for i,s in enumerate(original.body) if cache_write(s))
    match = next((n for n in ast.walk(after) if isinstance(n, ast.If)
                  and ast.dump(n.test) == ast.dump(original.test)), None)
    if not match:
        return False
    if len(match.body) + 1 == len(original.body):
        match.body.insert(slot, original.body[slot])
    elif len(match.body) == len(original.body):
        inner = match.body[slot]
        if not (isinstance(inner, ast.If) and not inner.orelse and plain_guard(inner.test)
                and len(inner.body) == 1 and cache_write(inner.body[0])):
            return False
        match.body[slot] = inner.body[0]
    else:
        return False
    return ast.dump(old) == ast.dump(new)


class LazyExports:
    def __init__(self, directory, tests):
        location = Path(directory) / "lazy_exports.json.gz"
        profile = json.loads(gzip.decompress(location.read_bytes())) if location.exists() else {}
        self.package = profile.get("package")
        self.exports = profile.get("exports", {})
        self.file = self.package.replace(".", "/") + "/__init__.py" if self.package else None
        clean = set(profile.get("observed_clean", ()))
        self.uncertain = {t["number"] for t in tests if t["id"] not in clean}

    def candidates(self, path, syntax, new, blocks):
        if not self.package or len(blocks) != 1:
            return None
        row, removed, added = blocks[0]
        if path == self.file and any("globals()" in line for line in removed):
            if cache_change(syntax.tree, ast.parse(new)):
                return self.uncertain.copy()
        if len(removed) != 1 or len(added) != 1:
            return None
        prefix = "from " + self.package
        if not (removed[0].lstrip().startswith(prefix + " import ")
                and added[0].lstrip().startswith(prefix + ".")):
            return None
        scope, old = syntax.contexts(row)
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) or not isinstance(old, ast.ImportFrom):
            return None
        # Sole physical edit, consisting solely of one local import statement.
        replacement = ast.parse(new)
        statement = next((n for n in ast.walk(replacement) if isinstance(n, ast.ImportFrom) and n.lineno == row), None)
        if not statement or old.module != self.package or old.level or statement.level:
            return None
        if len(old.names) != 1 or len(statement.names) != 1 or ast.dump(old.names[0]) != ast.dump(statement.names[0]):
            return None
        if self.exports.get(old.names[0].name) != statement.module:
            return None
        statement.module = old.module
        return self.uncertain.copy() if ast.dump(replacement) == ast.dump(syntax.tree) else None
