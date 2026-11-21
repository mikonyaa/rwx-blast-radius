"""Profile token table lookups on the pinned parser and its concrete dialects."""
import ast
from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path

from semantic import name_of, ref_name


def token(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "TokenType":
        return node.attr
    return None


def membership_edit(old, new):
    """Only changes to named token literals inside the same set expression."""
    if isinstance(old, ast.BinOp) and isinstance(new, ast.BinOp) and isinstance(old.op, (ast.Sub, ast.BitOr)):
        if type(old.op) is not type(new.op) or ast.dump(old.left) != ast.dump(new.left):
            return None
        return membership_edit(old.right, new.right)
    if not isinstance(old, ast.Set) or not isinstance(new, ast.Set):
        return None
    # A starred set has a fixed contribution across the edit.
    if [ast.dump(n) for n in old.elts if isinstance(n, ast.Starred)] != [
            ast.dump(n) for n in new.elts if isinstance(n, ast.Starred)]:
        return None
    a = {token(n) for n in old.elts if not isinstance(n, ast.Starred)}
    b = {token(n) for n in new.elts if not isinstance(n, ast.Starred)}
    return a ^ b if None not in a | b else None


def value_edit(old, new, profile):
    if not isinstance(new, ast.Dict) or len(old.keys) != len(new.keys):
        return None
    keys = set()
    for k1, k2, a, b in zip(old.keys, new.keys, old.values, new.values):
        if (ast.dump(k1) if k1 else None) != (ast.dump(k2) if k2 else None):
            return None
        if ast.dump(a) == ast.dump(b):
            continue
        if token(k1) is None or ref_name(a) not in profile["expr"] or ref_name(b) not in profile["expr"]:
            return None
        keys.add(token(k1))
    return keys


@lru_cache(maxsize=2)
def load(directory):
    path = Path(directory) / "parser_tables.json.gz"
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}


def selected(directory, path, old, new, catalog):
    if old is None or new is None or not name_of(old) or name_of(old) != name_of(new):
        return None
    profile = load(directory)
    site = profile.get("sites", {}).get(path, {}).get(str(old.lineno))
    if site is None:
        return None
    if isinstance(old.value, ast.Dict):
        keys = value_edit(old.value, new.value, profile)
    else:
        keys = membership_edit(old.value, new.value)
    if not keys or not keys.issubset(profile["tokens"]):
        return None
    # Every unchanged key retains its value. A set changing emptiness can affect
    # truthiness checks independently of the membership queries we traced.
    if not isinstance(old.value, ast.Dict) and len(keys) >= site["size"]:
        return None
    mask = int(site["any"], 16)
    for key in keys:
        mask |= int(site["keys"].get(key, "0"), 16)
    names = profile["nodes"]
    ids = set(profile["unknown"])
    while mask:
        bit = mask & -mask
        ids.add(names[bit.bit_length() - 1])
        mask ^= bit
    return ids | (catalog.keys() - profile_names(directory))


@lru_cache(maxsize=2)
def profile_names(directory):
    return set(load(directory)["nodes"])


def build(root, catalog, output):
    import os
    import sys
    import subprocess
    import pytest
    from coverage_build import Collector

    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import exp
    from sqlglot.dialects import Dialect
    from sqlglot.tokens import TokenType
    from sqlglot.parser import Parser

    classes = {dialect.parser_class for dialect in Dialect.classes.values()}
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    collector = Collector(root, nodes)
    token_names = {v: v.name for v in TokenType}
    active_child = 0
    run = subprocess.run

    def run_child(*a, **kw):
        nonlocal active_child
        active_child |= collector.mask
        return run(*a, **kw)

    subprocess.run = run_child
    data = defaultdict(dict)
    safe = defaultdict(set)
    eager = set()
    from semantic import import_expressions
    for folder in ("sqlglot", "tests"):
        for filename in (root/folder).rglob("*.py"):
            tree = ast.parse(filename.read_bytes())
            # Only accept table declarations of a leaf parser. Eager copies via a
            # named class can persist long past import and bypass its descriptor.
            for node in import_expressions(tree):
                if isinstance(node, ast.Attribute):
                    eager.add((getattr(node.value, "id", getattr(node.value, "attr", "")), node.attr))
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
                    v = node.comparators[0]
                elif isinstance(node, ast.Subscript):
                    v = node.value
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_match_set" and node.args:
                    v = node.args[0]
                else:
                    continue
                if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "self":
                    safe[v.attr].add((str(filename), v.lineno))

    class Reads:
        def __init__(self):
            self.whole = 0
            self.keys = defaultdict(int)
        def key(self, k):
            if k in token_names:
                self.keys[token_names[k]] |= collector.mask
            else:
                self.whole |= collector.mask

    class TokenSet(set):
        def __contains__(self, key):
            self.reads.key(key)
            return super().__contains__(key)

    class TokenMap(dict):
        def __contains__(self, key):
            self.reads.key(key)
            return super().__contains__(key)
        def __getitem__(self, key):
            self.reads.key(key)
            return super().__getitem__(key)

    class Attribute:
        def __init__(self, table, attr):
            self.table = table
            self.locations = safe[attr]
        def __get__(self, instance, owner):
            frame = sys._getframe(1)
            if (frame.f_code.co_filename, frame.f_lineno) not in self.locations:
                self.table.reads.whole |= collector.mask
            return self.table

    known = []
    for cls in classes:
        if cls is Parser or cls.__subclasses__() or not cls.__module__.startswith("sqlglot.parsers."):
            continue
        module = sys.modules[cls.__module__]
        path = Path(module.__file__).resolve().relative_to(root).as_posix()
        declarations = ast.parse((root/path).read_bytes())
        classname = next((n for n in declarations.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__), None)
        if not classname:
            continue
        for stmt in classname.body:
            name = name_of(stmt)
            obj = vars(cls).get(name)
            if type(obj) not in (set, dict) or not obj or not all(isinstance(k, TokenType) for k in obj):
                continue
            if (cls.__name__, name) in eager:
                continue
            if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
                   for other in classname.body if other is not stmt for n in import_expressions(other)):
                continue
            if isinstance(obj, dict) and not isinstance(stmt.value, ast.Dict):
                continue
            if any(other is obj and (own, key) != (cls, name) for own in classes for key, other in vars(own).items()):
                continue
            wrapped = TokenSet(obj) if isinstance(obj, set) else TokenMap(obj)
            wrapped.reads = Reads()
            known.append((path, str(stmt.lineno), len(obj), wrapped.reads))
            setattr(cls, name, Attribute(wrapped, name))

    result = pytest.main(["-q", "--disable-warnings", "--tb=short", "-p", "no:cacheprovider", "tests"], plugins=[collector])
    if result not in (0, 1) or not collector.passed:
        raise RuntimeError(result)
    for path, line, size, reads in known:
        data[path][line] = {"size": size, "any": format(reads.whole, "x"),
                            "keys": {k: format(v,"x") for k,v in reads.keys.items()}}
    unknown = (set(range(len(nodes))) - collector.passed) | collector.failed
    unknown.update(i for i in range(len(nodes)) if active_child & (1 << i))
    profile = {"nodes": nodes, "unknown": [nodes[i] for i in sorted(unknown)], "sites": data,
               "tokens": [t.name for t in TokenType],
               "expr": ["exp."+k for k,v in vars(exp).items() if isinstance(v,type) and issubclass(v,exp.Expr)]}
    output.write_bytes(gzip.compress(json.dumps(profile, separators=(",",":"),sort_keys=True).encode(),mtime=0))
    print("Profiled",len(known),"token tables,",len(unknown),"incomplete tests")


if __name__ == "__main__":
    import sys
    build(*(Path(p).resolve() for p in sys.argv[1:]))
