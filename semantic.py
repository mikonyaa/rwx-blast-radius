"""Measure which base-version configurations tests actually use, with no patched inputs."""
import ast
import base64
from collections import defaultdict
from functools import wraps
import gzip
import json
from pathlib import Path
import sys
from types import FunctionType

# The coverage runner replaces this at test boundaries.
active = None
_bindings = {}
_observed = set()


class Signal:
    __slots__ = ("nodes", "previous")

    def __init__(self):
        self.nodes = set()
        self.previous = None

    def hit(self):
        if active is not None and self.previous != active:
            if isinstance(active, tuple):
                self.nodes.update(active)
                _observed.update(active)
            else:
                self.nodes.add(active)
                _observed.add(active)
            self.previous = active


class Attribute:
    def __init__(self, value, signal):
        self.value = value
        self.signal = signal

    def __get__(self, instance, owner):
        self.signal.hit()
        return self.value


class ValueReads(dict):
    """A value-only map retains normal keys, membership, iteration and truthiness."""
    def __init__(self, values, signals):
        super().__init__(values)
        self.signals = signals

    def note(self, key):
        signal = self.signals.get(key)
        if signal is not None:
            signal.hit()

    def __getitem__(self, key):
        self.note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.note(key)
        return super().get(key, default)

    def items(self):
        for signal in self.signals.values():
            signal.hit()
        return super().items()

    def values(self):
        for signal in self.signals.values():
            signal.hit()
        return super().values()


def name_of(stmt):
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign):
        target = stmt.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) and stmt.value is not None else None


def literal(node):
    try:
        return True, ast.literal_eval(node)
    except (SyntaxError, ValueError, TypeError):
        return False, None


def key_string(node):
    return ast.dump(node, include_attributes=False)


def resolve_key(module, node):
    """Resolve fixed enum and expression-class keys without evaluating arbitrary AST."""
    if isinstance(node, ast.Name) and node.id in vars(module):
        return True, vars(module)[node.id]
    if isinstance(node, ast.Attribute):
        ok, owner = resolve_key(module, node.value)
        if ok and not node.attr.startswith("_"):
            try:
                return True, getattr(owner, node.attr)
            except AttributeError:
                pass
        return False, None
    return literal(node)


def trace_call(fn, signals):
    @wraps(fn)
    def observed(*args, **kwargs):
        for signal in signals:
            signal.hit()
        return fn(*args, **kwargs)
    return observed


def install():
    from sqlglot import exp
    from sqlglot.dialects import Dialect
    from sqlglot.tokens import Tokenizer

    root = Path.cwd()
    classes = set()
    # Include effective runtime classes, their inherited configurations and all
    # expression schemas, even when a dialect lazily constructs its components.
    for dialect in Dialect.classes.values():
        for cls in (dialect, dialect.parser_class, dialect.generator_class,
                    dialect.tokenizer_class, dialect.jsonpath_tokenizer_class):
            classes.update(cls.__mro__)
    for cls in exp.EXPR_CLASSES.values():
        classes.update(cls.__mro__)
    classes = {cls for cls in classes if cls.__module__.startswith("sqlglot")}
    syntax = {}
    read_while_initializing = set()
    for file in sorted((root / "sqlglot").rglob("*.py")):
        tree = ast.parse(file.read_text())
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and (fn.name.startswith("__init") or fn.name in {
                    "__new__", "_init_core", "_build_dispatch"}):
                read_while_initializing.update(n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute))
        syntax[str(file)] = tree
    # Instrument tokenizer tables that the scanner receives directly. A change
    # to one tuple of decoding parameters preserves both the keys and lookup path.
    tokenizer_tree = syntax[str(root / "sqlglot/tokens.py")]
    core = next(n for n in ast.walk(tokenizer_tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_init_core")
    scanner_tables = {
        kw.value.attr for call in ast.walk(core) if isinstance(call, ast.Call)
        for kw in call.keywords if isinstance(kw.value, ast.Attribute)
        and isinstance(kw.value.value, ast.Name) and kw.value.value.id == "self"
    }
    factory_signals = defaultdict(list)
    schemas = {}
    replacements = {}
    for cls in sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__)):
        module = sys.modules[cls.__module__]
        file = str(Path(module.__file__).resolve())
        if file not in syntax:
            continue
        body = syntax[file].body
        for name in cls.__qualname__.split("."):
            declaration = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == name), None)
            if declaration is None:
                break
            body = declaration.body
        else:
            path = str(Path(file).relative_to(root))
            for stmt in body:
                name = name_of(stmt)
                if name is None or name not in vars(cls):
                    continue
                value = vars(cls)[name]
                base_key = (path, stmt.lineno)
                if name == "arg_types" and isinstance(value, dict) and issubclass(cls, exp.Expr):
                    signal = Signal()
                    schemas[cls] = signal
                    _bindings[base_key] = {"all": signal}
                elif name.isupper() and name not in read_while_initializing and type(value) in (str, bool, int, float):
                    if literal(stmt.value) == (True, value):
                        signal = Signal()
                        setattr(cls, name, Attribute(value, signal))
                        _bindings[base_key] = {"all": signal}
                elif isinstance(value, dict) and isinstance(stmt.value, ast.Dict):
                    entries = {}
                    plain_values = {}
                    for key, expr in zip(stmt.value.keys, stmt.value.values):
                        if key is None:
                            continue
                        ok, actual_key = resolve_key(module, key)
                        if not ok or actual_key not in value:
                            continue
                        item = value[actual_key]
                        signal = Signal()
                        # A factory's return may be copied into other class maps. Follow
                        # the original object so calls through those copies count too.
                        if isinstance(expr, ast.Call) and isinstance(item, FunctionType) and item.__closure__:
                            factory_signals[item].append(signal)
                        # TokenizerCore retains this exact dictionary. Entries can be
                        # read by key long after the tokenizer was constructed.
                        elif (issubclass(cls, Tokenizer) and name in scanner_tables
                              and isinstance(expr, ast.Tuple) and literal(expr)[0]):
                            plain_values[actual_key] = signal
                        else:
                            continue
                        entries[key_string(key)] = signal
                    if plain_values:
                        replacements[cls, name] = ValueReads(value, plain_values)
                    if entries:
                        _bindings[base_key] = entries

    for (cls, name), value in replacements.items():
        setattr(cls, name, value)
    wrappers = {fn: trace_call(fn, signals) for fn, signals in factory_signals.items()}
    seen = set()
    for cls in classes:
        for value in vars(cls).values():
            if not isinstance(value, dict) or id(value) in seen:
                continue
            seen.add(id(value))
            for key, item in list(value.items()):
                if isinstance(item, FunctionType) and item in wrappers:
                    value[key] = wrappers[item]

    # A schema feeds derived fields such as required_args when classes are built.
    # Count every construction of its owning expression or a subclass using it.
    expr_new = exp.Expression.__new__
    if expr_new is object.__new__:
        signals = {cls: next((schemas[base] for base in cls.__mro__ if base in schemas), None)
                   for cls in classes if issubclass(cls, exp.Expression)}

        def construct(cls, *args, **kwargs):
            signal = signals.get(cls)
            if signal is not None:
                signal.hit()
            return object.__new__(cls)

        exp.Expression.__new__ = staticmethod(construct)
    # Some consumers inspect the schema directly, without constructing a node.
    for cls, signal in schemas.items():
        setattr(cls, "arg_types", Attribute(vars(cls)["arg_types"], signal))
    for cls in classes:
        if issubclass(cls, exp.Expr) and "required_args" in vars(cls):
            signal = next((schemas[b] for b in cls.__mro__ if b in schemas), None)
            if signal is not None:
                setattr(cls, "required_args", Attribute(vars(cls)["required_args"], signal))


def import_expressions(node):
    """Yield eager class/module expressions, including default arguments and decorators."""
    yield node
    if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
        children = [node.args, *getattr(node, "decorator_list", [])]
        if getattr(node, "returns", None) is not None:
            children.append(node.returns)
    else:
        children = ast.iter_child_nodes(node)
    for child in children:
        yield from import_expressions(child)


def discard_import_aliases(profile, root):
    # Runtime attribute reads cannot account for copies made before instrumentation.
    # Find such dependencies in pristine code, including class-level unpacking.
    eager_names = set()
    eager_schemas = set()
    declared = {}
    for file in sorted((Path(root) / "sqlglot").rglob("*.py")):
        tree = ast.parse(file.read_text())
        for node in import_expressions(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                eager_names.add(node.id)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                eager_names.add(node.attr)
                if node.attr == "arg_types":
                    owner = node.value
                    cls = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None)
                    if cls:
                        eager_schemas.add(cls)
        rel = str(file.relative_to(root))
        if rel not in profile["bindings"]:
            continue
        for owner in ast.walk(tree):
            if isinstance(owner, ast.ClassDef):
                for stmt in owner.body:
                    name = name_of(stmt)
                    if name is not None:
                        declared[rel, str(stmt.lineno)] = owner.name, name
    for file, bindings in profile["bindings"].items():
        for line, entry in list(bindings.items()):
            if "all" not in entry:
                continue
            owner, name = declared[file, line]
            if (name.isupper() and name in eager_names) or (name == "arg_types" and owner in eager_schemas):
                del bindings[line]
    return profile


def save(path):
    nodes = sorted(_observed)
    numbers = {node: 1 << i for i, node in enumerate(nodes)}
    files = defaultdict(dict)
    for (file, line), signals in _bindings.items():
        result = {}
        for key, signal in signals.items():
            mask = sum(numbers[node] for node in signal.nodes)
            result[key] = base64.b64encode(mask.to_bytes((mask.bit_length()+7)//8, "little")).decode()
        files[file][str(line)] = result
    profile = discard_import_aliases({"nodes": nodes, "bindings": files}, Path.cwd())
    raw = json.dumps(profile, separators=(",", ":"), sort_keys=True)
    with open(path, "wb") as out:
        out.write(gzip.compress(raw.encode(), mtime=0))


def comparable_literals(a, b):
    good_a, old = literal(a)
    good_b, new = literal(b)
    if not good_a or not good_b:
        return False

    def shape(value):
        if isinstance(value, (list, tuple)):
            return type(value), tuple(shape(child) for child in value)
        if isinstance(value, dict):
            return dict, tuple((key, shape(child)) for key, child in value.items())
        return type(value)
    return shape(old) == shape(new)


def comparable_args(a, b):
    return ast.dump(a) == ast.dump(b) or comparable_literals(a, b)


def selected_nodes(profile, path, old, new):
    """Return a set of pytest IDs for a comparable declaration, or None if unsupported."""
    if new is None or not name_of(old) or name_of(old) != name_of(new):
        return None
    entry = profile["bindings"].get(path, {}).get(str(old.lineno))
    if entry is None:
        return None
    if "all" in entry:
        if not comparable_literals(old.value, new.value):
            return None
        keys = ["all"]
    elif isinstance(old.value, ast.Dict) and isinstance(new.value, ast.Dict):
        if len(old.value.keys) != len(new.value.keys):
            return None
        keys = []
        for a, b, x, y in zip(old.value.keys, new.value.keys, old.value.values, new.value.values):
            if ast.dump(x) == ast.dump(y) and (ast.dump(a) if a else None) == (ast.dump(b) if b else None):
                continue
            if a is None or b is None or key_string(a) != key_string(b) or key_string(a) not in entry:
                return None
            if isinstance(x, ast.Call):
                if not isinstance(y, ast.Call) or ast.dump(x.func) != ast.dump(y.func):
                    return None
                if len(x.args) != len(y.args) or [k.arg for k in x.keywords] != [k.arg for k in y.keywords]:
                    return None
                if not all(comparable_args(a, b) for a, b in zip(x.args, y.args)):
                    return None
                if not all(comparable_args(a.value, b.value) for a, b in zip(x.keywords, y.keywords)):
                    return None
            elif not comparable_literals(x, y):
                return None
            keys.append(key_string(a))
    else:
        return None
    mask = 0
    for key in keys:
        mask |= int.from_bytes(base64.b64decode(entry[key]), "little")
    result = set()
    while mask:
        bit = mask & -mask
        result.add(profile["nodes"][bit.bit_length() - 1])
        mask ^= bit
    return result
