"""Measure which base-version configurations tests actually use, with no patched inputs."""
import ast
import base64
from collections import defaultdict
from functools import wraps
import gzip
import json
from pathlib import Path
import sys
from types import FunctionType, MethodType

# The coverage runner replaces this at test boundaries.
active = None
_bindings = {}
_observed = set()
_reference_lines = defaultdict(dict)
_expression_references = set()
_expression_imports = {}
_transform_key_signals = {}
_transform_names = {}
_transform_declarations = defaultdict(list)
_building_dispatch = False


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


_transform_iteration = Signal()


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


class TransformReads(ValueReads):
    """Observe membership of both existing and missing expression class keys."""
    def note_key(self, key):
        signal = _transform_key_signals.get(key)
        if signal is not None and not _building_dispatch:
            signal.hit()

    def note(self, key):
        self.note_key(key)
        super().note(key)

    def __contains__(self, key):
        self.note_key(key)
        return super().__contains__(key)

    def observed_keys(self):
        if not _building_dispatch:
            _transform_iteration.hit()

    def keys(self):
        self.observed_keys()
        return super().keys()

    def items(self):
        self.observed_keys()
        return super().items()

    def values(self):
        self.observed_keys()
        return super().values()

    def __iter__(self):
        self.observed_keys()
        return super().__iter__()

    def __len__(self):
        self.observed_keys()
        return super().__len__()

    def __eq__(self, other):
        self.observed_keys()
        return super().__eq__(other)

    def __ne__(self, other):
        self.observed_keys()
        return super().__ne__(other)

    def copy(self):
        self.observed_keys()
        return super().copy()


class Access:
    """One lookup may depend on declarations inherited or copied by several classes."""
    def __init__(self, signals):
        self.signals = signals
        self.previous = None

    def hit(self):
        if active is not None and self.previous != active:
            self.previous = active
            for signal in self.signals:
                signal.hit()


def value_identity(value):
    if isinstance(value, MethodType):
        return ("method", id(value.__func__), id(value.__self__))
    if type(value) in (str, bool, int, float):
        return type(value), value
    return ("object", id(value))


def dispatch_table(name):
    # These SQLGlot tables associate an existing SQL symbol or node type with
    # its handler or spelling. Value-only edits leave trie and membership keys intact.
    return name.endswith("_PARSERS") or name in {"FUNCTIONS", "TRANSFORMS", "TYPE_MAPPING"}


def ref_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = ref_name(node.value)
        if parent is not None:
            return parent + "." + node.attr
    return None


def dispatch_dependencies(module, cls, name, value, edges, ambiguous, rekeys):
    """Follow class-to-class copies in eager table definitions; retain unknown sources."""
    if not dispatch_table(name):
        return
    group = (cls, name)
    for node in import_expressions(value):
        if isinstance(node, ast.Attribute) and dispatch_table(node.attr):
            ok, owner = resolve_key(module, node.value)
            if ok and isinstance(owner, type):
                edges[group].add((owner, node.attr))
            else:
                ambiguous.add(group)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            source = node.value.attr
            if dispatch_table(source):
                ok, key = resolve_key(module, node.slice)
                rekeys[source].add(key if ok else None)
        elif isinstance(node, ast.DictComp):
            sources = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and dispatch_table(n.attr)}
            g = node.generators[0]
            t = g.target
            keeps_key = (isinstance(t, ast.Tuple) and len(t.elts) == 2
                         and isinstance(t.elts[0], ast.Name) and isinstance(node.key, ast.Name)
                         and t.elts[0].id == node.key.id)
            if not keeps_key:
                for source in sources:
                    rekeys[source].add(None)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None and not any(isinstance(n, ast.Attribute) and dispatch_table(n.attr)
                                         for n in ast.walk(v)):
                    ambiguous.add(group)
    if isinstance(value, ast.Name):
        ambiguous.add(group)


def install_dispatch_reads(classes, origins, edges, ambiguous, rekeys):
    from sqlglot import generator

    # Resolve the ancestry of actual effective tables. Unknown eager producers
    # retain every matching origin. Multiple owners of the same dictionary are
    # merged before installation, independent of module or MRO enumeration order.
    def ancestors(owner, name):
        result = set()
        pending = [(owner, name)]
        while pending:
            cls, key = pending.pop()
            if not isinstance(cls, type):
                return None
            for parent in cls.__mro__:
                group = (parent, key)
                if group in result:
                    continue
                if group in ambiguous:
                    return None
                result.add(group)
                pending.extend(edges.get(group, ()))
        return result

    owners = list(classes)
    owners += [module for name, module in list(sys.modules.items())
               if name.startswith("sqlglot.") and module is not None]
    tables = {}
    for owner in owners:
        for name, table in vars(owner).items():
            if type(table) is dict and dispatch_table(name):
                _, aliases = tables.setdefault(id(table), (table, []))
                aliases.append((owner, name))
    for table, aliases in tables.values():
        lineages = [ancestors(owner, name) for owner, name in aliases]
        known = None if any(xs is None for xs in lineages) else set().union(*lineages)
        hits = {}
        for key, value in table.items():
            sources = []
            for owner, name, original_key, signal in origins.get(value_identity(value), ()):
                if known is not None and (owner, name) not in known:
                    continue
                if (known is not None and original_key != key
                        and original_key not in rekeys[name] and None not in rekeys[name]):
                    continue
                sources.append(signal)
            if sources:
                hits[key] = Access(sources)
        is_transform = any(name == "TRANSFORMS" for _, name in aliases)
        if hits or is_transform:
            wrapped = TransformReads(table, hits) if is_transform else ValueReads(table, hits)
            for owner, name in aliases:
                setattr(owner, name, wrapped)

    # Generator compiles TRANSFORMS into one new dictionary for each concrete
    # dialect, then reuses it. Propagate origin information into that cache so
    # a test is observed on its own lookups, regardless of which test built it.
    original = generator._build_dispatch

    @wraps(original)
    def build(cls):
        global _building_dispatch
        _building_dispatch = True
        try:
            effective = original(cls)
        finally:
            _building_dispatch = False
        source = cls.TRANSFORMS
        if not isinstance(source, ValueReads):
            return effective
        return TransformReads(effective, {key: sig for key, sig in source.signals.items()
                                      if key in effective and effective[key] is dict.get(source, key)})

    generator._build_dispatch = build
    generator._DISPATCH_CACHE.clear()


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
    from sqlglot.generator import Generator

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
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "sqlglot":
                if any(a.name == "exp" and (a.asname is None or a.asname == "exp") for a in node.names):
                    _expression_imports[str(file.relative_to(root))] = node.lineno
    for name, value in vars(exp).items():
        if not name.startswith("_") and isinstance(value, type):
            _expression_references.add("exp." + name)
            if issubclass(value, exp.Expr):
                sig = _transform_key_signals.setdefault(value, Signal())
                _transform_names["exp." + name] = sig
            if callable(getattr(value, "from_arg_list", None)):
                _expression_references.add("exp." + name + ".from_arg_list")
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
    origins = defaultdict(list)
    edges = defaultdict(set)
    ambiguous = set()
    rekeys = defaultdict(set)
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
                dispatch_dependencies(module, cls, name, stmt.value, edges, ambiguous, rekeys)
                base_key = (path, stmt.lineno)
                if issubclass(cls, Generator) and name == "TRANSFORMS" and isinstance(stmt.value, ast.Dict):
                    _transform_declarations[path].append(stmt.lineno)
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
                        if dispatch_table(name):
                            direct = ref_name(expr)
                            kind = "c" if callable(item) else "l" if type(item) is str else None
                            # A factory may be replaced by a known callable; its inner
                            # arguments are handled separately by v1's closure tracing.
                            if kind is not None and isinstance(expr, (ast.Name, ast.Attribute,
                                                                        ast.Lambda, ast.Call, ast.Constant)):
                                sig = Signal()
                                origins[value_identity(item)].append((cls, name, actual_key, sig))
                                entries["@" + kind + key_string(key)] = sig
                                if direct and callable(item):
                                    ok_value, from_module = resolve_key(module, expr)
                                    if ok_value and value_identity(from_module) == value_identity(item):
                                        earlier = _reference_lines[path].get(direct, stmt.lineno)
                                        _reference_lines[path][direct] = min(earlier, stmt.lineno)
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
    install_dispatch_reads(classes, origins, edges, ambiguous, rekeys)
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
    def encoded(signal):
        m = sum(numbers[node] for node in signal.nodes)
        return base64.b64encode(m.to_bytes((m.bit_length()+7)//8, "little")).decode()
    # The canonical Generator dispatch cache is the sole persistent derived
    # value table. No eager source or test accesses an individual TRANSFORMS
    # key by subscript on this pinned revision; if that dependency ever exists,
    # key renames could break module import before lookup telemetry begins.
    subscript_at_import = False
    for folder in ("sqlglot", "tests"):
        for source in (Path.cwd() / folder).rglob("*.py"):
            tree = ast.parse(source.read_text())
            for n in import_expressions(tree):
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "TRANSFORMS":
                    subscript_at_import = True
    key_data = {ref: encoded(sig) for ref, sig in _transform_names.items()}
    key_data["iteration"] = encoded(_transform_iteration)
    profile = discard_import_aliases({"nodes": nodes, "bindings": files,
                                      "transform_decls": {} if subscript_at_import else _transform_declarations,
                                      "transform_keys": key_data,
                                      "known_refs": _reference_lines,
                                      "expression_refs": sorted(_expression_references),
                                      "expression_imports": _expression_imports}, Path.cwd())
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


def safe_dispatch_edit(profile, path, old, previous, replacement, kind):
    if kind == "l":
        # The trace records this tag only for a resolved plain-string value,
        # including a named constant that evaluates to that spelling.
        ok, value = literal(replacement)
        return ok and type(value) is str
    if isinstance(replacement, ast.Lambda):
        # Only the body is deferred. Defaults are evaluated during class creation.
        return (not replacement.args.defaults
                and not any(x is not None for x in replacement.args.kw_defaults))
    name = ref_name(replacement)
    if name is None:
        return False
    if profile.get("known_refs", {}).get(path, {}).get(name, float("inf")) <= old.lineno:
        return True
    return (profile.get("expression_imports", {}).get(path, float("inf")) < old.lineno
            and name in profile.get("expression_refs", ()))


def selected_nodes(profile, path, old, new):
    """Return a set of pytest IDs for a comparable declaration, or None if unsupported."""
    if new is None or not name_of(old) or name_of(old) != name_of(new):
        return None
    entry = profile["bindings"].get(path, {}).get(str(old.lineno))
    if entry is None:
        return None
    extra = []
    value_mask = 0
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
            if a is None or b is None:
                return None
            if key_string(a) != key_string(b):
                observed = profile.get("transform_keys", {})
                from_key, to_key = ref_name(a), ref_name(b)
                if (old.lineno not in profile.get("transform_decls", {}).get(path, ())
                        or profile.get("expression_imports", {}).get(path, float("inf")) >= old.lineno
                        or from_key not in observed or to_key not in observed or ast.dump(x) != ast.dump(y)):
                    return None
                extra.extend((observed[from_key], observed[to_key], observed["iteration"]))
                continue
            k = key_string(a)
            kind = "c" if "@c" + k in entry else "l" if "@l" + k in entry else None
            if kind and safe_dispatch_edit(profile, path, old, x, y, kind):
                tag = "@" + kind + k
                named = ref_name(a)
                observed = profile.get("transform_keys", {})
                if (named in observed and old.lineno in profile.get("transform_decls", {}).get(path, ())):
                    # Building a per-class dispatch cache copies all values at once.
                    # Only subsequent key observations can reveal a value's effect;
                    # explicit whole-map comparisons or iteration can inspect any.
                    origin = int.from_bytes(base64.b64decode(entry[tag]), "little")
                    actual = int.from_bytes(base64.b64decode(observed[named]), "little")
                    value_mask |= origin & actual
                    extra.append(observed["iteration"])
                else:
                    keys.append(tag)
                continue
            if k not in entry:
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
    mask = value_mask
    for encoded in [*(entry[k] for k in keys), *extra]:
        mask |= int.from_bytes(base64.b64decode(encoded), "little")
    result = set()
    while mask:
        bit = mask & -mask
        result.add(profile["nodes"][bit.bit_length() - 1])
        mask ^= bit
    return result
