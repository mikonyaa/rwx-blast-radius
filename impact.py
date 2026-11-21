"""Translate an independent unified diff to candidates using pinned-base execution traces."""
from __future__ import annotations

import ast
import base64
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import CodeType

import semantic

COMPARISON = re.compile(r"\bis\b|==|!=|<=|>=|<|>")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Change:
    before: str | None
    after: str | None
    hunks: list = field(default_factory=list)


@dataclass
class Analysis:
    paths: dict[str, str | None]
    tests: set[int]


def patch_path(value):
    name = value.split("\t", 1)[0].rstrip("\r\n")
    if name == "/dev/null":
        return None
    if name.startswith(("a/", "b/")):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or name.startswith('"'):
        raise ValueError("Unsupported patch path")
    return str(path)


def changes(patch):
    """Parse hunk bodies by their declared counts so source lines can resemble headers."""
    rows = patch.splitlines(keepends=True)
    result = []
    section_needs_header = False
    index = 0
    while index < len(rows):
        if rows[index].startswith("diff --git "):
            if section_needs_header:
                raise ValueError("Unsupported patch section")
            section_needs_header = True
        if rows[index].startswith("--- ") and index + 1 < len(rows) and rows[index + 1].startswith("+++ "):
            section_needs_header = False
            result.append(Change(patch_path(rows[index][4:]), patch_path(rows[index + 1][4:])))
            index += 2
            continue
        header = HUNK.match(rows[index])
        if not header:
            index += 1
            continue
        if not result:
            raise ValueError("Hunk precedes file header")
        old_line, old_count, _, new_count = header.groups()
        old_count = int(old_count if old_count is not None else 1)
        new_count = int(new_count if new_count is not None else 1)
        # Empty hunks indicate the *previous* source line, rather than the first one.
        start = int(old_line) - int(old_count != 0)
        index += 1
        body = []
        while old_count or new_count:
            if index == len(rows) or rows[index][0] not in " +-":
                raise ValueError("Malformed hunk")
            sign, content = rows[index][0], rows[index][1:]
            index += 1
            if index < len(rows) and rows[index].startswith("\\ No newline at end of file"):
                content = content.removesuffix("\n")
                index += 1
            body.append((sign, content))
            old_count -= sign != "+"
            new_count -= sign != "-"
            if old_count < 0 or new_count < 0:
                raise ValueError("Hunk count mismatch")
        result[-1].hunks.append((start, body))
    if not result or section_needs_header:
        raise ValueError("Missing text diff")
    return result


def apply(change, source):
    """Return the new text and positional old-side edit blocks."""
    old = source.splitlines(keepends=True)
    cursor = 0
    new = []
    edits = []
    for start, body in change.hunks:
        if start < cursor or start > len(old):
            raise ValueError("Hunk outside base")
        new.extend(old[cursor:start])
        cursor = start
        removed = []
        added = []
        for sign, content in [*body, (None, "")]:
            if sign == " " or sign is None:
                if removed or added:
                    edits.append((cursor - len(removed) + 1, removed, added))
                    removed, added = [], []
                if sign is None:
                    continue
            if sign != "+":
                if cursor >= len(old) or old[cursor] != content:
                    raise ValueError("Patch does not match pinned source")
                cursor += 1
            if sign != "-":
                new.append(content)
            if sign == "-":
                removed.append(content)
            elif sign == "+":
                added.append(content)
    new.extend(old[cursor:])
    return "".join(new), edits


def decode_mask(value):
    return int.from_bytes(base64.b64decode(value), "little")


def numbers(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class Trace:
    def __init__(self, path, tests):
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        test_ids = {t["id"]: t["number"] for t in tests}
        self.numbers = [test_ids.get(node) for node in data["nodes"]]
        self.identity = self.numbers == list(range(len(self.numbers)))
        self.lines = {path: {int(line): decode_mask(mask) for line, mask in lines.items()}
                      for path, lines in data.get("lines", {}).items()}
        self.reads = {path: decode_mask(mask) for path, mask in data.get("reads", {}).items()}
        self.imports = {path: decode_mask(mask) for path, mask in data.get("imports", {}).items()}
        self.lazy_lines = {path: set(lines) for path, lines in data.get("lazy_lines", {}).items()}
        self.incomplete = {self.numbers[i] for i in data.get("incomplete", [])
                           if self.numbers[i] is not None}
        self.incomplete.update(set(test_ids.values()) - set(self.numbers))

    def test_numbers(self, mask):
        if self.identity:
            return set(numbers(mask))
        return {self.numbers[i] for i in numbers(mask) if self.numbers[i] is not None}


PROCEDURES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
SCOPES = (*PROCEDURES, ast.ClassDef)



def normalized_code(code):
    """Ignore source positions while preserving constants (including nested code)."""
    constants = tuple(normalized_code(item) if isinstance(item, CodeType) else item
                      for item in code.co_consts)
    return code.replace(co_consts=constants, co_linetable=b"", co_firstlineno=1)

class Syntax:
    def __init__(self, source, path):
        self.tree = ast.parse(source, filename=path)
        self.nodes = defaultdict(list)
        self.string_interiors = set()

        def visit(node):
            if isinstance(node, (ast.Constant, ast.JoinedStr)) and getattr(node, "end_lineno", 0) > node.lineno:
                self.string_interiors.update(range(node.lineno + 1, node.end_lineno + 1))
            if isinstance(node, (ast.stmt, *SCOPES)):
                for line in range(node.lineno, node.end_lineno + 1):
                    self.nodes[line].append(node)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(self.tree)

    def contexts(self, line):
        chain = self.nodes.get(line, ())
        scope = next((node for node in reversed(chain) if isinstance(node, SCOPES)), None)
        stmt = next((node for node in reversed(chain) if isinstance(node, ast.stmt)), None)
        return scope, stmt


class Impact:
    def __init__(self, root, tests, map_path=None):
        self.root = Path(root)
        self.tests = tests["tests"] if isinstance(tests, dict) else tests
        self.all = {t["number"] for t in self.tests}
        self.files = defaultdict(set)
        for test in self.tests:
            self.files[test["file"]].add(test["number"])
        map_path = Path(map_path or Path(__file__).with_name("maps") / "coverage.json.gz")
        self.trace = Trace(map_path, self.tests)
        imports_path = map_path.with_name("imports.json.gz")
        self.import_trace = Trace(imports_path, self.tests) if imports_path.exists() else None
        if self.import_trace:
            same_order = self.import_trace.numbers == self.trace.numbers
            positions = {number: i for i, number in enumerate(self.trace.numbers)}
            for file, rows in self.import_trace.lines.items():
                combined = self.trace.lines.setdefault(file, {})
                for line, mask in rows.items():
                    if not same_order:
                        mask = sum(1 << positions[n] for n in self.import_trace.test_numbers(mask)
                                   if n in positions)
                    combined[line] = combined.get(line, 0) | mask
        semantic_path = map_path.with_name("semantic.json.gz")
        self.semantic = None
        if semantic_path.exists():
            with gzip.open(semantic_path, "rt") as handle:
                self.semantic = json.load(handle)
        self.by_id = {record["id"]: record["number"] for record in self.tests}
        self.source = {}
        self.syntax = {}
        self.compiled = {}
        try:
            from comparison import Comparison
        except ModuleNotFoundError as exc:
            if exc.name != "comparison":
                raise
            self.comparison = None
        else:
            self.comparison = Comparison(map_path.parent, self.tests)

    def baseline(self, path):
        if path not in self.source:
            self.source[path] = (self.root / path).read_bytes().decode("utf-8")
        return self.source[path]

    def analyze(self, patch):
        selected = set()
        paths = {}
        try:
            for change in changes(patch):
                before, after = change.before, change.after
                if not (before or after):
                    raise ValueError("Missing path")
                source = self.baseline(before) if before else ""
                new, blocks = apply(change, source)
                if before and before != after:
                    paths[before] = None
                if after:
                    paths[after] = new
                if before != after:
                    selected.update(self.all)
                    continue
                file_tests = self.for_file(before, source, new, blocks)
                if file_tests is None:
                    # A cached value may have changed before any observed test ran.
                    # A partial query replay cannot validate those initialization effects.
                    return Analysis({}, self.all.copy())
                selected.update(file_tests)
        except (ValueError, OSError, RecursionError):
            # An unrecognized form (e.g. binary/rename-only diff) must remain conservative.
            return Analysis({}, self.all.copy())
        return Analysis(paths, selected)

    def candidates(self, patch):
        return self.analyze(patch).tests

    def for_file(self, path, old, new, blocks):
        if old == new:
            return set()
        reads = self.trace.test_numbers(self.trace.reads.get(path, 0))
        if not path.endswith(".py"):
            if path in self.trace.reads:
                return reads | self.trace.incomplete
            return self.all.copy() if path.startswith(("sqlglot/", "tests/")) else reads

        try:
            # Compilation catches context-sensitive SyntaxErrors as well as malformed syntax.
            new_code = compile(new.encode(), path, "exec", dont_inherit=True)
        except (ValueError, SyntaxError):
            return self.dependents(path)
        if path not in self.syntax:
            self.syntax[path] = Syntax(old, path)
        syntax = self.syntax[path]
        # Source changes can compile to exactly the same instructions, including an
        # explicit import alias to the original name or a folded constant expression.
        # Keep direct readers of the file; they can observe the text independently.
        if path not in self.compiled:
            self.compiled[path] = normalized_code(compile(old.encode(), path, "exec", dont_inherit=True))
        if self.compiled[path] == normalized_code(new_code):
            return reads

        original_blocks = blocks
        neutral = [b for b in blocks if all(not line.strip() or line.lstrip().startswith("#")
                                             for line in [*b[1], *b[2]])]
        if any(start <= 2 and any(re.search(r"coding[:=]", line)
                                  for line in [*removed, *added])
               for start, removed, added in neutral):
            return self.all.copy()
        if neutral:
            # A comment-looking line inside a triple-quoted string is data. Parse again
            # only in that ambiguous case; source readers remain candidates either way.
            ambiguous = any(line in syntax.string_interiors
                            for start, removed, _ in neutral
                            for line in range(start, start + max(1, len(removed))))
            if ambiguous:
                source_lines = old.splitlines(keepends=True)
                for start, removed, added in reversed(neutral):
                    source_lines[start - 1:start - 1 + len(removed)] = added
                try:
                    comment_tree = ast.parse("".join(source_lines), filename=path)
                except SyntaxError:
                    neutral = []
                else:
                    if ast.dump(syntax.tree) != ast.dump(comment_tree):
                        neutral = []
            blocks = [b for b in blocks if b not in neutral]
            if not blocks:
                return reads

        if self.import_trace and self.touches_registry_build(path, syntax, blocks):
            return None
        shift = 0
        new_positions = {}
        for start, removed, added in original_blocks:
            new_positions[start] = start + shift
            shift += len(added) - len(removed)
        candidates = self.from_positions(path, syntax, blocks, new, new_positions)
        if self.comparison and all(len(removed) == len(added) == 1
                                   and COMPARISON.search(removed[0])
                                   and COMPARISON.search(added[0])
                                   for _, removed, added in blocks):
            matched = self.comparison.filter(path, syntax.tree, new)
            if matched is not None:
                candidates.intersection_update(matched)
        return candidates | reads | self.trace.incomplete

    def touches_registry_build(self, path, syntax, blocks):
        observed = self.import_trace.lazy_lines.get(path)
        if not observed:
            return False
        for start, removed, added in blocks:
            if any(n in observed for n in range(start, start + len(removed))):
                return True
            if not removed:
                scope, _ = syntax.contexts(start)
                if isinstance(scope, PROCEDURES) and any(scope.lineno <= n <= scope.end_lineno
                                                        for n in observed):
                    return True
        return False

    def from_positions(self, path, syntax, blocks, new, new_positions):
        lines = self.trace.lines.get(path, {})
        mask = 0
        explicit = set()
        new_syntax = None
        for start, removed, added in blocks:
            if not removed:
                scope, _ = syntax.contexts(start)
                if not isinstance(scope, PROCEDURES) or start <= scope.lineno:
                    return self.dependents(path)
                # A newly inserted branch can be taken on paths where neither neighbor ran.
                mask |= self.interval(lines, scope.lineno, scope.end_lineno)
                continue
            shape_words = re.compile(r"\b(?:yield|global|nonlocal)\b")
            scope_changed = shape_words.findall("".join(removed)) != shape_words.findall("".join(added))
            indentation_changed = any(
                a[:len(a) - len(a.lstrip())] != b[:len(b) - len(b.lstrip())]
                for a, b in zip(removed, added) if a.strip() and b.strip()
            )
            for line in range(start, start + len(removed)):
                scope, stmt = syntax.contexts(line)
                in_body = isinstance(scope, ast.Lambda) or (
                    isinstance(scope, PROCEDURES) and line >= scope.body[0].lineno > scope.lineno)
                if in_body:
                    if getattr(scope, "name", "") in {"__init_subclass__", "__new__"}:
                        return self.dependents(path)
                    if indentation_changed or scope_changed:
                        mask |= self.interval(lines, scope.lineno, scope.end_lineno)
                        continue
                    mask |= lines.get(line, 0)
                    if isinstance(scope, ast.Lambda):
                        mask |= lines.get(scope.lineno, 0)
                    elif stmt:
                        mask |= lines.get(stmt.lineno, 0)
                        # A change to Boolean grouping may reach operands skipped on the
                        # baseline. Parenthesized multiline condition headers sometimes
                        # have no line event; the first operand then supplies the entry.
                        if self.boolean_edit(removed, added):
                            for expr in ast.walk(stmt):
                                if isinstance(expr, ast.BoolOp) and expr.lineno <= line <= expr.end_lineno:
                                    mask |= self.interval(lines, expr.lineno, expr.end_lineno)
                    continue
                if self.semantic and added and isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    if new_syntax is None:
                        new_syntax = Syntax(new, path)
                    new_line = new_positions[start] + min(line - start, len(added) - 1)
                    _, updated = new_syntax.contexts(new_line)
                    nodes = semantic.selected_nodes(self.semantic, path, stmt, updated)
                    if nodes is not None:
                        explicit.update(self.by_id[node] for node in nodes if node in self.by_id)
                        continue
                # Module/class declarations often execute at collection, outside a test.
                if path in self.files:
                    explicit.update(self.dependents(path))
                else:
                    return self.all.copy()
        return self.trace.test_numbers(mask) | explicit

    def dependents(self, path):
        if path not in self.files or self.import_trace is None:
            return self.all.copy()
        mask = self.import_trace.imports.get(path)
        if mask is None:
            return self.all.copy()
        observed = self.interval(self.trace.lines.get(path, {}), 1, float("inf"))
        return (self.files[path] | self.import_trace.test_numbers(mask)
                | self.trace.test_numbers(observed) | self.trace.incomplete)

    @staticmethod
    def boolean_edit(removed, added):
        words = re.compile(r"\b(?:and|or)\b")
        return words.findall("".join(removed)) != words.findall("".join(added))

    @staticmethod
    def interval(lines, low, high):
        mask = 0
        for line, tests in lines.items():
            if low <= line <= high:
                mask |= tests
        return mask
