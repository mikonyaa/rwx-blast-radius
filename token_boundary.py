"""Fingerprint the externally observable result of the pinned tokenizer boundary."""

from functools import lru_cache
import hashlib
import json

STATE = (
    "size",
    "_start",
    "_current",
    "_line",
    "_col",
    "_comments",
    "_char",
    "_end",
    "_peek",
    "_prev_token_line",
)


def signature(core, sql, result):
    if core.sql != sql or result is not core.tokens:
        return None
    digest = hashlib.blake2b(digest_size=20)

    def add(value):
        digest.update(
            json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")

    add([getattr(core, n) for n in STATE])
    shared = {}
    batch = []
    for i, token in enumerate(result):
        # Preserve list aliasing, including the core's pending comment buffer.
        alias = shared.setdefault(id(token.comments), i)
        batch.append(
            [
                int(token.token_type),
                token.text,
                token.line,
                token.col,
                token.start,
                token.end,
                token.comments,
                alias if alias != i else None,
                token.comments is core._comments,
            ]
        )
        if len(batch) == 256:
            add(batch)
            batch = []
    add(batch)
    return digest.digest()


def mutation_span(old, updated):
    """Single scalar, comparison or pure string method substitution in a scanner body."""
    import ast

    original = next(
        (
            c
            for c in old.body
            if isinstance(c, ast.ClassDef) and c.name == "TokenizerCore"
        ),
        None,
    )
    if original is None:
        return None
    new = ast.parse(updated)
    mutations = []
    groups = (
        {"upper", "lower", "casefold"},
        {
            "isdigit",
            "isalpha",
            "isalnum",
            "isascii",
            "isspace",
            "isnumeric",
            "isdecimal",
        },
        {"startswith", "endswith"},
        {"find", "rfind"},
    )
    comparators = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
    )

    def same(a, b, owner=None, field=None):
        if type(a) is not type(b):
            if isinstance(a, comparators) and isinstance(b, comparators):
                mutations.append(owner)
                return True
            return False
        if isinstance(a, ast.AST):
            return all(same(getattr(a, k), getattr(b, k), a, k) for k in a._fields)
        if isinstance(a, list):
            return len(a) == len(b) and all(
                same(x, y, owner, field) for x, y in zip(a, b)
            )
        if a == b:
            return True
        if (
            isinstance(owner, ast.Constant)
            and field == "value"
            and type(a) in (bool, str, int)
            or isinstance(owner, ast.Attribute)
            and field == "attr"
            and any(a in g and b in g for g in groups)
        ):
            mutations.append(owner)
            return True
        return False

    if not same(old, new) or len(mutations) != 1:
        return None
    site = mutations[0]
    parents = {c: p for p in ast.walk(original) for c in ast.iter_child_nodes(p)}
    owner = site
    while owner in parents and not isinstance(owner, ast.stmt):
        owner = parents[owner]
    stmt = owner
    while owner in parents and not isinstance(owner, ast.FunctionDef):
        owner = parents[owner]
    if not (
        isinstance(owner, ast.FunctionDef)
        and owner in original.body
        and owner.name.startswith("_")
        and owner.name not in ("__init__", "reset", "_text")
        and site.lineno >= owner.body[0].lineno
        and not (
            stmt is owner.body[0]
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
        )
    ):
        return None
    if (
        isinstance(stmt, (ast.Assign, ast.AugAssign, ast.Return, ast.Expr))
        and stmt.lineno == stmt.end_lineno
        and stmt.lineno > owner.lineno
    ):
        return stmt.lineno, stmt.end_lineno
    # A compound condition or a folded expression may have no event on its own
    # textual line. Any executed line in the containing method is a safe witness.
    return owner.body[0].lineno, owner.end_lineno


def replay(configs, rows, work, common, channel):
    import os
    import pickle
    from sqlglot.tokenizer_core import TokenizerCore

    config_cache = {}
    outcomes = {}

    def identical(idx):
        if idx in outcomes:
            return outcomes[idx]
        conf, sql, expected, _, _ = rows[idx]
        if conf not in config_cache:
            config_cache[conf] = pickle.loads(configs[conf])
        core = TokenizerCore(**config_cache[conf])
        try:
            actual = core.tokenize(sql)
            matches = signature(core, sql, actual) == expected
        except Exception:
            matches = False
        outcomes[idx] = matches
        return matches

    if not all(identical(idx) for idx in common):
        return
    for number, required in work:
        if all(identical(idx) for idx in required):
            os.write(channel, f"{number}\n".encode())


def choose(profile, replacements, base, span, candidates, catalog, cap=1.8):
    """Probe only when validated exclusions could pay for the tokenizer CPU."""
    from pathlib import Path
    import tempfile
    from probe import _overlay, _schedule_cost, isolated_numbers

    rows = profile["rows"]
    line_mask = ((1 << (span[1] + 1)) - 1) ^ ((1 << span[0]) - 1)
    direct = profile["unknown"]
    for line, mask in profile["unsafe"].items():
        if span[0] <= line <= span[1]:
            direct |= mask
    common = tuple(idx for idx in profile["shared"] if rows[idx][3] & line_mask)

    def micros(indices):
        return sum(rows[idx][4] for idx in indices)

    shared_cost = micros(common)
    if any(rows[idx][2] is None for idx in common) or shared_cost > cap * 1e6:
        return candidates
    index = {node: i for i, node in enumerate(profile["nodes"])}
    seconds = {t["number"]: t["seconds"] for t in catalog.values()}
    common_set = set(common)
    plans = []
    free = set()
    for n in candidates:
        i = index.get(catalog[n]["id"])
        if i is None or (direct >> i) & 1:
            continue
        required = [
            idx
            for idx in profile["tests"][i]
            if rows[idx][3] & line_mask and idx not in common_set
        ]
        if not required and not common:
            free.add(n)
            continue
        if any(rows[idx][2] is None for idx in required):
            continue
        us = micros(required)
        if us < cap * 1e6:
            plans.append((-(seconds[n] + 0.002) / (us + 200), n, required))
    remaining = candidates - free
    if not plans:
        return remaining
    attempted = []
    seen = set()
    used = shared_cost
    for _, n, required in sorted(plans):
        extra = set(required) - seen
        add = micros(extra)
        if used + add <= cap * 1e6:
            seen.update(extra)
            used += add
            attempted.append((n, sorted(required, key=lambda idx: rows[idx][4])))
    if not attempted:
        return remaining
    saving = _schedule_cost(remaining, seconds) - _schedule_cost(
        remaining - {n for n, _ in attempted}, seconds
    )
    # Include cold interpreter/import CPU and budget for false positives.
    if saving < 0.18 + min(cap, used / 1e6) * 0.65:
        return remaining
    budget = min(cap + 0.15, max(0.24, saving * 0.45))
    with tempfile.TemporaryDirectory(prefix="token-replay-") as tmp:
        view = Path(tmp)
        _overlay(Path(base).resolve(), view, replacements)
        passed = isolated_numbers(
            view,
            replay,
            (profile["configs"], rows, attempted, common),
            budget * 3,
            budget,
            base,
        )
    return remaining - passed


@lru_cache(maxsize=1)
def load_profile(file):
    import gzip
    import marshal

    return marshal.loads(gzip.decompress(file.read_bytes()))


def refine_patch(impact, patch, analysis, candidates, catalog):
    target = "sqlglot/tokenizer_core.py"
    if target not in patch or not candidates:
        return candidates
    from impact import apply, changes

    edits = [edit for edit in changes(patch) if edit.before not in analysis.cosmetic]
    if len(edits) != 1 or edits[0].before != target or edits[0].after != target:
        return candidates
    source = impact.baseline(target)
    updated = analysis.paths.get(target) if analysis.paths else None
    if updated is None:
        updated, _ = apply(edits[0], source)
    import ast

    syntax = impact.syntax.get(target)
    tree = syntax.tree if syntax else ast.parse(source)
    span = mutation_span(tree, updated)
    if not span:
        return candidates
    file = impact.map_directory / "token_boundary.gz"
    if not file.exists():
        return candidates
    profile = load_profile(file)
    if profile["source_hash"] != hashlib.sha256(source.encode()).digest():
        return candidates
    keep = (
        impact.trace.test_numbers(impact.trace.reads.get(target, 0))
        | impact.trace.incomplete
    )
    return choose(
        profile, {target: updated}, impact.root, span, candidates, catalog
    ) | (candidates & keep)
