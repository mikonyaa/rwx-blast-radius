"""Collect CPython instruction coverage for literals and runtime operators."""

import ast
from collections import defaultdict
import dis
import gzip
import json
import os
from pathlib import Path
import sys
from types import CodeType

from expression_impact import position, scalar


class RuntimeSites(ast.NodeVisitor):
    def __init__(self):
        self.rows = {}
        self.parent = None
        self.in_function = False

    def visit(self, node):
        parent = self.parent
        self.parent = node
        if self.in_function:
            kind = None
            if scalar(node) and isinstance(
                parent,
                (
                    ast.Return,
                    ast.Assign,
                    ast.AnnAssign,
                    ast.Subscript,
                    ast.keyword,
                    ast.Call,
                    ast.BinOp,
                    ast.AugAssign,
                    ast.Compare,
                ),
            ):
                kind = "literal"
            elif isinstance(node, ast.Compare) and len(node.ops) == 1:
                kind = "compare"
            elif isinstance(node, ast.BinOp):
                kind = "binary"
            elif isinstance(node, ast.UnaryOp):
                kind = "unary"
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                kind = "attr"
            if kind:
                spot = (
                    node.lineno,
                    node.end_lineno,
                    node.col_offset,
                    node.end_col_offset,
                )
                self.rows[spot] = (position(node, kind), kind, node)
        super().visit(node)
        self.parent = parent

    def visit_FunctionDef(self, node):
        old = self.in_function
        self.in_function = True
        for child in node.body:
            self.visit(child)
        self.in_function = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        old = self.in_function
        self.in_function = True
        self.visit(node.body)
        self.in_function = old

    def visit_ClassDef(self, node):
        old = self.in_function
        self.in_function = False
        for child in node.body:
            self.visit(child)
        self.in_function = old


def build(root, catalog, destination):
    import pytest
    from coverage_build import Collector

    root = root.resolve()
    nodes = [r["id"] for r in json.loads(catalog.read_bytes())["tests"]]
    collector = Collector(root, nodes)
    code_locations = {}
    declared = defaultdict(list)
    sites = []
    indices = {}

    def index_code(code, path, potential):
        offsets = {}
        for instr in dis.get_instructions(code):
            found = potential.get(tuple(instr.positions))
            if found is None:
                continue
            key, kind, node = found
            name = instr.opname
            if kind == "literal":
                if name not in {"LOAD_CONST", "RETURN_CONST"}:
                    continue
                if (
                    type(instr.argval) is not type(node.value)
                    or instr.argval != node.value
                ):
                    continue
            elif kind == "compare" and name not in {
                "COMPARE_OP",
                "CONTAINS_OP",
                "IS_OP",
            }:
                continue
            elif kind == "binary" and name != "BINARY_OP":
                continue
            elif kind == "unary" and not name.startswith("UNARY_"):
                continue
            elif kind == "attr" and (name != "LOAD_ATTR" or instr.argval != node.attr):
                continue
            index = indices.get((path, key))
            if index is None:
                index = len(sites)
                sites.append(0)
                indices[path, key] = index
            offsets[instr.offset] = index
        if offsets:
            code_locations[code] = offsets
            declared[(code.co_filename, code.co_qualname, code.co_firstlineno)].append(
                offsets
            )
        for child in code.co_consts:
            if isinstance(child, CodeType):
                index_code(child, path, potential)

    for file in sorted((root / "sqlglot").rglob("*.py")):
        source = file.read_bytes()
        visitor = RuntimeSites()
        visitor.visit(ast.parse(source))
        code = compile(source, str(file), "exec", dont_inherit=True)
        index_code(code, file.relative_to(root).as_posix(), visitor.rows)

    cache = {}
    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "expression coverage")

    def start(code, offset):
        offsets = code_locations.get(code)
        if offsets:
            cache[id(code)] = offsets
            mon.set_local_events(tool, code, mon.events.INSTRUCTION)
        else:
            # Recompiled or rewritten Python code can have a different instruction
            # layout. A lexical name match alone does not justify a negative hit.
            for defined in declared.get(
                (code.co_filename, code.co_qualname, code.co_firstlineno), ()
            ):
                for site_index in defined.values():
                    sites[site_index] = -1
        return mon.DISABLE

    def instruction(code, offset):
        index = cache[id(code)].get(offset)
        if index is not None:
            sites[index] |= collector.mask or -1
        return mon.DISABLE

    complete = set()

    def audit(event, args):
        if (
            event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}
            and collector.mask
        ):
            complete.add(collector.mask)

    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(SKIP_INTEGRATION="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    mon.register_callback(tool, mon.events.PY_START, start)
    mon.register_callback(tool, mon.events.INSTRUCTION, instruction)
    mon.set_events(tool, mon.events.PY_START)
    sys.addaudithook(audit)
    try:
        status = pytest.main(
            ["-q", "-p", "no:cacheprovider", "--tb=short", "tests"], plugins=[collector]
        )
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)
    if status not in (0, 1) or not collector.passed:
        raise RuntimeError(status)
    unknown_mask = 0
    for mask in complete:
        unknown_mask |= mask
    confirmed = [
        i
        for i in sorted(collector.passed - collector.failed)
        if not unknown_mask & (1 << i)
    ]
    rows = defaultdict(dict)
    for (path, key), i in indices.items():
        rows[path][key] = "*" if sites[i] < 0 else format(sites[i], "x")
    payload = {"nodes": nodes, "complete": confirmed, "rows": rows}
    destination.write_bytes(
        gzip.compress(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
    )
    print(
        len(confirmed),
        "complete tests;",
        len(indices),
        "expression sites;",
        sum(bool(i) for i in sites),
        "hit",
    )


if __name__ == "__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
