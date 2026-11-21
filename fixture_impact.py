"""Measure consumers of SQL files loaded into TestExecutor's class fixtures."""
import ast
from collections import defaultdict
from functools import lru_cache
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

from fixture_projection import FixtureProjection


def fixture_fields(source):
    tree = ast.parse((source / "tests/test_executor.py").read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TestExecutor")
    setup = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "setUpClass")
    fields = {}
    # Only terminal assignments: later class setup must not derive any state from this data.
    for stmt in reversed(setup.body):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            break
        target, outer = stmt.targets[0], stmt.value
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == 'cls' and isinstance(outer, ast.Call)
                and isinstance(outer.func, ast.Name) and outer.func.id == 'list'
                and not outer.keywords and len(outer.args) == 1):
            break
        inner = outer.args[0]
        if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                and inner.func.id == 'load_sql_fixture_pairs' and not inner.keywords
                and len(inner.args) == 1 and isinstance(inner.args[0], ast.Constant)
                and isinstance(inner.args[0].value, str)):
            break
        fields[target.attr] = 'tests/fixtures/' + inner.args[0].value
    return fields


def build(source, destination):
    source = source.resolve()
    sys.path.insert(0, str(source))
    from tests.test_executor import TestExecutor

    fields = fixture_fields(source)
    reading = set()
    active_case = None
    projection = FixtureProjection((source / "tests/test_executor.py").read_text(), "TestExecutor", 3)
    column_reads = defaultdict(set)
    column_map = defaultdict(dict)
    paths = {str(source / file): field for field, file in fields.items()}

    def audit(event, args):
        if active_case is not None and event == 'open' and isinstance(args[0], str):
            field = paths.get(os.path.abspath(args[0]))
            if field:
                reading.add(field)
                column_reads[field].update(projection.all)
    sys.addaudithook(audit)

    class Observed:
        def __init__(self, field, value):
            self.field, self.value = field, value

        def __get__(self, obj, owner):
            reading.add(self.field)
            caller = sys._getframe(1)
            if caller.f_code.co_filename == str(source / "tests/test_executor.py"):
                columns = projection.columns(caller.f_code.co_name, caller.f_lineno, self.field)
            else:
                columns = projection.all
            column_reads[self.field].update(columns)
            return self.value

    independent = defaultdict(list)
    TestExecutor.setUpClass()
    try:
        for field in fields:
            setattr(TestExecutor, field, Observed(field, getattr(TestExecutor, field)))
        for name in unittest.defaultTestLoader.getTestCaseNames(TestExecutor):
            case = TestExecutor(name)
            active_case = name
            reading.clear()
            column_reads.clear()
            result = unittest.TestResult()
            case.run(result)
            active_case = None
            if result.wasSuccessful() and not result.skipped:
                for field, file in fields.items():
                    test_id = f'tests/test_executor.py::TestExecutor::{name}'
                    column_map[file][test_id] = sorted(column_reads[field])
                    if field not in reading:
                        independent[file].append(f'tests/test_executor.py::TestExecutor::{name}')
        # If teardown observes an attribute, it can affect the entire class.
        reading.clear()
        active_case = 'tearDownClass'
        TestExecutor.tearDownClass()
        for field in reading:
            independent.pop(fields[field], None)
            column_map.pop(fields[field], None)
    finally:
        active_case = None
    destination.write_bytes(gzip.compress(json.dumps(independent, sort_keys=True).encode(), mtime=0))
    projections = destination.with_name('fixture_columns.json.gz')
    projections.write_bytes(gzip.compress(json.dumps(column_map, sort_keys=True).encode(), mtime=0))


@lru_cache(maxsize=4)
def observed_columns(directory):
    path = directory / 'fixture_columns.json.gz'
    return json.loads(gzip.decompress(path.read_bytes())) if path.is_file() else {}


@lru_cache(maxsize=4)
def exclusions(directory):
    path = directory / 'fixture_fields.json.gz'
    if path.is_file():
        return json.loads(gzip.decompress(path.read_bytes()))
    return {}


@lru_cache(maxsize=2)
def helper(source):
    # This pinned helper uses only the standard library. Loading it separately
    # avoids importing sqlglot or polluting the isolated test probe's modules.
    spec = importlib.util.spec_from_file_location('fixture_pair_reader', source / 'tests/helpers.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_rows(module, path, content):
    with patch.object(module, 'open', lambda *args, **kw: io.StringIO(content), create=True):
        return list(module.load_sql_fixture_pairs(path))


def comparable_rows(old, new):
    """Keep query structure, metadata and cache semantics for numeric-only edits."""
    if len(old) != len(new):
        return False
    for (meta, sql, expected), (updated_meta, updated_sql, updated_expected) in zip(old, new):
        if meta != updated_meta:
            return False
        for original, updated in ((sql, updated_sql), (expected, updated_expected)):
            if original == updated:
                continue
            if not re.match(r'(?i)^(select|with)\b', original):
                return False
            if re.sub(r'[0-9]+', '0', original) != re.sub(r'[0-9]+', '0', updated):
                return False
    return True


def narrow(root, directory, path, original, content, impacted, ids, single_file=True):
    untouched = exclusions(directory).get(path)
    if untouched is None:
        return impacted
    # Some coverage runners attribute setUpClass reads only to the first test.
    # Start with the complete class, retaining new/unobserved catalog entries.
    selected = impacted.union(number for name, number in ids.items()
                              if name.startswith('tests/test_executor.py::TestExecutor::'))
    module = helper(root)
    # A parse failure would break class setup even for tests that never use this field.
    try:
        before = fixture_rows(module, path, original)
        after = fixture_rows(module, path, content)
        columns = observed_columns(directory).get(path, {}) if single_file else {}
        if columns and len(before) == len(after):
            changed = {i for left, right in zip(before, after)
                       for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]}
            # When no class member sees a changed value, shared state is unchanged.
            # Otherwise retain the existing pure, expected-numeric-edit guard.
            if all(changed.isdisjoint(used) for used in columns.values()) or comparable_rows(before, after):
                return selected.difference(ids[name] for name, used in columns.items()
                                           if name in ids and changed.isdisjoint(used))
        if not comparable_rows(before, after) or any(x[1] != y[1] for x, y in zip(before, after)):
            return selected
    except (ValueError, IndexError, TypeError, UnicodeError):
        return selected
    return selected.difference(ids[name] for name in untouched if name in ids)


if __name__ == '__main__':
    build(Path(sys.argv[1]), Path(sys.argv[2]))
