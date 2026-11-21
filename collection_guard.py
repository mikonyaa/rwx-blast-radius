"""Verify collection on risky edits using the same numbered unittest modules as pytest."""

import ast
import importlib
import os
from pathlib import Path
import tempfile

from impact import PROCEDURES, Syntax, apply, changes
from probe import _overlay, isolated_numbers


def _module_name(path):
    name = path.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def _observe(groups, modified_tests, imports, output):
    # A single failed source-module import is enough to cover all cold test
    # modules known to import it, even if retrying every import would time out.
    covered = set()
    for source, group_indices in imports.items():
        try:
            importlib.import_module(_module_name(source))
        except (Exception, SystemExit):
            for idx in group_indices - covered:
                os.write(output, f"{idx}\n".encode())
            covered.update(group_indices)

    for i, (filename, records) in enumerate(groups):
        if i in covered:
            continue
        try:
            module = importlib.import_module(_module_name(filename))
            if filename in modified_tests:
                # A renamed class or removed method no longer collects at its
                # original numbered node ID.
                for record in records:
                    _, cls, method = record["id"].split("::")
                    if not callable(getattr(getattr(module, cls), method)):
                        raise AttributeError(method)
        except (Exception, SystemExit):
            os.write(output, f"{i}\n".encode())


class CollectionGuard:
    def __init__(self, impact):
        self.impact = impact
        self.records = {test["number"]: test for test in impact.tests}

    def _target_mask(self, change):
        impact, trace = self.impact, self.impact.import_trace
        path = change.before or change.after
        if not path.endswith(".py"):
            return 0
        importers = trace.imports.get(path, 0)
        if not importers:
            return 0
        if change.before != change.after:
            return importers
        source = impact.baseline(path)
        _, blocks = apply(change, source)
        if path not in impact.syntax:
            impact.syntax[path] = Syntax(source, path)
        syntax = impact.syntax[path]
        observed = trace.lines.get(path, {})
        targeted = 0
        for start, removed, added in blocks:
            if all(not part.strip() or part.lstrip().startswith("#") for part in (*removed, *added)):
                continue
            for line in range(start, start + max(1, len(removed))):
                scope, _ = syntax.contexts(line)
                if isinstance(scope, ast.Lambda) or not isinstance(scope, PROCEDURES):
                    return importers
                if line < scope.body[0].lineno or line <= scope.lineno:
                    return importers
                if not removed:
                    # Newly inserted code can execute on any import that calls this function.
                    targeted |= impact.interval(observed, scope.lineno, scope.end_lineno)
                else:
                    targeted |= observed.get(line, 0)
        return targeted

    def augment(self, patch, analysis, selected):
        staged = analysis.paths
        if not staged or not self.impact.import_trace or len(selected) == len(self.records):
            return selected
        sources = {}
        try:
            for change in changes(patch):
                mask = self._target_mask(change)
                if mask:
                    sources[change.before or change.after] = self.impact.import_trace.test_numbers(mask)
            missing = set().union(*sources.values()) - selected
            if not missing:
                return selected
            groups = {}
            for number in sorted(missing):
                record = self.records[number]
                groups.setdefault(record["file"], []).append(record)
            # A long native dependency import should not prevent checking the
            # remaining, lightweight modules.
            ordered = sorted(groups.items(), key=lambda item: (
                item[0] in ("tests/test_executor.py", "tests/test_optimizer.py"), item[0]
            ))
            source_groups = {
                source: {idx for idx, (_, rows) in enumerate(ordered)
                         if any(record["number"] in numbers for record in rows)}
                for source, numbers in sources.items()
            }
            with tempfile.TemporaryDirectory(prefix="rwx-collect-") as temp:
                view = Path(temp)
                _overlay(self.impact.root.resolve(), view, staged)
                failed = isolated_numbers(view, _observe, (ordered, set(staged), source_groups), 2.3, 0.65,
                                          base=self.impact.root)
            expanded = set(selected)
            for idx in failed:
                expanded.update(self.impact.files[ordered[idx][0]])
            return expanded
        except (OSError, ValueError, KeyError, SyntaxError, RecursionError):
            # This guard only adds independently observed collection errors.
            return selected
