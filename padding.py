"""Use free worker capacity for additional coverage without moving required tests."""
from collections import Counter, defaultdict
from pathlib import PurePosixPath
import re

from impact import changes


def words(s):
    return {word for word in re.findall(r"[a-z]{3,}", s.lower())
            if word not in {"self", "test", "true", "false", "return", "none", "from",
                            "sqlglot", "tests", "args", "this", "def", "str", "and", "for"}}


class Padding:
    def __init__(self, impact):
        self.impact = impact
        self.duration = {}
        self.words = {}
        self.domains = defaultdict(set)
        counts = Counter()
        for test in impact.tests:
            n = test["number"]
            self.duration[n] = round(test["seconds"] * 1000)
            terms = words(test["id"].rsplit("::", 1)[-1])
            self.words[n] = terms
            counts.update(terms)
            self.domains[PurePosixPath(test["file"]).stem.removeprefix("test_")].add(n)
        self.weight = {word: 1.0 / (1 + count) ** 0.5 for word, count in counts.items()}
        self.ids = set(self.duration)
        self.zero = sorted(n for n in self.ids if not self.duration[n])

    def _neighbors(self, patch):
        tokens, domain = set(), set()
        proximity = 0
        for change in changes(patch):
            filename = change.before or change.after
            path = PurePosixPath(filename)
            if path.suffix != ".py" or path.parts[0] != "sqlglot":
                continue
            domain.update(self.domains.get(path.stem, ()))
            domain.update(self.domains.get(path.parent.name, ()))
            syntax = self.impact.syntax.get(filename)
            traced = self.impact.trace.lines.get(filename, {})
            for start, hunk in change.hunks:
                row = start + 1
                for sign, content in hunk:
                    if sign != " ":
                        tokens.update(words(content))
                        if syntax:
                            scope, stmt = syntax.contexts(row)
                            if scope:
                                tokens.update(words(getattr(scope, "name", "")))
                            if stmt:
                                proximity |= traced.get(stmt.lineno, 0)
                        proximity |= traced.get(row, 0)
                    row += sign != "+"
        return tokens, domain, self.impact.trace.test_numbers(proximity)

    def extend(self, workers, patch):
        if not workers:
            return workers
        groups = [list(worker) for worker in workers]
        used = {n for worker in groups for n in worker}
        loads = [sum(self.duration[n] for n in worker) for worker in groups]
        peak = max(loads)
        groups[0].extend(n for n in self.zero if n not in used)
        remaining = self.ids - used - set(self.zero)
        if not remaining or min(loads) == peak:
            return groups

        try:
            tokens, domain, proximity = self._neighbors(patch)
        except (ValueError, IndexError):
            tokens, domain, proximity = set(), set(), set()
        candidates = []
        for n in remaining:
            shared = self.words[n] & tokens
            weight = sum(self.weight[t] for t in sorted(shared))
            tier = (2 if n in proximity else 0) + (1 if n in domain else 0)
            # An exact function/feature word can unite tests across dialects.
            if weight >= 0.45:
                tier += 1
            candidates.append((-tier, -weight, self.duration[n], n))
        for _, _, ms, n in sorted(candidates):
            worker = min(range(len(groups)), key=loads.__getitem__)
            if loads[worker] + ms <= peak:
                groups[worker].append(n)
                loads[worker] += ms
        return groups
