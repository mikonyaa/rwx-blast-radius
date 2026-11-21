"""Select tests that read a changed token type from a pristine scanner table."""

import ast
import base64
import gzip
import json
import re
from pathlib import Path


def table_definitions(tree, origins):
    result = {}

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef):
                name = prefix + node.name
                visit(node.body, name + ".")
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            else:
                continue
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                key = (prefix[:-1], target.id)
                if key in origins:
                    result[key] = node.value

    visit(tree.body, "")
    return result


def token_name(expr):
    if (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "TokenType"
    ):
        return expr.attr
    return None


def one_entry_change(old, new, enums):
    """Isolate one inserted, removed or renamed literal token at the same order."""

    def entries(node):
        return [
            (ast.dump(k) if k else None, ast.dump(v))
            for k, v in zip(node.keys, node.values)
        ]

    left, right = entries(old), entries(new)
    start = 0
    while start < min(len(left), len(right)) and left[start] == right[start]:
        start += 1
    end_a, end_b = len(left), len(right)
    while start < min(end_a, end_b) and left[end_a - 1] == right[end_b - 1]:
        end_a -= 1
        end_b -= 1
    if max(end_a - start, end_b - start) != 1:
        return None
    words = []
    for node, end in ((old, end_a), (new, end_b)):
        if end == start:
            continue
        key = node.keys[start]
        if not (
            isinstance(key, ast.Constant)
            and type(key.value) is str
            and re.fullmatch(r"[A-Z][A-Z0-9_]{0,39}", key.value)
            and token_name(node.values[start]) in enums
        ):
            return None
        words.append(key.value)
    return words if len(set(words)) == len(words) else None


def decode(value):
    return int.from_bytes(base64.b64decode(value), "little")


class TokenizerImpact:
    def __init__(self, directory, tests):
        self.origins = {}
        path = Path(directory) / "tokenizer.json.gz"
        if not path.is_file():
            return
        data = json.loads(gzip.decompress(path.read_bytes()))
        ids = {t["id"]: t["number"] for t in tests}
        self.numbers = [ids.get(node) for node in data["nodes"]]
        self.unmapped = set(ids.values()) - set(self.numbers)
        self.unknown = decode(data["unknown"])
        self.uses = data["uses"]
        self.enums = set(data["enums"])
        self.words = data.get("words", {})
        self.reserved = data.get("reserved")
        self.blocked = set(data.get("blocked", ()))
        self.punctuation = set(data.get("punctuation", ()))
        self.origins = {
            path: {(cls, attr): family for cls, attr, family in rows}
            for path, rows in data["origins"].items()
        }

    def candidates(self, path, tree, updated_source):
        origins = self.origins.get(path)
        if not origins:
            return None
        updated = ast.parse(updated_source, filename=path)
        before = table_definitions(tree, origins)
        after = table_definitions(updated, origins)
        mask = self.unknown
        edited = False
        for key, original in before.items():
            new = after.get(key)
            if new is None:
                continue
            observed = self.uses.get(origins[key] + ":" + key[1])
            if observed is None:
                continue
            root_table = path == "sqlglot/tokens.py"
            if root_table and key[1] == "SINGLE_TOKENS":
                old_types = {token_name(node) for node in original.values}
                new_types = {token_name(node) for node in new.values}
                if None in old_types | new_types or self.reserved is None:
                    continue
                # SELECT is present independently and IDENTIFIER is always removed.
                changed = (old_types ^ new_types) - {"SELECT", "IDENTIFIER"}
                if changed:
                    mask |= decode(self.reserved.get("*", ""))
                for spelling in changed:
                    mask |= decode(self.reserved.get(spelling, ""))
            if key[1] == "KEYWORDS" and origins[key] in self.words:
                changes = one_entry_change(original, new, self.enums)
                if changes and all(
                    word not in self.blocked
                    and not any(c in word for c in self.punctuation)
                    for word in changes
                ):
                    for word in changes:
                        mask |= decode(self.words[origins[key]].get(word, ""))
                        if root_table:
                            mask |= decode(self.words.get("jsonpath", {}).get(word, ""))
                    # No other differences in this dictionary are accepted.
                    new.keys, new.values = original.keys, original.values
                    edited = True
                    continue
            if len(original.keys) != len(new.keys):
                continue
            for literal_key, old_value, new_value in zip(
                original.keys, original.values, new.values
            ):
                previous = token_name(old_value)
                current = token_name(new_value)
                if not (
                    previous in self.enums
                    and current in self.enums
                    and current != previous
                    and isinstance(literal_key, ast.Constant)
                    and isinstance(literal_key.value, str)
                ):
                    continue
                # Look up the original type across every effective tokenizer table of
                # this family. Copies retain the key and value, including inherited dialects.
                signature = literal_key.value + ":" + previous
                mask |= decode(observed.get(signature, ""))
                if root_table:
                    mask |= decode(
                        self.uses.get("jsonpath:" + key[1], {}).get(signature, "")
                    )
                edited = True
                new_value.attr = previous
        # Restoring only the recognized value substitutions must undo *all* changes.
        # In particular a new keyword/key or a missing enum member takes the fallback.
        if not edited or ast.dump(updated) != ast.dump(tree):
            return None
        selected = self.unmapped.copy()
        while mask:
            bit = mask & -mask
            number = self.numbers[bit.bit_length() - 1]
            if number is not None:
                selected.add(number)
            mask ^= bit
        return selected
