"""Combine repeat observations of the same pristine test catalog."""
import argparse
import base64
import gzip
import json


def union_mask(x, y):
    a, b = (int.from_bytes(base64.b64decode(value), "little") for value in (x, y))
    value = a | b
    return base64.b64encode(value.to_bytes((value.bit_length() + 7) // 8, "little")).decode()


def merge(a, b):
    if "lines" in a:
        if a["nodes"] != b["nodes"]:
            raise ValueError("different test catalogs")
        for path, lines in b["lines"].items():
            group = a["lines"].setdefault(path, {})
            for line, mask in lines.items():
                group[line] = union_mask(group.get(line, ""), mask)
        for path, mask in b["reads"].items():
            a["reads"][path] = union_mask(a["reads"].get(path, ""), mask)
        # This is supplemental coverage. A complete first run still counts even
        # when a test assumes a particular ordering and fails in the reversed run.
        a["incomplete"] = sorted(set(a["incomplete"]) & set(b["incomplete"]))
    else:
        if a["ids"] != b["ids"]:
            raise ValueError("different test catalogs")
        for feature in ("files", "scalars"):
            for path, sites in b.get(feature, {}).items():
                base = a.setdefault(feature, {}).setdefault(path, {})
                for loc, groups in sites.items():
                    recorded = base.setdefault(loc, {})
                    for key, indices in groups.items():
                        recorded[key] = sorted(set(recorded.get(key, ())) | set(indices))
        a["complete"] = sorted(set(a["complete"]) & set(b["complete"]))
    return a


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("first")
    args.add_argument("second")
    args.add_argument("output")
    paths = args.parse_args()
    with gzip.open(paths.first, "rt") as first, gzip.open(paths.second, "rt") as second:
        data = merge(json.load(first), json.load(second))
    with open(paths.output, "wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
            handle.write(json.dumps(data, separators=(",", ":")).encode())
