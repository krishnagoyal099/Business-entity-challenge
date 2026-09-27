#!/usr/bin/env python
"""Per-country breakdown of a submission (no labels needed).

Train truth averages ~3.46 matches per S1 entity with ~5.6% singletons; a
country far below that emit rate is losing recall (e.g. an unseen country).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict


def _read_lists(path, col):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            v = row.get(col) or ""
            out[row["source1_entity_id"]] = [x for x in v.split(",") if x]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matching", default="output/matching_results.tsv")
    p.add_argument("--candidate", default="output/candidate_pairs.tsv")
    p.add_argument("--source1", default="dataset/test/test_source1.tsv")
    a = p.parse_args(argv)
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)
    country = {}
    with open(a.source1, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            country[row["entity_id"]] = row.get("country") or "?"
    match = _read_lists(a.matching, "matched_entity_ids")
    cand = _read_lists(a.candidate, "candidate_entity_ids")
    agg = defaultdict(lambda: [0, 0, 0, 0])   # n, with_match, emitted, cands
    for sid, c in country.items():
        m = match.get(sid, [])
        g = agg[c]
        g[0] += 1
        g[1] += bool(m)
        g[2] += len(m)
        g[3] += len(cand.get(sid, []))
    print(f"{'country':<10}{'entities':>10}{'%matched':>10}{'emit/ent':>10}"
          f"{'cand/ent':>10}")
    for c, (n, w, e, k) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:<10}{n:>10}{100 * w / n:>9.1f}%{e / n:>10.2f}{k / n:>10.1f}")
    print("train reference: ~94.4% matched, ~3.46 true matches/entity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
