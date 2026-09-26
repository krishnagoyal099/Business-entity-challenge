#!/usr/bin/env python
"""Phase 4 benchmark: normalization view quality on the REAL train data.

Per view it measures:
- pair_recall: fraction of true (S1, pool) GT pairs whose view key is equal on
  both sides (exact-key blocking proxy; fuzzy channels are measured in Phase 5)
- collision profile on a background pool sample (group sizes, share of records
  in oversized groups, empty-key rate)
- token document-frequency profile of name core tokens

Memory-light: reservoir-samples S1 entities, streams the GT and both pools once.
Ground truth is used for DIAGNOSIS only; no fitted artifact is produced.

Usage:
  python scripts/bench_normalization.py --config configs/local.yaml
  python scripts/bench_normalization.py --config configs/local.yaml \
      --s1-sample 50000 --pool-sample 200000 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import local_artifact_path, mark_stage, stage_complete, write_json
from src.config import config_hash, load_config
from src.data_loader import TsvReader, ensure_source
from src.logging_utils import setup_logging, stage_timer
from src.normalization import build_address_views, build_name_views

# Approximate pool size per source (Phase 2 audit); converts --pool-sample into
# a Bernoulli keep-rate for background sampling.
ESTIMATED_POOL_ROWS_PER_SOURCE = 5_100_000

KEY_VIEWS = (
    ("name_alnum", lambda nv, av: nv.alnum),
    ("name_core_sorted", lambda nv, av: nv.core_sorted),
    ("name_sorted", lambda nv, av: nv.sorted),
    ("name_urls", lambda nv, av: " ".join(nv.urls)),
    ("addr_core_sorted", lambda nv, av: av.core_sorted),
    ("addr_sorted", lambda nv, av: av.sorted),
    ("addr_house", lambda nv, av: av.house),
    ("addr_digits", lambda nv, av: av.digits),
)
UNIONS = (
    ("union_name_core_or_urls", ("name_core_sorted", "name_urls")),
    ("union_name_any", ("name_core_sorted", "name_sorted", "name_urls")),
    ("union_name_or_addr", ("name_core_sorted", "name_sorted", "name_urls",
                            "addr_core_sorted", "addr_sorted")),
)


def _cols(header: List[str]) -> Dict[str, int]:
    def idx(name: str, fallback: int) -> int:
        return header.index(name) if name in header else fallback
    return {"id": idx("entity_id", 0), "name": idx("business_name", 1),
            "addr": idx("business_address", 2)}


def _reservoir_ids(path: Path, k: int, seed: int) -> List[str]:
    reader = TsvReader(path)
    c = _cols(reader.read_header())
    rng = random.Random(seed)
    reservoir: List[str] = []
    seen = 0
    for row in reader.iter_rows():
        seen += 1
        eid = row[c["id"]].strip()
        if len(reservoir) < k:
            reservoir.append(eid)
        else:
            j = rng.randrange(seen)
            if j < k:
                reservoir[j] = eid
    return reservoir


def _stream_selected(path: Path, want: set) -> Dict[str, Tuple[str, str]]:
    reader = TsvReader(path)
    c = _cols(reader.read_header())
    out: Dict[str, Tuple[str, str]] = {}
    for row in reader.iter_rows():
        eid = row[c["id"]].strip()
        if eid in want and eid not in out:
            out[eid] = (row[c["name"]], row[c["addr"]])
    return out


def _stream_gt(path: Path, want: set) -> Dict[str, set]:
    reader = TsvReader(path)
    header = reader.read_header()
    s1_idx = header.index("source1_entity_id")
    m_idx = header.index("matched_entity_ids")
    out: Dict[str, set] = {}
    for row in reader.iter_rows():
        s1 = row[s1_idx].strip()
        if s1 in want:
            out.setdefault(s1, set()).update(
                t.strip() for t in row[m_idx].split(",") if t.strip())
    return out


def _stream_pool(path: Path, needed: set, rate: float, rng: random.Random
                 ) -> Tuple[Dict[str, Tuple[str, str]], int]:
    """Keep rows whose id is needed (GT pairs) plus a Bernoulli background."""
    reader = TsvReader(path)
    c = _cols(reader.read_header())
    out: Dict[str, Tuple[str, str]] = {}
    total = 0
    for row in reader.iter_rows():
        total += 1
        eid = row[c["id"]].strip()
        if eid in needed or rate >= 1.0 or rng.random() < rate:
            if eid not in out:
                out[eid] = (row[c["name"]], row[c["addr"]])
    return out, total


def _collision_stats(keys: List[str]) -> Dict[str, Any]:
    nonempty = [k for k in keys if k]
    sizes = list(Counter(nonempty).values())
    n = len(keys)
    big = sum(s for s in sizes if s >= 100)
    return {
        "n_records": n, "n_keys": len(sizes),
        "empty_frac": round((n - len(nonempty)) / n, 4) if n else 0.0,
        "max_group": max(sizes) if sizes else 0,
        "groups_ge_2": sum(1 for s in sizes if s >= 2),
        "groups_ge_10": sum(1 for s in sizes if s >= 10),
        "groups_ge_100": sum(1 for s in sizes if s >= 100),
        "records_in_ge_100": big,
        "records_in_ge_100_frac": round(big / n, 4) if n else 0.0,
    }


def run_bench(cfg, s1_sample: int, pool_sample: int, seed: int) -> Dict[str, Any]:
    t0 = time.time()
    rng = random.Random(seed)
    log = setup_logging(cfg)

    s1_path = Path(ensure_source(cfg, "train", 1))
    sampled = set(_reservoir_ids(s1_path, s1_sample, seed))
    log.info("sampled %d S1 entities (reservoir, seed=%d)", len(sampled), seed)

    s1_rows = _stream_selected(s1_path, sampled)
    gt = _stream_gt(Path(ensure_source(cfg, "train", "ground_truth")), sampled)
    needed: set = set()
    for ids in gt.values():
        needed.update(ids)
    pairs = [(s1, pid) for s1, ids in gt.items() for pid in ids]
    log.info("GT pairs: %d (%d entities), needed pool ids: %d",
             len(pairs), len(gt), len(needed))

    rate = min(1.0, (pool_sample / 2) / ESTIMATED_POOL_ROWS_PER_SOURCE)
    pool: Dict[str, Tuple[str, str]] = {}
    totals: Dict[str, int] = {}
    for src in (2, 3):
        got, total = _stream_pool(Path(ensure_source(cfg, "train", src)),
                                  needed, rate, rng)
        totals[f"source{src}"] = total
        pool.update(got)
        log.info("source%d: %d rows streamed, kept %d", src, total, len(got))

    def views(rec: Tuple[str, str]):
        return build_name_views(rec[0]), build_address_views(rec[1])

    s1_v = {eid: views(rec) for eid, rec in s1_rows.items()}
    pool_v = {eid: views(rec) for eid, rec in pool.items()}

    view_hits: Dict[str, List[bool]] = {v: [] for v, _ in KEY_VIEWS}
    view_both: Dict[str, int] = {v: 0 for v, _ in KEY_VIEWS}
    for s1, pid in pairs:
        a, b = s1_v.get(s1), pool_v.get(pid)
        for vname, fn in KEY_VIEWS:
            if a is None or b is None:
                view_hits[vname].append(False)
                continue
            k1, k2 = fn(*a), fn(*b)
            if k1 and k2:
                view_both[vname] += 1
            view_hits[vname].append(bool(k1 and k1 == k2))

    n_pairs = len(pairs)
    views_report: Dict[str, Any] = {}
    for vname, fn in KEY_VIEWS:
        hits = view_hits[vname]
        views_report[vname] = {
            "pair_recall": round(sum(hits) / n_pairs, 4) if n_pairs else 0.0,
            "both_nonempty_frac": (round(view_both[vname] / n_pairs, 4)
                                   if n_pairs else 0.0),
            "collision": _collision_stats([fn(*pool_v[e]) for e in pool_v]),
        }
    unions_report = {
        uname: {"pair_recall": round(
            sum(any(view_hits[m][i] for m in members) for i in range(n_pairs))
            / n_pairs, 4) if n_pairs else 0.0}
        for uname, members in UNIONS}

    tok_df: Counter = Counter()
    for nv, _ in pool_v.values():
        tok_df.update(set(nv.core))
    counts = list(tok_df.values())
    n_tok = len(counts)
    tokens_report = {
        "distinct": n_tok,
        "top15": [[t, c] for t, c in tok_df.most_common(15)],
        "frac_df_le_5": round(sum(1 for c in counts if c <= 5) / n_tok, 4) if n_tok else 0.0,
        "frac_df_le_20": round(sum(1 for c in counts if c <= 20) / n_tok, 4) if n_tok else 0.0,
    }
    return {
        "meta": {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "seed": seed, "s1_sample": s1_sample, "pool_sample": pool_sample,
                 "rate": round(rate, 6), "elapsed_s": round(time.time() - t0, 1),
                 "pool_rows_seen": totals},
        "sampling": {"s1_entities": len(sampled), "s1_rows_found": len(s1_rows),
                     "gt_entities": len(gt), "gt_pairs": n_pairs,
                     "needed_pool_ids": len(needed),
                     "needed_pool_ids_found": sum(1 for p in needed if p in pool),
                     "pool_background_records": len(pool)},
        "views": views_report, "unions": unions_report, "tokens": tokens_report,
    }


def render_text(report: Dict[str, Any]) -> str:
    m, s = report["meta"], report["sampling"]
    lines = ["=" * 78, "NORMALIZATION BENCHMARK (Phase 4)",
             f"generated: {m['generated_at']} | seed: {m['seed']} | "
             f"elapsed: {m['elapsed_s']}s",
             f"S1 entities sampled: {s['s1_entities']} | GT pairs: {s['gt_pairs']} "
             f"| pool background: {s['pool_background_records']}",
             f"needed pool ids found: {s['needed_pool_ids_found']}/{s['needed_pool_ids']}",
             "=" * 78, "",
             f"{'view':<22}{'recall':>9}{'bothne%':>9}{'keys':>10}{'maxgrp':>8}"
             f"{'>=100':>7}{'rec>=100%':>11}{'empty%':>9}"]
    for vname, v in report["views"].items():
        c = v["collision"]
        lines.append(f"{vname:<22}{v['pair_recall']:>9}"
                     f"{100 * v['both_nonempty_frac']:>9.1f}{c['n_keys']:>10}"
                     f"{c['max_group']:>8}{c['groups_ge_100']:>7}"
                     f"{100 * c['records_in_ge_100_frac']:>11.1f}"
                     f"{100 * c['empty_frac']:>9.1f}")
    lines += ["", "UNION VIEWS (exact-key OR):"]
    for uname, u in report["unions"].items():
        lines.append(f"  {uname:<32} pair_recall = {u['pair_recall']}")
    t = report["tokens"]
    lines += ["", f"TOKEN DF (name core, background pool): distinct={t['distinct']}",
              f"  frac df<=5: {t['frac_df_le_5']} | frac df<=20: {t['frac_df_le_20']}",
              f"  top: {t['top15']}", "", "NOTES:",
              "  - pair_recall is EXACT key equality (blocking proxy).",
              "  - 'rec>=100%' = share of pool records in groups of >=100."]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Normalization view benchmark.")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--s1-sample", type=int, default=50_000)
    p.add_argument("--pool-sample", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    key = hashlib.sha1(json.dumps(
        {"cfg": config_hash(cfg, "norm_bench"), "s1": args.s1_sample,
         "pool": args.pool_sample, "seed": args.seed},
        sort_keys=True).encode()).hexdigest()[:12]
    if not args.force and stage_complete(cfg, "norm_bench", expected_hash=key):
        log.info("norm_bench already complete; use --force to rerun")
        return 0

    timer = stage_timer("norm_bench", cfg)
    timer.start()
    report = run_bench(cfg, args.s1_sample, args.pool_sample, args.seed)
    timer.stop()

    out_dir = local_artifact_path(cfg, "norm_bench")
    write_json(out_dir / "bench.json", report)
    (out_dir / "bench.txt").write_text(render_text(report), encoding="utf-8")
    mark_stage(cfg, "norm_bench",
               details={"s1_entities": report["sampling"]["s1_entities"],
                        "gt_pairs": report["sampling"]["gt_pairs"],
                        "unions": {k: v["pair_recall"]
                                   for k, v in report["unions"].items()}},
               artifacts=["norm_bench/bench.json", "norm_bench/bench.txt"],
               config_hash=key, duration_s=timer.seconds,
               peak_rss_mb=timer.peak_rss_mb)
    print(render_text(report))
    print(f"\nfull report: {out_dir / 'bench.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
