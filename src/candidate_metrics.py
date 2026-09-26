"""Retrieval quality measurement (Phase 5).

recall@k, per-channel recall, candidate-budget statistics: the numbers that set
k and the channel mix. GT is used for DIAGNOSIS on train only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .aws_utils import publish_artifact, write_json
from .candidate_generation import (CHANNEL_BIT, CHANNELS, RANK_COLUMNS,
                                   SCORE_COLUMNS, candidates_dir, candidates_rel)

log = logging.getLogger(__name__)

KS = (5, 10, 25, 50, 100, 150)


def _read_ids(path: Path) -> np.ndarray:
    t = pq.read_table(path, columns=["entity_id"])
    return np.asarray(t.column(0).to_pylist(), dtype=object)


def load_candidate_table(cfg, split: str, columns: Optional[List[str]] = None,
                         dry_run: bool = False) -> pd.DataFrame:
    parts = sorted(candidates_dir(cfg, split, dry_run).glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=columns or [])
    return pd.concat([pq.read_table(p, columns=columns).to_pandas()
                      for p in parts], ignore_index=True)


def retrieval_report(cfg, split: str, gt,
                     restrict_s1_ids: Optional[Set[str]] = None,
                     dry_run: bool = False,
                     log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    log = log or logging.getLogger(__name__)
    base = candidates_dir(cfg, split, dry_run)
    s1_ids = _read_ids(base / "s1_ids.parquet")
    pool_ids = _read_ids(base / "pool_ids.parquet")
    n_pool, n_s1 = int(pool_ids.size), int(s1_ids.size)
    s1_pos = {eid: i for i, eid in enumerate(s1_ids.tolist())}
    pool_pos = {eid: i for i, eid in enumerate(pool_ids.tolist())}
    mask = None
    if restrict_s1_ids is not None:
        mask = np.array([eid in restrict_s1_ids for eid in s1_ids.tolist()])

    keys: List[int] = []
    entity_true = np.zeros(n_s1, dtype=np.int64)
    unmapped = 0
    for s1, toks in gt.matches.items():
        i = s1_pos.get(s1)
        if i is None or (mask is not None and not mask[i]):
            continue
        for t in toks:
            j = pool_pos.get(t)
            if j is None:
                unmapped += 1
                continue
            keys.append(i * n_pool + j)
            entity_true[i] += 1
    gt_keys = np.unique(np.asarray(keys, dtype=np.int64)) if keys \
        else np.empty(0, np.int64)
    n_true = int(gt_keys.size)

    total_pairs = hits = 0
    chan_pairs = {c: 0 for c in CHANNELS}
    chan_hits = {c: 0 for c in CHANNELS}
    cand_counts = np.zeros(n_s1, dtype=np.int64)
    hit_counts = np.zeros(n_s1, dtype=np.int64)
    hits_at_k = {k: 0 for k in KS}
    col_names = (["s1_idx", "pool_idx", "channel_bits"]
                 + [RANK_COLUMNS[c] for c in CHANNELS]
                 + [SCORE_COLUMNS[c] for c in SCORE_COLUMNS])

    for path in sorted(base.glob("part_*.parquet")):
        t = pq.read_table(path)
        arrays = {name: t.column(name).to_numpy() for name in col_names}
        if mask is not None:
            sel = mask[arrays["s1_idx"]]
            if not sel.any():
                continue
            arrays = {k: v[sel] for k, v in arrays.items()}
        s1_idx, pool_idx = arrays["s1_idx"], arrays["pool_idx"]
        pair_keys = s1_idx.astype(np.int64) * n_pool + pool_idx.astype(np.int64)
        if n_true:
            pos = np.minimum(np.searchsorted(gt_keys, pair_keys), n_true - 1)
            is_true = gt_keys[pos] == pair_keys
        else:
            is_true = np.zeros(pair_keys.size, dtype=bool)
        total_pairs += int(pair_keys.size)
        hits += int(is_true.sum())
        bits = arrays["channel_bits"]
        for c in CHANNELS:
            b = (bits & CHANNEL_BIT[c]) != 0
            chan_pairs[c] += int(b.sum())
            chan_hits[c] += int((b & is_true).sum())
        cand_counts += np.bincount(s1_idx, minlength=n_s1)
        hit_counts += np.bincount(s1_idx[is_true], minlength=n_s1)
        # union ordering: entity, then best (min) channel rank, then max score
        rk = np.minimum.reduce([np.where(arrays[RANK_COLUMNS[c]] > 0,
                                         arrays[RANK_COLUMNS[c]], 255)
                                for c in CHANNELS])
        sc = np.maximum.reduce([arrays[SCORE_COLUMNS[c]] for c in SCORE_COLUMNS])
        order = np.lexsort((-sc.astype(np.float32), rk, s1_idx))
        s_sorted = s1_idx[order]
        if s_sorted.size:
            starts = np.r_[0, np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1]
            sizes = np.diff(np.r_[starts, s_sorted.size])
            position = np.arange(s_sorted.size) - np.repeat(starts, sizes)
            true_sorted = is_true[order]
            for k in KS:
                hits_at_k[k] += int((true_sorted & (position < k)).sum())

    denom = mask if mask is not None else np.ones(n_s1, dtype=bool)
    with_true = entity_true > 0
    complete = (hit_counts >= entity_true) & with_true
    n_denom = int(denom.sum())
    report: Dict[str, Any] = {
        "split": split, "n_pool": n_pool, "n_s1": n_denom,
        "n_true_pairs": n_true, "unmapped_gt_pairs": unmapped,
        "total_candidate_pairs": total_pairs,
        "avg_candidates_per_entity":
            round(float(cand_counts[denom].mean()), 2) if n_denom else 0.0,
        "p50_candidates_per_entity":
            round(float(np.percentile(cand_counts[denom], 50)), 2) if n_denom else 0.0,
        "p95_candidates_per_entity":
            round(float(np.percentile(cand_counts[denom], 95)), 2) if n_denom else 0.0,
        "pair_recall": round(hits / n_true, 4) if n_true else None,
        "entity_complete_recall":
            round(float(complete.sum() / with_true.sum()), 4) if with_true.any() else None,
        "entity_any_recall":
            round(float((hit_counts[with_true] > 0).sum() / with_true.sum()), 4)
            if with_true.any() else None,
        "per_channel": {c: {"pairs": chan_pairs[c],
                            "pair_recall": (round(chan_hits[c] / n_true, 4)
                                            if n_true else None)}
                        for c in CHANNELS},
        "recall_at_k": {str(k): (round(hits_at_k[k] / n_true, 4) if n_true else None)
                        for k in KS},
        "reduction_ratio": (round(total_pairs / (n_denom * n_pool), 8)
                            if n_pool and n_denom else None),
    }
    log.info("retrieval: pair_recall=%s avg_cand/entity=%s",
             report["pair_recall"], report["avg_candidates_per_entity"])
    return report


def render_text(report: Dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        f"RETRIEVAL REPORT ({report['split']})",
        f"pool={report['n_pool']} s1={report['n_s1']} "
        f"true_pairs={report['n_true_pairs']} (unmapped={report['unmapped_gt_pairs']})",
        f"candidate pairs: {report['total_candidate_pairs']} "
        f"| avg/entity={report['avg_candidates_per_entity']} "
        f"p50={report['p50_candidates_per_entity']} "
        f"p95={report['p95_candidates_per_entity']}",
        f"pair_recall={report['pair_recall']} "
        f"entity_complete={report['entity_complete_recall']} "
        f"entity_any={report['entity_any_recall']}",
        "-" * 70,
        f"{'channel':<12}{'pairs':>12}{'pair_recall':>14}",
    ]
    for c, v in report["per_channel"].items():
        lines.append(f"{c:<12}{v['pairs']:>12}{str(v['pair_recall']):>14}")
    lines += ["-" * 70, "recall@k (union ordering: min rank, then max score):"]
    for k, v in report["recall_at_k"].items():
        lines.append(f"  @{k:<4} {v}")
    lines.append(f"reduction_ratio={report['reduction_ratio']}")
    return "\n".join(lines)


def write_report(cfg, split: str, report: Dict[str, Any],
                 dry_run: bool = False) -> List[str]:
    base = candidates_dir(cfg, split, dry_run)
    write_json(base / "retrieval_report.json", report)
    (base / "retrieval_report.txt").write_text(render_text(report), encoding="utf-8")
    rel = candidates_rel(split, dry_run)
    out = [f"{rel}/retrieval_report.json", f"{rel}/retrieval_report.txt"]
    if not dry_run:
        for r in out:
            publish_artifact(cfg, r)
    return out
