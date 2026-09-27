#!/usr/bin/env python
"""Phase 12: score test features, decide, write the submission files.

Reads the policy KIND from policy.json (threshold or bayes). The Bayes policy
needs each entity's FULL candidate list (low-probability pairs included), so all
scored pairs are kept in memory (~3 GB for 245M pairs) and the policy runs once
at the end.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import local_artifact_path, publish_artifact
from src.candidate_generation import candidates_dir
from src.config import load_config
from src.entity_decision import DecisionPolicy, decide, load_policy
from src.logging_utils import setup_logging, stage_timer
from src.model import load_feature_names, load_model, predict_scores
from src.pair_features import LEGACY_FEATURE_COLUMNS
from src.submission import write_candidate_pairs, write_matching_results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test inference + submission.")
    p.add_argument("--config", default="configs/aws_cpu.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="override for threshold-kind policies only")
    p.add_argument("--exclusive", default=None, choices=("none", "hard", "soft"),
                   help="override the policy's pool-exclusivity mode")
    p.add_argument("--cand-top", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg)
    model_path = args.model or local_artifact_path(cfg, "models/verifier_v1/model.txt")
    model = load_model(model_path)
    feature_cols = load_feature_names(model_path, LEGACY_FEATURE_COLUMNS)
    log.info("model uses %d features", len(feature_cols))
    pol_path = args.policy or local_artifact_path(cfg, "models/verifier_v1/policy.json")
    policy = load_policy(pol_path)
    if args.threshold is not None and isinstance(policy, DecisionPolicy):
        policy.threshold = float(args.threshold)
    if args.exclusive is not None:
        policy.exclusive = args.exclusive
    log.info("policy: %s", policy.to_dict())

    cand_base = candidates_dir(cfg, "test")
    s1_ids = pq.read_table(cand_base / "s1_ids.parquet").column(0).to_pylist()
    pool_ids = pq.read_table(cand_base / "pool_ids.parquet").column(0).to_pylist()
    n_s1 = len(s1_ids)

    fdir = local_artifact_path(cfg, "features/test")
    parts = sorted(fdir.glob("part_*.parquet"))
    if not parts:
        log.error("no test features at %s - run build_features --split test", fdir)
        return 2

    timer = stage_timer("predict_test", cfg)
    timer.start()
    all_s, all_p, all_pr, cand_s, cand_p = [], [], [], [], []
    t0 = time.time()
    for i, part in enumerate(parts):
        t = pq.read_table(part)
        s = t.column("s1_idx").to_numpy()
        pl = t.column("pool_idx").to_numpy()
        X = np.column_stack([t.column(c).to_numpy()
                             for c in feature_cols]).astype(np.float32)
        prob = predict_scores(model, X)
        all_s.append(s)
        all_p.append(pl)
        all_pr.append(prob)
        order = np.lexsort((-prob, s))          # candidates: top-N per entity
        so, po = s[order], pl[order]
        starts = np.r_[0, np.flatnonzero(so[1:] != so[:-1]) + 1]
        sizes = np.diff(np.r_[starts, so.size])
        position = np.arange(so.size) - np.repeat(starts, sizes)
        keepc = position < args.cand_top
        cand_s.append(so[keepc])
        cand_p.append(po[keepc])
        if (i + 1) % 10 == 0 or i == len(parts) - 1:
            el = time.time() - t0
            log.info("scored %d/%d parts (%.0fs, eta %.0fs)", i + 1, len(parts),
                     el, el / (i + 1) * (len(parts) - i - 1))
    ks, kp, kpr = np.concatenate(all_s), np.concatenate(all_p), np.concatenate(all_pr)
    cs, cp = np.concatenate(cand_s), np.concatenate(cand_p)
    del all_s, all_p, all_pr, cand_s, cand_p
    log.info("scored %d pairs; running decision policy...", ks.size)

    # exclusivity sees every S1 entity's pairs at once (all parts concatenated)
    preds_idx = decide(ks, kp, kpr, policy, n_jobs=args.n_jobs)
    timer.stop()

    preds = {s1_ids[a]: {pool_ids[b] for b in bs} for a, bs in preds_idx.items()}
    cands: dict = {}
    for a, b in zip(cs.tolist(), cp.tolist()):
        cands.setdefault(s1_ids[a], set()).add(pool_ids[b])
    del cs, cp
    # matches must stay a subset of candidates (validator soft check)
    for sid, ids in preds.items():
        cands.setdefault(sid, set()).update(ids)

    out = Path(cfg.paths.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = write_matching_results(preds, out / "matching_results.tsv",
                                   required_ids=s1_ids)
    write_candidate_pairs(cands, out / "candidate_pairs.tsv", required_ids=s1_ids,
                          max_ids_per_row=max(args.cand_top, 16))
    for rel in ("matching_results.tsv", "candidate_pairs.tsv"):
        try:
            publish_artifact(cfg, f"submissions/final/{rel}", local_path=out / rel)
        except Exception as exc:
            log.warning("publish failed for %s (%s)", rel, exc)

    print("\n=== PREDICT SUMMARY ===")
    print(f"  policy: {policy.to_dict()}")
    print(f"  entities: {n_s1} | with >=1 match: {len(preds)} "
          f"({100 * len(preds) / max(1, n_s1):.1f}%)")
    print(f"  emitted pairs: {stats['total_ids']} "
          f"(avg {stats['total_ids'] / max(1, n_s1):.2f}/entity)")
    print(f"  files: {out / 'matching_results.tsv'}")
    print("  NEXT: python scripts/validate_submission.py "
          f"--config {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
