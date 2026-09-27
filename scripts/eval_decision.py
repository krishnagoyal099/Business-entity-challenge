#!/usr/bin/env python
"""Compare decision policies on the holdout; write the winner to policy.json.

Reproduces the v1 threshold number first (sanity), then evaluates the Bayes
expected-F policy over a temperature grid. The winner (either kind) is written
to models/verifier_v1/policy.json for predict_test to use.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import local_artifact_path, write_json
from src.candidate_generation import candidates_dir
from src.config import load_config
from src.entity_decision import BayesPolicy, evaluate_policy, load_policy
from src.evaluator import macro_f05
from src.ground_truth import load_ground_truth
from src.logging_utils import setup_logging
from src.model import load_model, predict_scores
from src.pair_features import FEATURE_COLUMNS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate decision policies.")
    p.add_argument("--config", default="configs/aws_cpu.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--features-dir", default=None)
    p.add_argument("--candidates-dir", default=None)
    p.add_argument("--holdout-mod", type=int, default=5)
    p.add_argument("--taus", default="0.6,0.75,0.9,1.0,1.15,1.3,1.5")
    p.add_argument("--n-jobs", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg)
    mdir = local_artifact_path(cfg, "models/verifier_v1/policy.json").parent
    model = load_model(args.model or (mdir / "model.txt"))
    fdir = (Path(args.features_dir) if args.features_dir
            else local_artifact_path(cfg, "features/train_dryrun"))
    cand_dir = (Path(args.candidates_dir) if args.candidates_dir
                else candidates_dir(cfg, "train", dry_run=True))

    t0 = time.time()
    s1_l, pool_l, y_l, X_l = [], [], [], []
    for part in sorted(fdir.glob("part_*.parquet")):
        t = pq.read_table(part)
        s1_l.append(t.column("s1_idx").to_numpy())
        pool_l.append(t.column("pool_idx").to_numpy())
        y_l.append(t.column("label").to_numpy())
        X_l.append(np.column_stack(
            [t.column(c).to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32))
    s1, pool, y = np.concatenate(s1_l), np.concatenate(pool_l), np.concatenate(y_l)
    X = np.vstack(X_l)
    del s1_l, pool_l, y_l, X_l
    hold = (s1 % args.holdout_mod) == 0
    prob = predict_scores(model, X[hold])
    del X
    log.info("holdout scored in %.0fs (%d rows)", time.time() - t0, int(hold.sum()))

    # full-GT truth (same construction as train.py)
    s1_ids = pq.read_table(cand_dir / "s1_ids.parquet").column(0).to_pylist()
    pool_ids = pq.read_table(cand_dir / "pool_ids.parquet").column(0).to_pylist()
    pool_pos = {eid: j for j, eid in enumerate(pool_ids)}
    gt = load_ground_truth(cfg, "train")
    truth: dict = {}
    for i in range(0, len(s1_ids), args.holdout_mod):
        truth[i] = {pool_pos[t] for t in gt.matches.get(s1_ids[i], ())
                    if t in pool_pos}
    del pool_pos, pool_ids, gt

    ceiling: dict = {}
    for a, b, lab in zip(s1[hold].tolist(), pool[hold].tolist(), y[hold].tolist()):
        if lab:
            ceiling.setdefault(a, set()).add(b)
    ceil_rep = macro_f05(ceiling, truth, source_prefixes=())

    old = load_policy(mdir / "policy.json")
    rep0 = evaluate_policy(s1[hold], pool[hold], prob, truth, old, n_jobs=args.n_jobs)
    print(f"\nbaseline policy ({old.to_dict()}): macro_f05={rep0.macro_f05:.4f} "
          f"emitted/entity={(rep0.total_tp + rep0.total_fp) / rep0.n_entities:.2f}")
    print(f"retrieval ceiling:     {ceil_rep.macro_f05:.4f}")

    best_pol, best_rep = old, rep0
    rows = []
    for tau in [float(x) for x in args.taus.split(",")]:
        pol = BayesPolicy(temperature=tau)
        rep = evaluate_policy(s1[hold], pool[hold], prob, truth, pol,
                              n_jobs=args.n_jobs)
        emitted = (rep.total_tp + rep.total_fp) / rep.n_entities
        rows.append({"kind": "bayes", "temperature": tau,
                     "macro_f05": rep.macro_f05, "emitted_per_entity": emitted,
                     "total_fp": rep.total_fp, "total_fn": rep.total_fn})
        print(f"bayes tau={tau:<5}  macro_f05={rep.macro_f05:.4f} "
              f"emitted/entity={emitted:.2f} FP={rep.total_fp} FN={rep.total_fn}")
        if rep.macro_f05 > best_rep.macro_f05:
            best_pol, best_rep = pol, rep

    write_json(mdir / "policy.json", best_pol.to_dict())
    write_json(mdir / "policy_eval.json",
               {"baseline": rep0.macro_f05, "ceiling": ceil_rep.macro_f05,
                "rows": rows, "winner": best_pol.to_dict(),
                "winner_macro_f05": best_rep.macro_f05})
    print(f"\nWINNER: {best_pol.to_dict()}  macro_f05={best_rep.macro_f05:.4f}")
    print(f"verification gap now: {ceil_rep.macro_f05 - best_rep.macro_f05:.4f}")
    print(f"policy written to {mdir / 'policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
