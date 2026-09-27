#!/usr/bin/env python
"""Stage 2: group-aware re-scoring of the top-K candidates per S1 entity.

train:   K-fold stage-1 models (by s1 % folds) give out-of-fold pair probs on
         the train sample; group features (src/group_features.py) are built on
         the top-K pairs; a stage-2 LightGBM is trained on non-holdout entities
         and the decision policy (threshold/Bayes x exclusivity) is tuned on
         the holdout. Stage-1 OOF baseline on the same holdout is printed.
predict: the fold models' MEAN prob scores test pairs (same distribution as
         OOF), top-K per entity -> group features -> stage 2 -> policy ->
         matching_results.tsv + candidate_pairs.tsv (= the top-K set scored).
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
from src.blocking import _list_parts, normalized_dir
from src.candidate_generation import candidates_dir
from src.config import load_config
from src.entity_decision import (EXCLUSIVE_MODES, BayesPolicy, decide,
                                 evaluate_policy, load_policy, tune_policy)
from src.group_features import (GROUP_COLUMNS, compute_group_features,
                                topk_per_entity)
from src.ground_truth import load_ground_truth
from src.logging_utils import setup_logging
from src.model import (load_feature_names, load_model, predict_scores,
                       save_feature_names, save_model, train_lgbm)
from src.pair_features import FEATURE_COLUMNS, build_side_table
from src.submission import write_candidate_pairs, write_matching_results

FOLD_PARAMS = {"learning_rate": 0.1}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "predict"])
    p.add_argument("--config", default="configs/aws_cpu.yaml")
    p.add_argument("--model-dir", default="models/verifier_v4")
    p.add_argument("--features-dir", default=None,
                   help="train: default features/train_dryrun; predict: features/test")
    p.add_argument("--candidates-dir", default=None)
    p.add_argument("--holdout-mod", type=int, default=5)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--fold-trees", type=int, default=600)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--taus", default="0.75,1.0,1.3")
    p.add_argument("--exclusive", default=None, choices=EXCLUSIVE_MODES,
                   help="predict: override the tuned exclusivity mode")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--reuse-folds", default=None,
                   help="train: model dir whose stage1_fold*.txt to reuse")
    p.add_argument("--leaves", type=int, default=None,
                   help="train: stage-2 num_leaves override")
    return p.parse_args(argv)


def _load_parts(parts, cols, with_label):
    s_l, p_l, y_l, X_l = [], [], [], []
    for part in parts:
        t = pq.read_table(part)
        s_l.append(t.column("s1_idx").to_numpy())
        p_l.append(t.column("pool_idx").to_numpy())
        if with_label:
            y_l.append(t.column("label").to_numpy())
        X_l.append(np.column_stack([t.column(c).to_numpy() for c in cols])
                   .astype(np.float32))
    y = np.concatenate(y_l) if with_label else None
    return np.concatenate(s_l), np.concatenate(p_l), y, np.vstack(X_l)


def _pool_side(cfg, split, log):
    t0 = time.time()
    paths = (_list_parts(normalized_dir(cfg, split, "source2"))
             + _list_parts(normalized_dir(cfg, split, "source3")))
    side = build_side_table(paths)
    log.info("pool side table (%d rows) in %.0fs", len(side["addr_core"]),
             time.time() - t0)
    return side


def _stage2_matrix(X, group):
    return np.column_stack([X] + [group[c] for c in GROUP_COLUMNS]).astype(np.float32)


def train(args, cfg, log) -> int:
    mdir = local_artifact_path(cfg, args.model_dir + "/model.txt").parent
    fdir = (Path(args.features_dir) if args.features_dir
            else local_artifact_path(cfg, "features/train_dryrun"))
    cand_dir = (Path(args.candidates_dir) if args.candidates_dir
                else candidates_dir(cfg, "train", dry_run=True))
    parts = sorted(fdir.glob("part_*.parquet"))
    if not parts:
        log.error("no feature parts at %s", fdir)
        return 2
    cols = list(FEATURE_COLUMNS)
    t0 = time.time()
    s1, pool, y, X = _load_parts(parts, cols, True)
    log.info("loaded %d pairs (%d pos) in %.0fs", s1.size, int(y.sum()),
             time.time() - t0)

    # ---- stage 1: out-of-fold probabilities ----------------------------
    oof = np.zeros(s1.size, dtype=np.float32)
    fold = s1 % args.folds
    rdir = (local_artifact_path(cfg, args.reuse_folds + "/model.txt").parent
            if args.reuse_folds else None)
    for f in range(args.folds):
        t0 = time.time()
        tr = fold != f
        if rdir is not None:
            m = load_model(rdir / f"stage1_fold{f}.txt")
        else:
            m = train_lgbm(X[tr], y[tr].astype(np.int32),
                           n_estimators=args.fold_trees, params=FOLD_PARAMS)
        oof[~tr] = predict_scores(m, X[~tr])
        save_model(m, mdir / f"stage1_fold{f}.txt")
        log.info("fold %d/%d trained+scored in %.0fs", f + 1, args.folds,
                 time.time() - t0)
    del fold
    write_json(mdir / "stage1_features.json", cols)

    # ---- holdout truth ---------------------------------------------------
    s1_ids = pq.read_table(cand_dir / "s1_ids.parquet").column(0).to_pylist()
    pool_ids = pq.read_table(cand_dir / "pool_ids.parquet").column(0).to_pylist()
    pool_pos = {eid: j for j, eid in enumerate(pool_ids)}
    gt = load_ground_truth(cfg, "train")
    truth = {i: {pool_pos[t] for t in gt.matches.get(s1_ids[i], ()) if t in pool_pos}
             for i in range(0, len(s1_ids), args.holdout_mod)}
    del pool_pos, pool_ids, gt

    hold_all = (s1 % args.holdout_mod) == 0
    base, _, _ = tune_policy(s1[hold_all], pool[hold_all], oof[hold_all], truth,
                             log=log, exclusive="soft")
    base_rep = evaluate_policy(s1[hold_all], pool[hold_all], oof[hold_all], truth, base)
    print(f"\nstage-1 OOF baseline (holdout): macro_f05={base_rep.macro_f05:.4f} "
          f"FP={base_rep.total_fp} FN={base_rep.total_fn}")

    # ---- stage 2 ---------------------------------------------------------
    keep = topk_per_entity(s1, oof, args.topk)
    lost = int(y.sum()) - int(y[keep].sum())
    log.info("top-%d keeps %d rows; positives outside top-k: %d (%.3f%%)",
             args.topk, keep.size, lost, 100 * lost / max(1, int(y.sum())))
    s1, pool, y, X, oof = s1[keep], pool[keep], y[keep], X[keep], oof[keep]
    side = _pool_side(cfg, "train", log)
    t0 = time.time()
    group = compute_group_features(s1, pool, oof, side, n_jobs=args.n_jobs)
    del side
    log.info("group features in %.0fs", time.time() - t0)
    X2 = _stage2_matrix(X, group)
    del X, group
    hold = (s1 % args.holdout_mod) == 0
    t0 = time.time()
    model = train_lgbm(X2[~hold], y[~hold].astype(np.int32),
                       n_estimators=args.n_estimators,
                       params={"num_leaves": args.leaves} if args.leaves else None)
    prob = predict_scores(model, X2[hold])
    log.info("stage 2 trained in %.0fs", time.time() - t0)
    s_h, p_h = s1[hold], pool[hold]

    best_pol, best_rep, rows = None, None, []
    for mode in EXCLUSIVE_MODES:
        pol, rep, _ = tune_policy(s_h, p_h, prob, truth, log=log, exclusive=mode)
        rows.append({"kind": "threshold", "exclusive": mode,
                     "threshold": pol.threshold, "macro_f05": rep.macro_f05})
        print(f"stage2 threshold excl={mode:<4} t={pol.threshold:.3f} "
              f"macro_f05={rep.macro_f05:.4f} FP={rep.total_fp} FN={rep.total_fn}")
        if best_rep is None or rep.macro_f05 > best_rep.macro_f05:
            best_pol, best_rep = pol, rep
        for tau in [float(x) for x in args.taus.split(",")]:
            bp = BayesPolicy(temperature=tau, exclusive=mode)
            rep = evaluate_policy(s_h, p_h, prob, truth, bp, n_jobs=args.n_jobs)
            rows.append({"kind": "bayes", "exclusive": mode, "temperature": tau,
                         "macro_f05": rep.macro_f05})
            print(f"stage2 bayes excl={mode:<4} tau={tau:<4} "
                  f"macro_f05={rep.macro_f05:.4f} FP={rep.total_fp} FN={rep.total_fn}")
            if rep.macro_f05 > best_rep.macro_f05:
                best_pol, best_rep = bp, rep

    names = cols + list(GROUP_COLUMNS)
    save_model(model, mdir / "model.txt")
    save_feature_names(mdir, names)
    write_json(mdir / "policy.json", best_pol.to_dict())
    imp = sorted(zip(names, model.feature_importance(importance_type="gain").tolist()),
                 key=lambda kv: -kv[1])
    write_json(mdir / "metrics.json", {
        "stage1_oof_macro_f05": base_rep.macro_f05,
        "stage2_macro_f05": best_rep.macro_f05, "policy": best_pol.to_dict(),
        "topk": args.topk, "positives_outside_topk": lost, "rows": rows,
        "feature_importance_gain": imp[:20]})
    print(f"\nstage-1 OOF: {base_rep.macro_f05:.4f}  ->  stage 2: "
          f"{best_rep.macro_f05:.4f}  policy={best_pol.to_dict()}")
    print("top stage-2 features:", ", ".join(k for k, _ in imp[:10]))
    print(f"saved to {mdir}")
    return 0


def predict(args, cfg, log) -> int:
    mdir = local_artifact_path(cfg, args.model_dir + "/model.txt").parent
    import json
    cols = json.loads((mdir / "stage1_features.json").read_text())
    folds = [load_model(p) for p in sorted(mdir.glob("stage1_fold*.txt"))]
    model = load_model(mdir / "model.txt")
    names = load_feature_names(mdir / "model.txt", None)
    if names != cols + list(GROUP_COLUMNS):
        log.error("stage-2 feature list mismatch")
        return 2
    policy = load_policy(mdir / "policy.json")
    if args.exclusive is not None:
        policy.exclusive = args.exclusive
    log.info("stage 2 predict: %d fold models, policy %s", len(folds),
             policy.to_dict())

    cand_base = candidates_dir(cfg, "test")
    s1_ids = pq.read_table(cand_base / "s1_ids.parquet").column(0).to_pylist()
    pool_ids = pq.read_table(cand_base / "pool_ids.parquet").column(0).to_pylist()
    fdir = (Path(args.features_dir) if args.features_dir
            else local_artifact_path(cfg, "features/test"))
    parts = sorted(fdir.glob("part_*.parquet"))
    if not parts:
        log.error("no test features at %s", fdir)
        return 2
    S, P, PR, XS = [], [], [], []
    t0 = time.time()
    for i, part in enumerate(parts):          # s1 groups never span parts
        s, p, _, X = _load_parts([part], cols, False)
        pr = np.mean([predict_scores(m, X) for m in folds], axis=0).astype(np.float32)
        k = topk_per_entity(s, pr, args.topk)
        S.append(s[k]); P.append(p[k]); PR.append(pr[k]); XS.append(X[k])
        if (i + 1) % 10 == 0 or i == len(parts) - 1:
            log.info("stage 1 scored %d/%d parts (%.0fs)", i + 1, len(parts),
                     time.time() - t0)
    s1, pool, pr, X = (np.concatenate(S), np.concatenate(P), np.concatenate(PR),
                       np.vstack(XS))
    del S, P, PR, XS
    order = topk_per_entity(s1, pr, args.topk)      # global group/sort order
    s1, pool, pr, X = s1[order], pool[order], pr[order], X[order]
    side = _pool_side(cfg, "test", log)
    t0 = time.time()
    group = compute_group_features(s1, pool, pr, side, n_jobs=args.n_jobs)
    del side
    log.info("group features in %.0fs (%d rows)", time.time() - t0, s1.size)
    prob = predict_scores(model, _stage2_matrix(X, group))
    del X, group
    preds_idx = decide(s1, pool, prob, policy, n_jobs=args.n_jobs)

    preds = {s1_ids[a]: {pool_ids[b] for b in bs} for a, bs in preds_idx.items()}
    cands: dict = {}
    for a, b in zip(s1.tolist(), pool.tolist()):
        cands.setdefault(s1_ids[a], set()).add(pool_ids[b])
    for sid, ids in preds.items():
        cands.setdefault(sid, set()).update(ids)
    out = Path(cfg.paths.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = write_matching_results(preds, out / "matching_results.tsv",
                                   required_ids=s1_ids)
    write_candidate_pairs(cands, out / "candidate_pairs.tsv", required_ids=s1_ids,
                          max_ids_per_row=max(args.topk, 16))
    n = len(s1_ids)
    print(f"\n=== STAGE 2 PREDICT ===\n  entities: {n} | with match: {len(preds)} "
          f"| emitted: {stats['total_ids']} ({stats['total_ids'] / max(1, n):.2f}/entity)")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg)
    return train(args, cfg, log) if args.mode == "train" else predict(args, cfg, log)


if __name__ == "__main__":
    raise SystemExit(main())
