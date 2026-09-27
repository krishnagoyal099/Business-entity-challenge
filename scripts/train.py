#!/usr/bin/env python
"""Phase 8-lite: train the verifier on the 300k sample + tune the decision.

Holdout truth is built from the FULL ground truth for every holdout entity,
including zero-match entities (empty sets, worth 1.0 when predicted empty) and
unretrieved true pairs (real FNs). This reproduces the leaderboard metric, so
the tuned threshold optimizes the true objective.

Also reports the retrieval ceiling: macro F0.5 of a perfect verifier that emits
exactly the retrieved true pairs. ceiling - holdout = verification gap.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import local_artifact_path, mark_stage, publish_artifact, write_json
from src.candidate_generation import candidates_dir
from src.config import load_config
from src.entity_decision import tune_policy
from src.evaluator import macro_f05
from src.ground_truth import load_ground_truth
from src.logging_utils import setup_logging, stage_timer
from src.model import predict_scores, save_model, train_lgbm
from src.pair_features import FEATURE_COLUMNS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train verifier + tune decision.")
    p.add_argument("--config", default="configs/aws_cpu.yaml")
    p.add_argument("--features-dir", default=None,
                   help="default: features/train_dryrun (300k sample)")
    p.add_argument("--candidates-dir", default=None,
                   help="default: candidates/train_dryrun (id mapping)")
    p.add_argument("--holdout-mod", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=500)
    return p.parse_args(argv)


def _build_full_truth(gt_matches, s1_ids, pool_pos, holdout_mod, log):
    """s1_idx -> set(pool_idx) from the FULL GT for every holdout entity.

    Entities with no true matches get empty sets (a correct empty prediction
    scores 1.0 under the both-empty=one convention).
    """
    truth: dict = {}
    n_zero = n_unmapped = 0
    for i in range(0, len(s1_ids), holdout_mod):
        js = set()
        for t in gt_matches.get(s1_ids[i], ()):
            j = pool_pos.get(t)
            if j is not None:
                js.add(j)
            else:
                n_unmapped += 1
        if not js:
            n_zero += 1
        truth[i] = js
    log.info("holdout truth: %d entities (%d zero-match, %d unmapped tokens)",
             len(truth), n_zero, n_unmapped)
    return truth


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg)
    fdir = (Path(args.features_dir) if args.features_dir
            else local_artifact_path(cfg, "features/train_dryrun"))
    cand_dir = (Path(args.candidates_dir) if args.candidates_dir
                else candidates_dir(cfg, "train", dry_run=True))
    parts = sorted(fdir.glob("part_*.parquet"))
    if not parts:
        log.error("no feature parts at %s", fdir)
        return 2

    log.info("loading %d feature parts...", len(parts))
    t0 = time.time()
    s1_l, pool_l, y_l, X_l = [], [], [], []
    for part in parts:
        t = pq.read_table(part)
        s1_l.append(t.column("s1_idx").to_numpy())
        pool_l.append(t.column("pool_idx").to_numpy())
        y_l.append(t.column("label").to_numpy())
        X_l.append(np.column_stack(
            [t.column(c).to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32))
    s1 = np.concatenate(s1_l)
    pool = np.concatenate(pool_l)
    y = np.concatenate(y_l)
    X = np.vstack(X_l)
    del s1_l, pool_l, y_l, X_l
    log.info("loaded %d pairs (%d positives, %.2f%%) in %.0fs",
             s1.size, int(y.sum()), 100 * y.mean(), time.time() - t0)

    # ---- full-GT holdout truth -------------------------------------------
    s1_ids = pq.read_table(cand_dir / "s1_ids.parquet").column(0).to_pylist()
    pool_ids = pq.read_table(cand_dir / "pool_ids.parquet").column(0).to_pylist()
    if int(s1.max()) + 1 > len(s1_ids):
        log.error("features cover %d s1 rows but candidates dir has %d ids "
                  "- wrong --candidates-dir?", int(s1.max()) + 1, len(s1_ids))
        return 2
    pool_pos = {eid: j for j, eid in enumerate(pool_ids)}
    gt = load_ground_truth(cfg, "train")
    truth = _build_full_truth(gt.matches, s1_ids, pool_pos, args.holdout_mod, log)
    del pool_pos, pool_ids, s1_ids, gt
    hold = (s1 % args.holdout_mod) == 0
    log.info("holdout rows: %d, positives: %d", int(hold.sum()), int(y[hold].sum()))

    model = train_lgbm(X[~hold], y[~hold].astype(np.int32),
                       n_estimators=args.n_estimators)
    prob = predict_scores(model, X[hold])

    # retrieval ceiling: a perfect verifier emits exactly the retrieved truth
    ceiling: dict = {}
    for a, b, lab in zip(s1[hold].tolist(), pool[hold].tolist(), y[hold].tolist()):
        if lab:
            ceiling.setdefault(a, set()).add(b)
    ceil_rep = macro_f05(ceiling, truth, source_prefixes=())
    log.info("retrieval ceiling (perfect verifier): macro_f05=%.4f",
             ceil_rep.macro_f05)

    timer = stage_timer("train_v1", cfg)
    timer.start()
    policy, report, table = tune_policy(s1[hold], pool[hold], prob, truth, log=log)
    timer.stop()

    mdir = local_artifact_path(cfg, "models/verifier_v1")
    save_model(model, mdir / "model.txt")
    write_json(mdir / "policy.json", policy.to_dict())
    importance = sorted(zip(FEATURE_COLUMNS,
                            model.feature_importance(importance_type="gain").tolist()),
                        key=lambda kv: -kv[1])
    write_json(mdir / "metrics.json", {
        "holdout_rows": int(hold.sum()),
        "holdout_positives": int(y[hold].sum()),
        "ceiling_macro_f05": ceil_rep.macro_f05,
        "macro_f05": report.macro_f05,
        "verification_gap": round(ceil_rep.macro_f05 - report.macro_f05, 4),
        "macro_precision": report.macro_precision,
        "macro_recall": report.macro_recall,
        "total_fp": report.total_fp, "total_fn": report.total_fn,
        "zero_true": report.zero_true, "singleton_true": report.singleton_true,
        "threshold_table": table,
        "feature_importance_gain": importance[:15],
    })
    rels = ["models/verifier_v1/model.txt", "models/verifier_v1/policy.json",
            "models/verifier_v1/metrics.json"]
    for rel in rels:
        try:
            publish_artifact(cfg, rel)
        except Exception as exc:
            log.warning("publish failed for %s (%s)", rel, exc)
    mark_stage(cfg, "train_v1",
               details={"macro_f05": report.macro_f05,
                        "ceiling_macro_f05": ceil_rep.macro_f05,
                        "threshold": policy.threshold,
                        "holdout_pairs": int(hold.sum())},
               artifacts=rels, config_hash="v1", duration_s=timer.seconds,
               peak_rss_mb=timer.peak_rss_mb)
    try:
        from src.experiment import new_experiment, record_metrics
        exp = new_experiment(cfg, name="verifier_v1")
        record_metrics(cfg, exp, {"macro_f05": report.macro_f05,
                                  "ceiling_macro_f05": ceil_rep.macro_f05,
                                  "threshold": policy.threshold,
                                  "total_fp": report.total_fp,
                                  "total_fn": report.total_fn}, stage="holdout")
    except Exception as exc:
        log.warning("experiment registry skipped (%s)", exc)

    print("\n=== TRAIN SUMMARY ===")
    print(f"  retrieval ceiling (perfect verifier): {ceil_rep.macro_f05:.4f}")
    print(f"  holdout macro F0.5 = {report.macro_f05:.4f} "
          f"(P={report.macro_precision}, R={report.macro_recall})")
    print(f"  verification gap = {ceil_rep.macro_f05 - report.macro_f05:.4f}")
    print(f"  threshold={policy.threshold} max_emit={policy.max_emit}")
    print(f"  FP={report.total_fp} FN={report.total_fn} "
          f"entities_with_fp={report.entities_with_fp}")
    print(f"  zero-true acc={report.zero_true} singleton acc={report.singleton_true}")
    print("  top features:")
    for name, gain in importance[:10]:
        print(f"    {name:<20}{gain:>12.0f}")
    print(f"  model: {mdir}/model.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
