import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from src.aws_utils import local_artifact_path, write_json
from src.candidate_generation import run_retrieval
from src.entity_decision import (DecisionPolicy, evaluate_policy,
                                 make_entity_predictions, tune_policy)
from src.model import predict_scores, save_model, train_lgbm
from src.pair_features import FEATURE_COLUMNS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

TINY_CHANNELS = {
    "exact": {"enabled": True, "k": 10, "max_postings": 10},
    "char_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "word_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "rare": {"enabled": True, "k": 3, "max_df_abs": 100},
    "word_addr": {"enabled": True, "k": 3, "max_df": 1.0},
}


def test_threshold_and_cap():
    s1 = np.array([0, 0, 0], np.int32)
    pool = np.array([1, 2, 3], np.int32)
    prob = np.array([0.9, 0.7, 0.6], np.float32)
    assert make_entity_predictions(s1, pool, prob,
                                   DecisionPolicy(threshold=0.65, max_emit=2)) == {0: {1, 2}}
    assert make_entity_predictions(s1, pool, prob,
                                   DecisionPolicy(threshold=0.65, max_emit=1)) == {0: {1}}


def test_nothing_above_threshold():
    preds = make_entity_predictions(np.array([0], np.int32), np.array([1], np.int32),
                                    np.array([0.3], np.float32),
                                    DecisionPolicy(threshold=0.5))
    assert preds == {}


def test_tune_policy_finds_optimum():
    truth = {0: {1}, 1: {3, 4}}
    s1 = np.array([0, 0, 1, 1, 1], np.int32)
    pool = np.array([1, 2, 3, 4, 5], np.int32)
    prob = np.array([0.9, 0.6, 0.8, 0.7, 0.3], np.float32)
    pol, rep, table = tune_policy(s1, pool, prob, truth, refine=False)
    assert rep.macro_f05 == 1.0
    assert 0.6 < pol.threshold <= 0.75
    assert table and "macro_f05" in table[0]


def test_evaluate_policy_counts_empty_entities():
    truth = {0: {1}, 1: {2}}               # entity 1 gets no prediction
    rep = evaluate_policy(np.array([0], np.int32), np.array([1], np.int32),
                          np.array([0.9], np.float32), truth,
                          DecisionPolicy(threshold=0.5))
    assert rep.n_entities == 2 and rep.macro_f05 == 0.5


def test_zero_match_entity_rewards_empty_prediction():
    truth = {0: {1}, 1: set()}             # entity 1 has no true matches
    # threshold high: nothing emitted for entity 1 -> scores 1.0; entity 0 -> 0
    rep = evaluate_policy(np.array([0], np.int32), np.array([1], np.int32),
                          np.array([0.1], np.float32), truth,
                          DecisionPolicy(threshold=0.5))
    assert rep.macro_f05 == 0.5
    # a low threshold predicts for entity 0 only: 1.0 and 1.0
    rep2 = evaluate_policy(np.array([0], np.int32), np.array([1], np.int32),
                           np.array([0.9], np.float32), truth,
                           DecisionPolicy(threshold=0.5))
    assert rep2.macro_f05 == 1.0


def test_train_lgbm_tiny():
    rng = np.random.default_rng(0)
    X = rng.random((4000, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    m = train_lgbm(X, y, n_estimators=25,
                   params={"num_leaves": 7, "min_data_in_leaf": 5, "verbose": -1})
    p = predict_scores(m, X)
    assert p.shape == (4000,) and p.min() >= 0.0 and p.max() <= 1.0
    assert p[y == 1].mean() > p[y == 0].mean()


def test_build_full_truth_includes_zero_match_and_unmapped():
    sys.path.insert(0, str(SCRIPTS))
    import train
    gt = {"S1-A": {"S2-1", "S2-9"}, "S1-B": set()}
    s1_ids = ["S1-A", "S1-B", "S1-C", "S1-D"]
    pool_pos = {"S2-1": 10}
    truth = train._build_full_truth(gt, s1_ids, pool_pos, 2,
                                    logging.getLogger("t"))
    assert set(truth) == {0, 2}            # every 2nd entity
    assert truth[0] == {10}                # S2-9 unmapped -> dropped
    assert truth[2] == set()               # S1-C absent from GT -> zero-match


def test_full_pipeline_train_to_submission(synth, tmp_path):
    """Retrieval -> features -> tiny model -> predict_test on synthetic data
    (the train files are copied to the test split names)."""
    cfg, _ = synth
    data = Path(cfg.paths.data_dir)
    for n in (1, 2, 3):
        shutil.copy(data / "train" / f"train_source{n}.tsv",
                    data / "test" / f"test_source{n}.tsv") \
            if (data / "test").exists() else None
    (data / "test").mkdir(exist_ok=True)
    for n in (1, 2, 3):
        shutil.copy(data / "train" / f"train_source{n}.tsv",
                    data / "test" / f"test_source{n}.tsv")
    cfg.retrieval.channels = {k: dict(v) for k, v in TINY_CHANNELS.items()}
    cfg.retrieval.normalize_part_rows = 3
    cfg.retrieval.s1_chunk_rows = 2
    cfg.execution.n_jobs = 1
    conf = tmp_path / "cfg.yaml"
    conf.write_text(yaml.safe_dump({
        "paths": {"data_dir": cfg.paths.data_dir,
                  "artifact_dir": cfg.paths.artifact_dir,
                  "output_dir": cfg.paths.output_dir},
        "retrieval": {"channels": TINY_CHANNELS, "normalize_part_rows": 3,
                      "s1_chunk_rows": 2},
    }), encoding="utf-8")

    run_retrieval(cfg, "train")
    run_retrieval(cfg, "test")
    sys.path.insert(0, str(SCRIPTS))
    import build_features
    import predict_test
    for split in ("train", "test"):
        assert build_features.main(["--config", str(conf), "--split", split,
                                    "--n-jobs", "1"]) == 0

    fdir = Path(cfg.paths.artifact_dir) / "features" / "train"
    Xs, ys = [], []
    for part in sorted(fdir.glob("part_*.parquet")):
        t = pq.read_table(part)
        Xs.append(np.column_stack([t.column(c).to_numpy()
                                   for c in FEATURE_COLUMNS]).astype(np.float32))
        ys.append(t.column("label").to_numpy())
    X, y = np.vstack(Xs), np.concatenate(ys).astype(np.int32)
    assert y.sum() >= 1 and (y == 0).sum() >= 1
    model = train_lgbm(X, y, n_estimators=5,
                       params={"num_leaves": 3, "min_data_in_leaf": 1,
                               "min_data_in_bin": 1, "verbose": -1})
    mdir = local_artifact_path(cfg, "models/verifier_v1")
    save_model(model, mdir / "model.txt")
    write_json(mdir / "policy.json", {"threshold": 0.3, "max_emit": 15})

    assert predict_test.main(["--config", str(conf)]) == 0
    lines = (Path(cfg.paths.output_dir) / "matching_results.tsv"
             ).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 1 + 4             # one row per test S1 entity
    cand_lines = (Path(cfg.paths.output_dir) / "candidate_pairs.tsv"
                  ).read_text(encoding="utf-8").splitlines()
    assert cand_lines[0] == "source1_entity_id\tcandidate_entity_ids"
