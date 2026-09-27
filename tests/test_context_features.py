import numpy as np
import pytest

from src.blocking import ensure_normalized
from src.pair_features import (CONTEXT_COLUMNS, FEATURE_COLUMNS,
                               LEGACY_FEATURE_COLUMNS, build_side_table,
                               compute_part_features)


def _sides(synth):
    cfg, _ = synth
    norm = ensure_normalized(cfg, "train")
    return (build_side_table(norm["source1"]),
            build_side_table(norm["source2"] + norm["source3"]))


def _retrieval(n):
    cols = ("rank_exact", "rank_char_name", "rank_word_name", "rank_rare",
            "rank_word_addr", "rank_char_addr", "score_char_name",
            "score_word_name", "score_rare", "score_word_addr",
            "score_char_addr")
    return {c: np.zeros(n, dtype=np.float32) for c in cols}


def test_context_features(synth):
    s1, pool = _sides(synth)
    s1_idx = np.array([0, 0, 1], np.int32)
    pool_idx = np.array([0, 1, 0], np.int32)
    retrieval = _retrieval(3)
    retrieval["score_word_addr"] = np.array([0.9, 0.5, 0.7], np.float32)
    retrieval["score_char_name"] = np.array([0.8, 0.4, 0.6], np.float32)
    feats = compute_part_features(s1_idx, pool_idx, np.array([1, 1, 1], np.uint8),
                                  retrieval, s1, pool)
    assert feats["ctx_rank_best"][0] == 1.0 and feats["ctx_rank_best"][1] == 2.0
    assert feats["ctx_n_cand"][0] == 2.0 and feats["ctx_n_cand"][2] == 1.0
    assert feats["ctx_top1"][0] == pytest.approx(0.9, abs=1e-5)
    assert feats["ctx_margin1"][1] == pytest.approx(0.4, abs=1e-5)
    assert feats["ctx_rel_best"][1] == pytest.approx(0.5 / 0.9, abs=1e-3)
    assert feats["ctx_rank_wa"][0] == 1.0 and feats["ctx_rank_wa"][1] == 2.0
    assert feats["ctx_rank_best"][2] == 1.0
    assert feats["ctx_margin1"][2] == pytest.approx(0.0, abs=1e-6)
    assert set(CONTEXT_COLUMNS) <= set(FEATURE_COLUMNS)


def test_context_ranks_are_per_entity_and_order_independent(synth):
    s1, pool = _sides(synth)
    # entity rows interleaved and out of order in the input
    s1_idx = np.array([1, 0, 1, 0], np.int32)
    pool_idx = np.array([0, 1, 1, 0], np.int32)
    retrieval = _retrieval(4)
    retrieval["score_word_addr"] = np.array([0.2, 0.3, 0.8, 0.9], np.float32)
    f = compute_part_features(s1_idx, pool_idx, np.ones(4, np.uint8),
                              retrieval, s1, pool)
    # entity 1 rows: idx0 (0.2), idx2 (0.8) -> ranks 2, 1
    assert f["ctx_rank_wa"][0] == 2.0 and f["ctx_rank_wa"][2] == 1.0
    # entity 0 rows: idx1 (0.3), idx3 (0.9) -> ranks 2, 1
    assert f["ctx_rank_wa"][1] == 2.0 and f["ctx_rank_wa"][3] == 1.0
    assert f["ctx_n_cand"].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_legacy_columns_are_a_subset_in_order():
    assert set(LEGACY_FEATURE_COLUMNS) <= set(FEATURE_COLUMNS)
    assert [c for c in FEATURE_COLUMNS if c in LEGACY_FEATURE_COLUMNS] == \
        list(LEGACY_FEATURE_COLUMNS)
