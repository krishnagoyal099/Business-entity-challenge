import numpy as np
import pytest

from src.aws_utils import stage_complete
from src.blocking import (build_exact_index, code_fingerprint,
                          ensure_normalized, iter_column, load_ids)
from src.candidate_generation import (_merge_channel_results, _pack_chunks,
                                      _topk_rows, candidates_dir, run_retrieval)
from src.candidate_metrics import retrieval_report
from src.ground_truth import load_ground_truth

TINY_CHANNELS = {
    "exact": {"enabled": True, "k": 10, "max_postings": 10},
    "char_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "word_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "rare": {"enabled": True, "k": 3, "max_df_abs": 100},
    "char_addr": {"enabled": True, "k": 3, "max_df": 1.0},
}


def _tiny_cfg(cfg):
    cfg.retrieval.channels = {k: dict(v) for k, v in TINY_CHANNELS.items()}
    cfg.retrieval.normalize_part_rows = 3
    cfg.retrieval.s1_chunk_rows = 2
    return cfg


def test_code_fingerprint(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    b = tmp_path / "b.py"
    b.write_text("x = 2\n")
    assert code_fingerprint([a]) == code_fingerprint([a])
    assert code_fingerprint([a]) != code_fingerprint([b])


def test_pack_chunks_bounds():
    assert _pack_chunks(np.array([10, 60, 30, 5.0]), 100, 2) == [(0, 2), (2, 4)]
    assert _pack_chunks(np.array([1000.0]), 100, 2) == [(0, 1)]  # oversized row runs


def test_topk_rows():
    import scipy.sparse as sp
    S = sp.csr_matrix(np.array([[0.1, 0.9, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
                               dtype=np.float32))
    q, p, r, s = _topk_rows(S, 2, 10)
    assert list(q) == [10, 10] and list(p) == [1, 2] and list(r) == [1, 2]
    assert s[0] == pytest.approx(0.9)


def test_merge_channel_results():
    per = {
        "exact": (np.array([0, 0], np.int32), np.array([5, 7], np.int32),
                  np.array([1, 2], np.uint8), np.array([1.0, 1.0], np.float32)),
        "char_name": (np.array([0, 1], np.int32), np.array([5, 9], np.int32),
                      np.array([1, 2], np.uint8), np.array([0.9, 0.5], np.float32)),
    }
    merged = _merge_channel_results(10, per)
    assert merged["n"] == 3
    assert merged["s1_idx"][0] == 0 and merged["pool_idx"][0] == 5
    assert merged["channel_bits"][0] == 0b00000011
    assert merged["ranks"]["exact"][0] == 1 and merged["ranks"]["char_name"][0] == 1
    assert merged["scores"]["char_name"][0] == pytest.approx(0.9)
    assert merged["ranks"]["word_name"][0] == 0


def test_ensure_normalized_roundtrip(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    norm = ensure_normalized(cfg, "train")
    assert set(norm) == {"source1", "source2", "source3"}
    assert list(iter_column(norm["source1"], "name_core_sorted"))[0] == "acme"
    assert list(load_ids(norm["source2"])) == ["S2-201", "S2-202", "S2-203", "S2-204"]


def test_exact_index_cap(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    norm = ensure_normalized(cfg, "train")
    index, truncated = build_exact_index(norm["source2"], cap=1)
    assert index["acme"] == [0] and truncated == 1


def test_run_retrieval_and_report(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    summary = run_retrieval(cfg, "train")
    assert summary["n_pool"] == 8 and summary["total_pairs"] > 0
    base = candidates_dir(cfg, "train")
    assert (base / "s1_ids.parquet").exists() and (base / "pool_ids.parquet").exists()
    assert list(sorted(base.glob("part_*.parquet")))
    assert (base / "retrieval_report.json").exists()

    rep = retrieval_report(cfg, "train", load_ground_truth(cfg, "train"))
    assert rep["unmapped_gt_pairs"] == 1          # S2-999 invalid label
    assert rep["n_true_pairs"] == 5
    assert rep["pair_recall"] >= 0.8
    assert rep["per_channel"]["exact"]["pair_recall"] == pytest.approx(0.8)
    assert rep["avg_candidates_per_entity"] > 0
    assert rep["recall_at_k"]["5"] >= 0.8


def test_dry_run_isolated(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    run_retrieval(cfg, "train", limit_rows=2, dry_run=True)
    assert candidates_dir(cfg, "train", True).exists()
    assert stage_complete(cfg, "candidates_train") is None
