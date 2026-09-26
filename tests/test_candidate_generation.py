import multiprocessing

import numpy as np
import pandas as pd
import pytest

from conftest import _write_tsv, make_config
from src.aws_utils import stage_complete
from src.blocking import (_name_variants, build_exact_index, code_fingerprint,
                          ensure_normalized, fetch_rows, iter_column,
                          iter_variants, load_ids)
from src.candidate_generation import (_merge_channel_results, _pack_chunks,
                                      _reduce_pairs, _topk_rows,
                                      candidates_dir, run_retrieval)
from src.candidate_metrics import load_candidate_table, retrieval_report
from src.ground_truth import load_ground_truth

TINY_CHANNELS = {
    "exact": {"enabled": True, "k": 10, "max_postings": 10},
    "char_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "word_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "rare": {"enabled": True, "k": 3, "max_df_abs": 100},
    "word_addr": {"enabled": True, "k": 3, "max_df": 1.0},
}


def _tiny_cfg(cfg):
    cfg.retrieval.channels = {k: dict(v) for k, v in TINY_CHANNELS.items()}
    cfg.retrieval.normalize_part_rows = 3
    cfg.retrieval.s1_chunk_rows = 2
    cfg.execution.n_jobs = 1
    return cfg


def test_code_fingerprint(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    b = tmp_path / "b.py"
    b.write_text("x = 2\n")
    assert code_fingerprint([a]) == code_fingerprint([a])
    assert code_fingerprint([a]) != code_fingerprint([b])


def test_name_variants():
    assert _name_variants("Rizaevo aka Custom Total LLC") == ["rizaevo", "custom total"]
    assert _name_variants("Ectolumdrex dba X+ Madison Inc") == ["ectolumdrex", "madison"]
    assert _name_variants("Lyraumbra D.B.A. S+ Aspac L.L.C.") == ["lyraumbra", "aspac"]
    assert _name_variants("Acme Corporation") == ["acme"]
    assert _name_variants("doing business as Acme Inc") == ["acme"]
    assert _name_variants("") == [""]


def test_pack_chunks_bounds():
    assert _pack_chunks(np.array([10, 60, 30, 5.0]), 100, 2) == [(0, 2), (2, 4)]
    assert _pack_chunks(np.array([1000.0]), 100, 2) == [(0, 1)]


def test_topk_rows():
    import scipy.sparse as sp
    S = sp.csr_matrix(np.array([[0.1, 0.9, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
                               dtype=np.float32))
    q, p, r, s = _topk_rows(S, 2, 10)
    assert list(q) == [10, 10] and list(p) == [1, 2] and list(r) == [1, 2]
    assert s[0] == pytest.approx(0.9)


def test_reduce_pairs_min_rank():
    q2, p2, r2, s2 = _reduce_pairs(np.array([0, 0], np.int32),
                                   np.array([5, 5], np.int32),
                                   np.array([2, 1], np.uint8),
                                   np.array([0.4, 0.9], np.float32), 100)
    assert q2.size == 1 and r2[0] == 1 and s2[0] == pytest.approx(0.9)


def test_merge_channel_results():
    per = {
        "exact": (np.array([0, 0], np.int32), np.array([5, 7], np.int32),
                  np.array([1, 1], np.uint8), np.array([1.0, 1.0], np.float32)),
        "char_name": (np.array([0, 1], np.int32), np.array([5, 9], np.int32),
                      np.array([1, 2], np.uint8), np.array([0.9, 0.5], np.float32)),
    }
    merged = _merge_channel_results(10, per)
    assert merged["n"] == 3
    assert merged["channel_bits"][0] == 0b00000011
    assert merged["ranks"]["exact"][0] == 1
    assert merged["scores"]["char_name"][0] == pytest.approx(0.9)
    assert merged["ranks"]["word_name"][0] == 0


def test_ensure_normalized_roundtrip(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    norm = ensure_normalized(cfg, "train")
    assert set(norm) == {"source1", "source2", "source3"}
    assert list(iter_column(norm["source1"], "name_core_sorted"))[0] == "acme"
    assert list(iter_variants(norm["source2"]))[0] == ["acme"]
    assert list(load_ids(norm["source2"])) == ["S2-201", "S2-202", "S2-203", "S2-204"]
    rows = fetch_rows(cfg, "train", "pool", [0, 5], ["entity_id", "name_alnum"])
    assert rows[0]["entity_id"] == "S2-201" and rows[5]["entity_id"] == "S3-303"


def test_stale_normalized_parts_are_not_reused(synth):
    """A stage-hash change must wipe old parts, not resume from them."""
    import json
    from src.blocking import normalized_dir
    cfg, _ = synth
    _tiny_cfg(cfg)
    ensure_normalized(cfg, "train")
    mp = normalized_dir(cfg, "train", "source1") / "parts.json"
    data = json.loads(mp.read_text())
    data["hash"] = "stale-hash"          # simulate a code/config change
    mp.write_text(json.dumps(data))
    from src.aws_utils import local_artifact_path
    (local_artifact_path(cfg, "manifests/normalize_train_source1") / "manifest.json").unlink()
    norm = ensure_normalized(cfg, "train")
    assert json.loads(mp.read_text())["hash"] != "stale-hash"
    assert len(list(iter_column(norm["source1"], "entity_id"))) == 4


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
    assert summary["pair_recall"] >= 0.8
    base = candidates_dir(cfg, "train")
    assert (base / "s1_ids.parquet").exists() and (base / "retrieval_report.json").exists()

    rep = retrieval_report(cfg, "train", load_ground_truth(cfg, "train"))
    assert rep["unmapped_gt_pairs"] == 1 and rep["n_true_pairs"] == 5
    assert rep["per_channel"]["exact"]["pair_recall"] == pytest.approx(0.8)
    assert rep["avg_candidates_per_entity"] > 0
    assert rep["recall_at_k"]["5"] >= 0.8
    assert "unique_pair_recall" in rep["per_channel"]["exact"]
    assert rep["missed_pairs_total"] is not None
    if rep["missed_pairs_total"]:
        assert (base / "missed_pairs_sample.tsv").exists()


def test_aka_variant_retrieval(tmp_path):
    cfg = make_config(tmp_path)
    cfg.retrieval.channels = {"word_name": {"enabled": True, "k": 3, "max_df": 1.0}}
    cfg.retrieval.normalize_part_rows = 10
    cfg.execution.n_jobs = 1
    data = tmp_path / "dataset" / "train"
    hdr = ["entity_id", "business_name", "business_address", "country"]
    _write_tsv(data / "train_source1.tsv", hdr,
               [["S1-A", "Custom Total LLC", "1 Oak Street, Springfield", "US"],
                ["S1-B", "Unrelated Business", "2 Elm Street", "US"]])
    _write_tsv(data / "train_source2.tsv", hdr,
               [["S2-A", "Rizaevo aka Custom Total LLC",
                 "1 Oak Street, Springfield", "US"],
                ["S2-B", "Other Shop", "9 Far Lane", "US"]])
    _write_tsv(data / "train_source3.tsv", hdr,
               [["S3-A", "Third Firm", "3 Pine Road", "US"]])
    _write_tsv(data / "train_ground_truth.tsv",
               ["source1_entity_id", "matched_entity_ids"], [["S1-A", "S2-A"]])
    run_retrieval(cfg, "train")
    table = load_candidate_table(cfg, "train",
                                 columns=["s1_idx", "pool_idx", "rank_word_name"])
    row = table[(table.s1_idx == 0) & (table.pool_idx == 0)]
    assert len(row) == 1 and row.rank_word_name.iloc[0] == 1
    rep = retrieval_report(cfg, "train", load_ground_truth(cfg, "train"))
    assert rep["pair_recall"] == 1.0


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="fork unavailable on this platform")
def test_parallel_content_matches_serial(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    run_retrieval(cfg, "train", n_jobs=1)
    serial = load_candidate_table(cfg, "train")
    for f in candidates_dir(cfg, "train").glob("*"):
        if f.is_file() and f.suffix in (".parquet", ".json"):
            f.unlink()
    run_retrieval(cfg, "train", n_jobs=2)
    parallel = load_candidate_table(cfg, "train")
    cols = ["s1_idx", "pool_idx", "channel_bits"] + \
        [f"rank_{c}" for c in ("exact", "char_name", "word_name", "rare", "word_addr")]
    a = serial[cols].sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    b = parallel[cols].sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_resume_is_idempotent(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    first = run_retrieval(cfg, "train")
    second = run_retrieval(cfg, "train")          # resumes: nothing pending
    assert second["total_pairs"] == first["total_pairs"]


def test_dry_run_isolated(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    run_retrieval(cfg, "train", limit_rows=2, dry_run=True)
    assert candidates_dir(cfg, "train", True).exists()
    assert stage_complete(cfg, "candidates_train") is None


def test_candidate_resume_invalidates_on_k_change(synth):
    cfg, _ = synth
    _tiny_cfg(cfg)
    run_retrieval(cfg, "train")
    t1 = load_candidate_table(cfg, "train", columns=["rank_exact"])
    assert (t1.rank_exact > 1).any()           # k=10: some pair ranked 2nd
    cfg.retrieval.channels["exact"]["k"] = 1   # content change, same chunking
    run_retrieval(cfg, "train")
    t2 = load_candidate_table(cfg, "train", columns=["rank_exact"])
    assert (t2.rank_exact <= 1).all()          # regenerated, not stale-resumed
