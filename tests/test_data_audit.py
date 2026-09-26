import json

import pytest

from src.data_audit import run_audit, write_profile


def test_run_audit_synthetic(synth):
    cfg, facts = synth
    rep = run_audit(cfg, split="train")
    assert set(rep["sources"]) == {"source1", "source2", "source3"}
    for key, n in facts["row_counts"].items():
        assert rep["sources"][key]["row_count"] == n
    assert rep["sources"]["source1"]["blank_rows"] == 1
    assert rep["sources"]["source2"]["ragged_rows"] == 1
    assert rep["sources"]["source3"]["short_rows"] == 1
    for key, col in facts["id_cols"].items():
        assert rep["sources"][key]["roles"]["id"]["columns"] == [col]
    roles1 = rep["sources"]["source1"]["roles"]
    assert roles1["name"]["columns"] == ["business_name"]
    assert roles1["address"]["columns"] == ["business_address"]
    assert roles1["country"]["columns"] == ["country"]
    assert rep["warnings"] == []

    gt = rep["ground_truth"]
    assert gt["status"] == "parsed"
    assert gt["match_columns"]["matched_entity_ids"]["separator"] == ","
    assert gt["entities_in_gt"] == facts["gt"]["entities"]
    assert gt["duplicate_s1_rows"] == facts["gt"]["dup_s1_rows"]
    assert gt["duplicate_pairs"] == facts["gt"]["dup_pairs"]
    assert gt["zero_marker_rows"] == facts["gt"]["zero_marker_rows"]
    assert gt["s1_ids_absent_from_gt"]["count"] == facts["gt"]["absent"]
    assert gt["invalid_tokens"]["count"] == facts["gt"]["invalid"]
    assert gt["ambiguous_tokens"]["count"] == facts["gt"]["ambiguous"]
    assert gt["cardinality"] == facts["gt"]["card"]
    mr = gt["match_rate"]
    for k in ("any", "s2", "s3", "both"):
        assert mr[k]["count"] == facts["gt"][k]
        assert mr[k]["frac_of_s1_file"] == pytest.approx(
            facts["gt"][k] / facts["gt"]["s1_total"])
    assert rep["cross_source"]["s2_s3_id_overlap"]["count"] == facts["overlap"]


def test_id_detected_when_unique_counter_capped(synth):
    cfg, _ = synth
    cfg.audit.unique_cap = 2  # smaller than the number of distinct ids
    rep = run_audit(cfg, split="train")
    for key in ("source1", "source2", "source3"):
        assert rep["sources"][key]["roles"]["id"]["columns"] == ["entity_id"]
    assert rep["ground_truth"]["status"] == "parsed"


def test_write_profile_files(synth, tmp_path):
    cfg, _ = synth
    rep = run_audit(cfg, split="train")
    out = tmp_path / "audit_out"
    written = write_profile(rep, out)
    assert "profile.json" in written and "profile.txt" in written
    loaded = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert loaded["meta"]["split"] == "train"
    txt = (out / "profile.txt").read_text(encoding="utf-8")
    assert "GROUND TRUTH" in txt and "OPEN QUESTIONS" in txt
    sample = out / "samples" / "train_source1_head.csv"
    assert len(sample.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_audit_test_split_without_gt(synth):
    cfg, _ = synth
    rep = run_audit(cfg, split="test")
    assert rep["sources"] == {}
    assert len(rep["warnings"]) == 3
    assert rep["ground_truth"]["status"] == "not_configured"
