import pandas as pd
import pytest

from conftest import GT_HEADER, GT_ROWS, S1_IDS, S2_IDS, S3_IDS
from src.ground_truth import (GroundTruth, load_ground_truth,
                              parse_ground_truth, validate_labels)


def test_load_and_parse(synth):
    cfg, _ = synth
    gt = load_ground_truth(cfg, "train")
    assert isinstance(gt, GroundTruth)
    assert gt.matches["S1-101"] == {"S2-201", "S3-301"}
    assert gt.matches["S1-102"] == set()          # explicit zero marker
    assert gt.matches["S1-103"] == {"S2-203", "S3-303", "S2-204", "S2-999"}
    assert gt.stats["entities"] == 3
    assert gt.stats["duplicate_s1_rows"] == 2
    assert gt.stats["duplicate_pairs"] == 1
    assert gt.stats["zero_marker_entities"] == 1
    assert gt.stats["cardinality"] == {"0": 1, "1": 0, "2": 1, "3_plus": 1}
    assert any("duplicate" in w for w in gt.warnings)


def test_parse_from_dataframe():
    df = pd.DataFrame([dict(zip(GT_HEADER, r)) for r in GT_ROWS])
    gt = parse_ground_truth(df)  # columns auto-detected
    assert gt.stats["entities"] == 3
    assert gt.matches["S1-101"] == {"S2-201", "S3-301"}


def test_parse_explicit_columns_and_error():
    df = pd.DataFrame([dict(zip(GT_HEADER, r)) for r in GT_ROWS])
    gt = parse_ground_truth(df, s1_column="source1_entity_id",
                            match_column="matched_entity_ids")
    assert gt.stats["entities"] == 3
    with pytest.raises(ValueError):
        parse_ground_truth(pd.DataFrame([{"a": "x", "b": "y"}]))


def test_validate_labels(synth):
    cfg, _ = synth
    gt = load_ground_truth(cfg, "train")
    v = validate_labels(gt, S1_IDS, S2_IDS, S3_IDS)
    assert v["n_s1"] == 4 and v["n_gt_entities"] == 3
    assert v["absent_from_gt"]["count"] == 1          # S1-104
    assert v["unknown_match_ids"]["count"] == 1       # S2-999
    assert v["ambiguous_ids"]["count"] == 1           # S2-204 in both pools
    assert "absent" in v["recommendation"]


def test_label_sets_absent_policy(synth):
    cfg, _ = synth
    gt = load_ground_truth(cfg, "train")
    labels = gt.label_sets(S1_IDS, absent_is_zero=True)
    assert set(labels) == S1_IDS and labels["S1-104"] == set()
    strict = gt.label_sets(S1_IDS, absent_is_zero=False)
    assert set(strict) == {"S1-101", "S1-102", "S1-103"}
