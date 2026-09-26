import pytest

from conftest import make_config
from src.validation import make_entity_folds, split_labels

LABELS = {
    "S1-101": {"S2-201", "S3-301"},
    "S1-102": set(),
    "S1-103": {"S2-203", "S3-303", "S2-204"},
    "S1-104": {"S2-202"},
    "S1-105": set(),
    "S1-106": {"S3-304"},
}


def test_partition_and_disjoint():
    ids = sorted(LABELS)
    folds = make_entity_folds(ids, LABELS, n_folds=2, seed=42)
    assert sorted(x for f in folds for x in f.val_ids) == ids
    assert set(folds[0].val_ids).isdisjoint(folds[1].val_ids)
    for f in folds:
        assert sorted(f.train_ids + f.val_ids) == ids


def test_deterministic():
    ids = sorted(LABELS)
    f1 = make_entity_folds(ids, LABELS, n_folds=2, seed=42)
    f2 = make_entity_folds(ids, LABELS, n_folds=2, seed=42)
    assert [x.val_ids for x in f1] == [x.val_ids for x in f2]


def test_stratified_balance():
    folds = make_entity_folds(sorted(LABELS), LABELS, n_folds=2, seed=42)
    for f in folds:
        assert f.val_strata.get("0", 0) == 1 and f.val_strata.get("1", 0) == 1
    assert sum(f.val_strata.get("2", 0) + f.val_strata.get("3_plus", 0)
               for f in folds) == 2


def test_split_labels():
    folds = make_entity_folds(sorted(LABELS), LABELS, n_folds=2, seed=42)
    train, val = split_labels(LABELS, folds[0])
    assert set(val) == set(folds[0].val_ids)
    assert set(train) == set(folds[0].train_ids)


def test_cfg_wiring(tmp_path):
    cfg = make_config(tmp_path)  # validation.n_folds = 2
    assert len(make_entity_folds(sorted(LABELS), LABELS, cfg=cfg)) == 2


def test_too_many_folds_raises():
    with pytest.raises(ValueError):
        make_entity_folds(["a", "b"], {"a": set(), "b": set()}, n_folds=3)
