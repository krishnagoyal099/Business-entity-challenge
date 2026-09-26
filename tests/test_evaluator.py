import pytest

from src.evaluator import (entity_f05, evaluate_report, f_beta, macro_f05,
                           summarize)


def test_f_beta_formula():
    assert f_beta(1.0, 1.0) == 1.0
    assert f_beta(1.0, 0.5) == pytest.approx(0.8333333)
    assert f_beta(0.5, 1.0) == pytest.approx(0.5555556)
    assert f_beta(0.0, 1.0) == 0.0


def test_case_perfect():
    m = entity_f05({"S2-A"}, {"S2-A"})
    assert (m.tp, m.fp, m.fn, m.f05) == (1, 0, 0, 1.0)


def test_case_missing_one_of_two():
    m = entity_f05({"S2-A"}, {"S2-A", "S3-B"})
    assert m.precision == 1.0 and m.recall == 0.5
    assert m.f05 == pytest.approx(0.8333333) and m.fn == 1


def test_case_extra_false_positive():
    m = entity_f05({"S2-A", "S2-B"}, {"S2-A"})
    assert m.precision == 0.5 and m.recall == 1.0
    assert m.f05 == pytest.approx(0.5555556) and m.fp == 1


def test_empty_set_conventions():
    assert entity_f05(set(), set()).f05 == 1.0
    assert entity_f05(set(), set(), both_empty="zero").f05 == 0.0
    assert entity_f05(set(), {"S2-A"}).f05 == 0.0
    assert entity_f05({"S2-A"}, set()).f05 == 0.0


def test_macro_report():
    preds = {"S1-1": {"S2-201", "S3-301"}, "S1-2": set(),
             "S1-3": {"S2-203", "S3-303"}, "S1-4": {"S2-999"}}
    trues = {"S1-1": {"S2-201", "S3-301"}, "S1-2": set(),
             "S1-3": {"S2-203"}, "S1-4": set()}
    rep = macro_f05(preds, trues)
    assert rep.macro_f05 == pytest.approx((1.0 + 1.0 + 0.5555556 + 0.0) / 4)
    assert rep.n_entities == 4 and rep.n_scored == 4
    assert rep.total_fp == 2 and rep.total_fn == 0
    assert rep.entities_with_fp == 2 and rep.entities_with_fn == 0
    assert rep.zero_true == {"n": 2, "pred_empty": 1, "accuracy": 0.5}
    assert rep.singleton_true["n"] == 1 and rep.singleton_true["accuracy"] == 0.0
    assert rep.f05_by_source["S2-"] == pytest.approx(2 / 3)
    assert rep.f05_by_source["S3-"] == pytest.approx(0.5)
    assert rep.f05_histogram["1"] == 2 and rep.f05_histogram["0"] == 1


def test_macro_skip_mode():
    preds = {"S1-1": {"S2-A"}, "S1-5": set()}
    trues = {"S1-1": {"S2-A"}, "S1-5": set()}
    one = macro_f05(preds, trues, both_empty="one")
    assert one.n_scored == 2 and one.macro_f05 == 1.0
    skip = macro_f05(preds, trues, both_empty="skip")
    assert skip.n_scored == 1 and skip.n_both_empty == 1 and skip.macro_f05 == 1.0


def test_per_entity_and_alias():
    preds = trues = {"S1-1": {"S2-A"}}
    rep = macro_f05(preds, trues, include_per_entity=True)
    assert rep.per_entity and rep.per_entity[0]["s1_id"] == "S1-1"
    assert evaluate_report(preds, trues).macro_f05 == rep.macro_f05
    assert "macro F0.5" in summarize(rep)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        macro_f05({}, {}, both_empty="bogus")
