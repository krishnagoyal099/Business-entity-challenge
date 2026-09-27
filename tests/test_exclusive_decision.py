import numpy as np
import pytest

from src.entity_decision import (BayesPolicy, DecisionPolicy, decide,
                                 evaluate_policy, exclusive_adjust, load_policy)


def test_hard_keeps_only_best_s1_per_pool_record():
    pool = np.array([7, 7, 7, 8], np.int32)
    prob = np.array([0.6, 0.9, 0.3, 0.4], np.float32)
    out = exclusive_adjust(pool, prob, "hard")
    assert out.tolist() == pytest.approx([0.0, 0.9, 0.0, 0.4])
    assert out.dtype == prob.dtype


def test_soft_rescales_only_oversubscribed_records():
    pool = np.array([7, 7, 8], np.int32)
    prob = np.array([0.8, 0.8, 0.5], np.float32)
    out = exclusive_adjust(pool, prob, "soft")
    assert out.tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_none_is_identity_and_bad_mode_raises():
    prob = np.array([0.2, 0.9], np.float32)
    assert exclusive_adjust(np.array([1, 1]), prob, "none") is prob
    with pytest.raises(ValueError):
        exclusive_adjust(np.array([1, 1]), prob, "bogus")


def test_decide_removes_duplicate_claim():
    # pool record 5 is claimed by S1 0 (0.95) and S1 1 (0.9): only S1 0 keeps it
    s1 = np.array([0, 1, 1], np.int32)
    pool = np.array([5, 5, 6], np.int32)
    prob = np.array([0.95, 0.9, 0.8], np.float32)
    assert decide(s1, pool, prob, DecisionPolicy(0.5)) == {0: {5}, 1: {5, 6}}
    hard = DecisionPolicy(0.5, exclusive="hard")
    assert decide(s1, pool, prob, hard) == {0: {5}, 1: {6}}
    bayes = decide(s1, pool, prob, BayesPolicy(exclusive="hard"))
    assert bayes == {0: {5}, 1: {6}}


def test_exclusivity_improves_score_when_truth_is_exclusive():
    s1 = np.array([0, 1, 1], np.int32)
    pool = np.array([5, 5, 6], np.int32)
    prob = np.array([0.95, 0.9, 0.8], np.float32)
    truth = {0: {5}, 1: {6}}
    base = evaluate_policy(s1, pool, prob, truth, DecisionPolicy(0.5))
    hard = evaluate_policy(s1, pool, prob, truth,
                           DecisionPolicy(0.5, exclusive="hard"))
    assert hard.macro_f05 == pytest.approx(1.0)
    assert base.macro_f05 < hard.macro_f05


def test_policy_exclusive_roundtrip(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text('{"kind": "bayes", "temperature": 1.0}')      # older file
    assert load_policy(p).exclusive == "none"
    import json
    p.write_text(json.dumps(BayesPolicy(exclusive="soft").to_dict()))
    assert load_policy(p).exclusive == "soft"
    p.write_text(json.dumps(DecisionPolicy(0.4, exclusive="hard").to_dict()))
    pol = load_policy(p)
    assert isinstance(pol, DecisionPolicy) and pol.exclusive == "hard"
