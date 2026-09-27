import numpy as np

from src.entity_decision import (BayesPolicy, DecisionPolicy, bayes_decide,
                                 load_policy, make_entity_predictions_bayes)


def test_confident_singleton():
    assert bayes_decide([0.95, 0.01, 0.01]) == 1


def test_zero_entity_firewall():
    assert bayes_decide([0.05] * 10) == 0
    assert bayes_decide([0.02] * 40) == 0


def test_multi_match():
    assert bayes_decide([0.9, 0.85, 0.01]) == 2


def test_hedge_when_second_is_plausible():
    assert bayes_decide([0.9, 0.75]) == 2
    assert bayes_decide([0.9, 0.6]) == 1


def test_empty():
    assert bayes_decide([]) == 0


def test_end_to_end_predictions():
    s1 = np.array([0, 0, 0, 1, 1], np.int32)
    pool = np.array([1, 2, 3, 4, 5], np.int32)
    prob = np.array([0.95, 0.75, 0.01, 0.9, 0.85], np.float32)
    preds = make_entity_predictions_bayes(s1, pool, prob, BayesPolicy())
    assert preds.get(0) == {1, 2}
    assert preds.get(1) == {4, 5}


def test_temperature_keeps_firewall():
    assert bayes_decide([0.95, 0.01], temperature=1.5) == 1
    assert bayes_decide([0.05] * 10, temperature=0.6) == 0


def test_policy_roundtrip_and_legacy_format(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text('{"threshold": 0.7, "max_emit": 15}')          # v1 file, no kind
    assert isinstance(load_policy(p), DecisionPolicy)
    p.write_text('{"kind": "bayes", "temperature": 0.9, "max_emit": 15, '
                 '"window": 25}')
    pol = load_policy(p)
    assert isinstance(pol, BayesPolicy) and pol.temperature == 0.9
    assert pol.to_dict()["kind"] == "bayes"
