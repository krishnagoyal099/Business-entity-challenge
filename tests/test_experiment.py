from pathlib import Path

import pytest

from conftest import make_config
from src import experiment as exp_mod


def test_new_experiment_increments(tmp_path):
    cfg = make_config(tmp_path)
    assert exp_mod.new_experiment(cfg, name="baseline") == "exp_001"
    assert exp_mod.new_experiment(cfg) == "exp_002"
    reg = exp_mod.load_registry(cfg)
    assert reg["next"] == 3 and reg["experiments"]["exp_001"]["name"] == "baseline"


def test_record_metrics_persists(tmp_path):
    cfg = make_config(tmp_path)
    e1 = exp_mod.new_experiment(cfg, name="oof-run")
    exp_mod.record_metrics(cfg, e1, {"macro_f05": 0.62, "total_fp": 3}, stage="oof")
    entry = exp_mod.get_experiment(cfg, e1)
    assert entry["metrics"]["macro_f05"] == 0.62 and len(entry["history"]) == 1
    assert (Path(cfg.paths.artifact_dir) / "experiments" / e1 / "metrics.json").exists()


def test_unknown_experiment_raises(tmp_path):
    with pytest.raises(KeyError):
        exp_mod.record_metrics(make_config(tmp_path), "exp_999", {"m": 0.1})


def test_run_manifest(tmp_path):
    cfg = make_config(tmp_path)
    p = exp_mod.write_run_manifest(cfg, exp_mod.new_experiment(cfg), {"stages": ["audit"]})
    assert Path(p).exists()


def test_cost_ledger(tmp_path):
    cfg = make_config(tmp_path)
    e1 = exp_mod.new_experiment(cfg)
    entry = exp_mod.append_cost_ledger(cfg, e1, "oof", 600, "ml.m5.xlarge", 4)
    assert entry["est_cost_usd"] and entry["est_cost_usd"] > 0
    ledger = exp_mod._load_json(cfg, "experiments/cost_ledger.json", None)
    assert ledger["total_est_cost_usd"] == entry["est_cost_usd"]
    assert exp_mod.estimate_cost_usd("ml.m5.xlarge", 1.0) == pytest.approx(0.23)
    assert exp_mod.estimate_cost_usd("ml.unknown", 1.0) is None
