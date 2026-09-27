"""LightGBM pair verification model (Phase 8-lite)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.07,
    "num_leaves": 96,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 0,          # all cores
    "verbose": -1,
    "seed": 42,
}


def train_lgbm(X: np.ndarray, y: np.ndarray, n_estimators: int = 500,
               params: Optional[Dict[str, Any]] = None) -> lgb.Booster:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=int(n_estimators))


def predict_scores(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=np.float32)


def save_model(model: lgb.Booster, path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))
    return str(p)


def load_model(path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
