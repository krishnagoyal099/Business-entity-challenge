"""Experiment registry, metric recording, run manifests and the cost ledger.

Monotonic ids (exp_001, ...). Stored under artifacts/experiments/ and published
to S3 when enabled.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .aws_utils import (git_sha, local_artifact_path, publish_artifact,
                        read_json, s3_artifact_uri, s3_download, s3_exists,
                        write_json)

log = logging.getLogger(__name__)

# Approximate on-demand USD/hour. VERIFY against current regional pricing.
APPROX_INSTANCE_PRICES_USD_PER_HOUR = {
    "ml.t3.medium": 0.047, "ml.t3.large": 0.093,
    "ml.m5.xlarge": 0.230, "ml.m5.2xlarge": 0.461, "ml.m5.4xlarge": 0.922,
    "ml.r5.xlarge": 0.263, "ml.r5.2xlarge": 0.526, "ml.r5.4xlarge": 1.052,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(cfg, rel: str, default: Any) -> Any:
    local = local_artifact_path(cfg, rel)
    if not local.exists():
        uri = s3_artifact_uri(cfg, rel)
        region = getattr(cfg.aws, "region", None)
        if uri and s3_exists(uri, region=region):
            s3_download(uri, local, region=region)
    if not local.exists():
        return default
    try:
        return read_json(local)
    except Exception as exc:
        log.warning("_load_json: corrupt %s (%s); using default", rel, exc)
        return default


def _save_json(cfg, rel: str, obj: Any) -> str:
    p = local_artifact_path(cfg, rel)
    write_json(p, obj)
    publish_artifact(cfg, rel)
    return str(p)


_REGISTRY = "experiments/registry.json"
_LEDGER = "experiments/cost_ledger.json"


def load_registry(cfg) -> Dict[str, Any]:
    return _load_json(cfg, _REGISTRY, {"next": 1, "experiments": {}})


def save_registry(cfg, registry: Dict[str, Any]) -> str:
    return _save_json(cfg, _REGISTRY, registry)


def new_experiment(cfg, name: Optional[str] = None,
                   notes: Optional[str] = None) -> str:
    """Register a new experiment; returns its id (exp_001, ...)."""
    registry = load_registry(cfg)
    n = int(registry.get("next", 1))
    exp_id = f"exp_{n:03d}"
    registry["next"] = n + 1
    entry = {"exp_id": exp_id, "name": name, "notes": notes,
             "created_at": _utcnow(), "git_sha": git_sha(),
             "config": cfg.to_dict(), "metrics": {}, "history": []}
    registry.setdefault("experiments", {})[exp_id] = entry
    save_registry(cfg, registry)
    _save_json(cfg, f"experiments/{exp_id}/experiment.json", entry)
    log.info("new experiment %s (name=%s)", exp_id, name)
    return exp_id


def get_experiment(cfg, exp_id: str) -> Dict[str, Any]:
    entry = load_registry(cfg).get("experiments", {}).get(exp_id)
    if entry is None:
        raise KeyError(f"unknown experiment id: {exp_id}")
    return entry


def record_metrics(cfg, exp_id: str, metrics: Dict[str, Any],
                   stage: Optional[str] = None) -> Dict[str, Any]:
    """Attach metrics to an experiment (appended to history, latest kept)."""
    registry = load_registry(cfg)
    entry = registry.get("experiments", {}).get(exp_id)
    if entry is None:
        raise KeyError(f"unknown experiment id: {exp_id}")
    stamped = {"recorded_at": _utcnow(), "stage": stage or "latest",
               "metrics": dict(metrics)}
    entry["metrics"] = dict(metrics)
    entry.setdefault("history", []).append(stamped)
    save_registry(cfg, registry)
    _save_json(cfg, f"experiments/{exp_id}/metrics.json",
               {"latest": entry["metrics"], "history": entry["history"]})
    log.info("recorded metrics for %s (stage=%s)", exp_id, stage or "latest")
    return stamped


def write_run_manifest(cfg, exp_id: str, manifest: Dict[str, Any]) -> Path:
    return Path(_save_json(cfg, f"experiments/{exp_id}/run.json", manifest))


def estimate_cost_usd(instance_type: Optional[str], hours: float) -> Optional[float]:
    price = APPROX_INSTANCE_PRICES_USD_PER_HOUR.get(instance_type or "")
    return None if price is None else round(price * max(hours, 0.0), 4)


def append_cost_ledger(cfg, exp_id: str, stage: str, duration_s: Optional[float],
                       instance_type: Optional[str] = None,
                       n_jobs: int = 1) -> Dict[str, Any]:
    """Append a run to the cumulative cost ledger (approximate)."""
    est = estimate_cost_usd(instance_type, (duration_s or 0.0) / 3600.0)
    ledger = _load_json(cfg, _LEDGER, {"entries": [], "total_est_cost_usd": 0.0})
    entry = {"ts": _utcnow(), "exp_id": exp_id, "stage": stage,
             "duration_s": duration_s, "instance_type": instance_type,
             "n_jobs": n_jobs, "est_cost_usd": est}
    ledger["entries"].append(entry)
    ledger["total_est_cost_usd"] = round(
        (ledger.get("total_est_cost_usd") or 0.0) + (est or 0.0), 4)
    _save_json(cfg, _LEDGER, ledger)
    return entry
