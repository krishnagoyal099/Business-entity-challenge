"""Entity-level decision layer (Phase 9-lite).

Policy v0: per-entity threshold on pair probability + per-entity cap. Tuned by
grid search on a holdout using the REAL metric (macro F0.5 via src.evaluator),
never pair AUC. Cap 15 > max true cardinality (11).

Iteration levers if the holdout shows pain (cheapest first):
- margin rules for singletons (emit 2 when top-1 is uncertain but top-2 is
  nearly certain)
- per-entity rank/competitor-count features fed back into the model
- separate thresholds by candidate-count regime
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .evaluator import MetricsReport, macro_f05

log = logging.getLogger(__name__)


@dataclass
class DecisionPolicy:
    threshold: float = 0.5
    max_emit: int = 15

    def to_dict(self) -> Dict[str, Any]:
        return {"threshold": round(float(self.threshold), 4),
                "max_emit": int(self.max_emit)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionPolicy":
        return cls(threshold=float(d["threshold"]),
                   max_emit=int(d.get("max_emit", 15)))


def make_entity_predictions(s1_idx: np.ndarray, pool_idx: np.ndarray,
                            prob: np.ndarray,
                            policy: DecisionPolicy) -> Dict[Any, Set[Any]]:
    """Pairs above threshold, capped per entity at max_emit (top-prob first)."""
    mask = prob >= policy.threshold
    if not mask.any():
        return {}
    s, p, pr = s1_idx[mask], pool_idx[mask], prob[mask]
    order = np.lexsort((-pr, s))
    s, p = s[order], p[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    sizes = np.diff(np.r_[starts, s.size])
    position = np.arange(s.size) - np.repeat(starts, sizes)
    keep = position < policy.max_emit
    s, p = s[keep], p[keep]
    preds: Dict[Any, Set[Any]] = {}
    for a, b in zip(s.tolist(), p.tolist()):
        preds.setdefault(a, set()).add(b)
    return preds


def evaluate_policy(s1_idx: np.ndarray, pool_idx: np.ndarray, prob: np.ndarray,
                    truth: Dict[Any, Set[Any]],
                    policy: DecisionPolicy) -> MetricsReport:
    preds = make_entity_predictions(s1_idx, pool_idx, prob, policy)
    # int keys: skip the per-source (S2-/S3- prefix) breakdown during tuning
    return macro_f05(preds, truth, source_prefixes=(), include_per_entity=False)


def tune_policy(s1_idx: np.ndarray, pool_idx: np.ndarray, prob: np.ndarray,
                truth: Dict[Any, Set[Any]],
                coarse: Optional[Sequence[float]] = None,
                refine: bool = True, log: Optional[logging.Logger] = None
                ) -> Tuple[DecisionPolicy, MetricsReport, List[Dict[str, Any]]]:
    log = log or logging.getLogger(__name__)
    table: List[Dict[str, Any]] = []
    best_pol: Optional[DecisionPolicy] = None
    best_rep: Optional[MetricsReport] = None

    def _try(t: float) -> None:
        nonlocal best_pol, best_rep
        pol = DecisionPolicy(threshold=float(t))
        rep = evaluate_policy(s1_idx, pool_idx, prob, truth, pol)
        table.append({"threshold": round(float(t), 3),
                      "macro_f05": rep.macro_f05,
                      "total_fp": rep.total_fp, "total_fn": rep.total_fn,
                      "entities_with_fp": rep.entities_with_fp})
        if best_rep is None or rep.macro_f05 > best_rep.macro_f05:
            best_pol, best_rep = pol, rep

    grid = list(coarse or np.arange(0.20, 0.905, 0.05))
    for t in grid:
        _try(float(t))
    if refine and best_pol is not None:
        center = best_pol.threshold
        lo, hi = max(0.01, center - 0.04), min(0.99, center + 0.04)
        for t in np.arange(lo, hi + 0.001, 0.01):
            if any(abs(t - g) < 1e-6 for g in grid):
                continue
            _try(float(t))
    log.info("tuned policy: threshold=%.3f macro_f05=%.4f (fp=%d fn=%d)",
             best_pol.threshold, best_rep.macro_f05, best_rep.total_fp,
             best_rep.total_fn)
    return best_pol, best_rep, table
