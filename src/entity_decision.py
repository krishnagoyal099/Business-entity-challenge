"""Entity-level decision layer (Phase 9).

v2 adds the Bayes expected-F policy: per entity, emit the prefix maximizing
E[F0.5] under the model's pair probabilities (independent pairs), with exact
Poisson-binomial distributions over a top-window plus a Poisson tail. It covers
activation, singleton hedging, cardinality estimation and the zero-match
firewall in one policy; the only tuned knob is a calibration temperature on the
logits. The threshold policy (v1) is kept as baseline and fallback.
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .evaluator import MetricsReport, macro_f05

log = logging.getLogger(__name__)


@dataclass
class DecisionPolicy:
    threshold: float = 0.5
    max_emit: int = 15

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "threshold", "threshold": round(float(self.threshold), 4),
                "max_emit": int(self.max_emit)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionPolicy":
        return cls(threshold=float(d["threshold"]),
                   max_emit=int(d.get("max_emit", 15)))


@dataclass
class BayesPolicy:
    temperature: float = 1.0
    max_emit: int = 15
    window: int = 25

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "bayes", "temperature": round(float(self.temperature), 4),
                "max_emit": int(self.max_emit), "window": int(self.window)}


def load_policy(path) -> Any:
    d = json.loads(Path(path).read_text())
    if d.get("kind") == "bayes":
        return BayesPolicy(temperature=float(d["temperature"]),
                           max_emit=int(d.get("max_emit", 15)),
                           window=int(d.get("window", 25)))
    return DecisionPolicy.from_dict(d)


# ---------------------------------------------------------------------------
# threshold policy (v1)
# ---------------------------------------------------------------------------


def make_entity_predictions(s1_idx: np.ndarray, pool_idx: np.ndarray,
                            prob: np.ndarray,
                            policy: DecisionPolicy) -> Dict[Any, Set[Any]]:
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


# ---------------------------------------------------------------------------
# Bayes expected-F policy (v2)
# ---------------------------------------------------------------------------

_F_TABLE = None
_F_MAX_TOT = 64


def _f_table() -> np.ndarray:
    """F0.5 for (m, t, t+r): P = t/m, R = t/(t+r)."""
    global _F_TABLE
    if _F_TABLE is None:
        tab = np.zeros((16, 16, _F_MAX_TOT), dtype=np.float64)
        for m in range(1, 16):
            for t in range(1, m + 1):
                p = t / m
                for tot in range(t, _F_MAX_TOT):
                    r = t / tot
                    tab[m, t, tot] = 1.25 * p * r / (0.25 * p + r)
        _F_TABLE = tab
    return _F_TABLE


def _apply_temperature(ps: np.ndarray, temperature: float) -> np.ndarray:
    if abs(temperature - 1.0) < 1e-9:
        return ps
    q = np.clip(ps, 1e-6, 1.0 - 1e-6)
    lo = np.log(q / (1.0 - q))
    return 1.0 / (1.0 + np.exp(-lo / temperature))


def bayes_decide(probs, max_emit: int = 15, window: int = 25,
                 temperature: float = 1.0) -> int:
    """Best emitted prefix length m (0 = emit nothing) by expected F0.5."""
    n = len(probs)
    if n == 0:
        return 0
    max_emit = min(int(max_emit), 15)
    ps = _apply_temperature(np.asarray(probs, dtype=np.float64), temperature)
    w = ps[:window]
    f0 = float(np.prod(1.0 - ps))          # E[F] of emitting nothing
    J = len(w)
    suff = [None] * (J + 1)                # suff[m] = PB(w[m:])
    suff[J] = np.ones(1)
    for m in range(J - 1, -1, -1):
        p = float(w[m])
        prev = suff[m + 1]
        dist = np.zeros(prev.size + 1)
        dist[:prev.size] += prev * (1.0 - p)
        dist[1:prev.size + 1] += prev * p
        suff[m] = dist
    lam = float(ps[window:].sum()) if n > window else 0.0   # Poisson tail
    if lam > 1e-9:
        k = max(2, min(12, int(lam + 4.0 * math.sqrt(lam) + 2)))
        pois = np.zeros(k + 1)
        term = math.exp(-lam)
        pois[0] = term
        for r in range(1, k + 1):
            term *= lam / r
            pois[r] = term
    else:
        pois = np.ones(1)
    tab = _f_table()
    best_m, best_f = 0, f0
    pref = np.ones(1)                      # pref = PB(w[:m])
    for m in range(1, min(J, max_emit) + 1):
        p = float(w[m - 1])
        new = np.zeros(m + 1)
        new[:m] += pref * (1.0 - p)
        new[1:m + 1] += pref * p
        pref = new
        rest = suff[m]
        if pois.size > 1:
            conv = np.zeros(rest.size + pois.size - 1)
            for k2, pv in enumerate(pois):
                if pv > 1e-14:
                    conv[k2:k2 + rest.size] += pv * rest
            rest = conv
        R = rest.size
        tvec = np.arange(1, m + 1)
        tots = tvec[:, None] + np.arange(R)[None, :]
        valid = tots < _F_MAX_TOT
        toto = np.where(valid, tots, 0)
        gathered = tab[m, tvec[:, None], toto]
        ef = float(np.sum(np.outer(pref[1:], rest) * gathered * valid))
        if ef > best_f:
            best_f, best_m = ef, m
    return best_m


_BAYES_STATE: Dict[str, Any] = {}


def _bayes_range(bounds: Tuple[int, int]) -> Dict[int, Set[int]]:
    st = _BAYES_STATE
    s, p, pr = st["s"], st["p"], st["pr"]
    starts, sizes = st["starts"], st["sizes"]
    pol = st["policy"]
    lo_e, hi_e = bounds
    out: Dict[int, Set[int]] = {}
    for i in range(lo_e, hi_e):
        lo = int(starts[i])
        m = bayes_decide(pr[lo:lo + int(sizes[i])], pol.max_emit,
                         pol.window, pol.temperature)
        if m > 0:
            out[int(s[lo])] = set(p[lo:lo + m].tolist())
    return out


def make_entity_predictions_bayes(s1_idx, pool_idx, prob, policy: BayesPolicy,
                                  n_jobs: int = 1) -> Dict[Any, Set[Any]]:
    order = np.lexsort((-prob, s1_idx))
    s, p, pr = s1_idx[order], pool_idx[order], prob[order]
    del order
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    sizes = np.diff(np.r_[starts, s.size])
    n_ent = int(starts.size)
    bounds = [(i, min(i + 20000, n_ent)) for i in range(0, n_ent, 20000)]
    _BAYES_STATE.update({"s": s, "p": p, "pr": pr, "starts": starts,
                         "sizes": sizes, "policy": policy})
    try:
        if (n_jobs > 1 and n_ent > 50000
                and "fork" in multiprocessing.get_all_start_methods()):
            ctx = multiprocessing.get_context("fork")
            preds: Dict[Any, Set[Any]] = {}
            with ctx.Pool(processes=n_jobs) as pool:
                for r in pool.map(_bayes_range, bounds):
                    preds.update(r)
            return preds
        preds = {}
        for b in bounds:
            preds.update(_bayes_range(b))
        return preds
    finally:
        _BAYES_STATE.clear()


# ---------------------------------------------------------------------------
# evaluation / tuning
# ---------------------------------------------------------------------------


def evaluate_policy(s1_idx, pool_idx, prob, truth, policy,
                    n_jobs: int = 1) -> MetricsReport:
    if isinstance(policy, BayesPolicy):
        preds = make_entity_predictions_bayes(s1_idx, pool_idx, prob, policy,
                                              n_jobs=n_jobs)
    else:
        preds = make_entity_predictions(s1_idx, pool_idx, prob, policy)
    return macro_f05(preds, truth, source_prefixes=(), include_per_entity=False)


def tune_policy(s1_idx, pool_idx, prob, truth,
                coarse: Optional[Sequence[float]] = None, refine: bool = True,
                log: Optional[logging.Logger] = None
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
        for t in np.arange(max(0.01, center - 0.04), min(0.99, center + 0.041), 0.01):
            if any(abs(t - g) < 1e-6 for g in grid):
                continue
            _try(float(t))
    log.info("tuned policy: threshold=%.3f macro_f05=%.4f (fp=%d fn=%d)",
             best_pol.threshold, best_rep.macro_f05, best_rep.total_fp,
             best_rep.total_fn)
    return best_pol, best_rep, table
