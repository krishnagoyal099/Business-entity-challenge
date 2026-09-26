"""S1-entity-level macro F0.5 evaluator (model-independent, stdlib-only).

Per entity (pred set P, true set T):
- precision = |P&T|/|P|, recall = |P&T|/|T|; F0.5 = 1.25*P*R / (0.25*P + R).
- pred empty & true non-empty -> 0; pred non-empty & true empty -> 0.
- both empty -> mode "one" (F=1, default) | "skip" (excluded) | "zero" (F=0).
  The official convention MUST be verified: it shifts the score by the
  zero-match entity fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

BETA = 0.5
DEFAULT_SOURCE_PREFIXES: Tuple[str, ...] = ("S2-", "S3-")


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


@dataclass
class EntityMetrics:
    s1_id: str
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f05: float
    pred_count: int
    true_count: int
    both_empty: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"s1_id": self.s1_id, "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "precision": self.precision,
                "recall": self.recall, "f05": round(self.f05, 6),
                "pred_count": self.pred_count, "true_count": self.true_count,
                "both_empty": self.both_empty}


def entity_f05(pred: Iterable[str], true: Iterable[str], both_empty: str = "one",
               beta: float = BETA, s1_id: str = "<entity>") -> EntityMetrics:
    """F0.5 for one S1 entity, handling all empty-set conventions."""
    p = {str(x) for x in (pred or ())}
    t = {str(x) for x in (true or ())}
    tp, fp, fn = len(p & t), len(p - t), len(t - p)
    if not p and not t:
        f = 1.0 if both_empty == "one" else 0.0
        return EntityMetrics(s1_id, 0, 0, 0, None, None, f, 0, 0, both_empty=True)
    precision = (tp / len(p)) if p else None
    recall = (tp / len(t)) if t else None
    f = f_beta(precision, recall, beta) if precision and recall else 0.0
    return EntityMetrics(s1_id, tp, fp, fn, precision, recall, f,
                         len(p), len(t), both_empty=False)


@dataclass
class MetricsReport:
    macro_f05: float
    macro_precision: Optional[float]
    macro_recall: Optional[float]
    beta: float = BETA
    both_empty_mode: str = "one"
    n_entities: int = 0
    n_scored: int = 0
    n_both_empty: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    entities_with_fp: int = 0
    entities_with_fn: int = 0
    zero_true: Dict[str, Optional[float]] = field(default_factory=dict)
    singleton_true: Dict[str, Optional[float]] = field(default_factory=dict)
    multi_true: Dict[str, Optional[float]] = field(default_factory=dict)
    f05_by_source: Dict[str, Optional[float]] = field(default_factory=dict)
    f05_histogram: Dict[str, int] = field(default_factory=dict)
    per_entity: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        r = lambda x: round(x, 6) if x is not None else None  # noqa: E731
        return {
            "macro_f05": round(self.macro_f05, 6),
            "macro_precision": r(self.macro_precision),
            "macro_recall": r(self.macro_recall),
            "beta": self.beta, "both_empty_mode": self.both_empty_mode,
            "n_entities": self.n_entities, "n_scored": self.n_scored,
            "n_both_empty": self.n_both_empty,
            "total_tp": self.total_tp, "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "entities_with_fp": self.entities_with_fp,
            "entities_with_fn": self.entities_with_fn,
            "zero_true": self.zero_true, "singleton_true": self.singleton_true,
            "multi_true": self.multi_true, "f05_by_source": self.f05_by_source,
            "f05_histogram": self.f05_histogram, "per_entity": self.per_entity,
        }


def _restrict(d: Dict[str, Iterable[str]], prefix: str) -> Dict[str, set]:
    return {k: {str(v) for v in (vv or ()) if str(v).startswith(prefix)}
            for k, vv in d.items()}


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 6) if values else None


def macro_f05(preds: Dict[str, Iterable[str]], trues: Dict[str, Iterable[str]],
              both_empty: str = "one", beta: float = BETA,
              source_prefixes: Tuple[str, ...] = DEFAULT_SOURCE_PREFIXES,
              include_per_entity: bool = False) -> MetricsReport:
    """Macro F0.5 over the union of entity ids in preds and trues."""
    if both_empty not in ("one", "skip", "zero"):
        raise ValueError(f"both_empty must be one|skip|zero, got {both_empty!r}")
    universe = sorted(set(preds) | set(trues))
    metrics = [entity_f05(preds.get(sid, ()), trues.get(sid, ()),
                          both_empty=both_empty, beta=beta, s1_id=sid)
               for sid in universe]
    scored = [m for m in metrics if not m.both_empty] if both_empty == "skip" else metrics
    macro = _mean([m.f05 for m in scored]) or 0.0

    # zero-match entities: true set empty (includes both-empty ones)
    zero = [m for m in metrics if m.true_count == 0]
    zero_pred_empty = sum(1 for m in zero if m.pred_count == 0)
    single = [m for m in metrics if m.true_count == 1]
    single_exact = sum(1 for m in single if m.pred_count == 1 and m.tp == 1)
    multi = [m for m in metrics if m.true_count >= 2]
    multi_exact = sum(1 for m in multi if m.fp == 0 and m.fn == 0)

    hist = {"0": 0, "(0,0.5)": 0, "[0.5,1)": 0, "1": 0}
    for m in scored:
        if m.f05 <= 0.0:
            hist["0"] += 1
        elif m.f05 >= 1.0:
            hist["1"] += 1
        elif m.f05 < 0.5:
            hist["(0,0.5)"] += 1
        else:
            hist["[0.5,1)"] += 1

    by_source: Dict[str, Optional[float]] = {}
    for prefix in source_prefixes:
        sub = macro_f05(_restrict(preds, prefix), _restrict(trues, prefix),
                        both_empty="skip", beta=beta,
                        source_prefixes=(), include_per_entity=False)
        by_source[prefix] = sub.macro_f05

    acc = lambda n, k: round(k / n, 4) if n else None  # noqa: E731
    return MetricsReport(
        macro_f05=macro,
        macro_precision=_mean([m.precision for m in scored if m.precision is not None]),
        macro_recall=_mean([m.recall for m in scored if m.recall is not None]),
        beta=beta, both_empty_mode=both_empty,
        n_entities=len(metrics), n_scored=len(scored),
        n_both_empty=sum(1 for m in metrics if m.both_empty),
        total_tp=sum(m.tp for m in metrics), total_fp=sum(m.fp for m in metrics),
        total_fn=sum(m.fn for m in metrics),
        entities_with_fp=sum(1 for m in metrics if m.fp > 0),
        entities_with_fn=sum(1 for m in metrics if m.fn > 0),
        zero_true={"n": len(zero), "pred_empty": zero_pred_empty,
                   "accuracy": acc(len(zero), zero_pred_empty)},
        singleton_true={"n": len(single), "exact": single_exact,
                        "accuracy": acc(len(single), single_exact)},
        multi_true={"n": len(multi), "exact": multi_exact,
                    "accuracy": acc(len(multi), multi_exact)},
        f05_by_source=by_source, f05_histogram=hist,
        per_entity=[m.to_dict() for m in metrics] if include_per_entity else None,
    )


def evaluate_report(preds, trues, **kwargs) -> MetricsReport:
    """Phase 1 API alias for macro_f05."""
    return macro_f05(preds, trues, **kwargs)


def summarize(report: MetricsReport) -> str:
    z, s = report.zero_true, report.singleton_true
    return "\n".join([
        f"macro F0.5 = {report.macro_f05:.4f} (both_empty={report.both_empty_mode})",
        f"entities: {report.n_entities} (scored {report.n_scored}, "
        f"both-empty {report.n_both_empty})",
        f"macro P = {report.macro_precision} | macro R = {report.macro_recall}",
        f"TP {report.total_tp} | FP {report.total_fp} | FN {report.total_fn} "
        f"| entities with FP {report.entities_with_fp}",
        f"zero-match entities: n={z.get('n')} accuracy={z.get('accuracy')}",
        f"singleton entities: n={s.get('n')} accuracy={s.get('accuracy')}",
        f"per-source F0.5: {report.f05_by_source}",
        f"histogram: {report.f05_histogram}",
    ])
