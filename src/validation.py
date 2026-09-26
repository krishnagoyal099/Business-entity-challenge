"""Leakage-safe S1-entity-level folds (Phase 3).

The split unit is ALWAYS the S1 entity. Stratification is by match-cardinality
bucket (0 / 1 / 2 / 3+) so every fold sees the same mix of zero-, singleton-
and multi-match entities.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class Fold:
    fold_id: int
    train_ids: List[str]
    val_ids: List[str]
    val_strata: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"fold_id": self.fold_id, "train_ids": self.train_ids,
                "val_ids": self.val_ids, "val_strata": self.val_strata}


def _stratum(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3_plus"


def make_entity_folds(s1_ids: Iterable[str],
                      labels: Optional[Dict[str, Set[str]]] = None,
                      cfg=None, n_folds: Optional[int] = None,
                      seed: Optional[int] = None,
                      stratify: Optional[bool] = None) -> List[Fold]:
    """Deterministic stratified K-fold over S1 entities.

    n_folds/stratify/seed come from cfg.validation / cfg.project when omitted.
    """
    ids = sorted({str(x).strip() for x in s1_ids if str(x).strip()})
    if cfg is not None:
        v = getattr(cfg, "validation", None)
        if v is not None:
            n_folds = v.n_folds if n_folds is None else n_folds
            stratify = v.stratify if stratify is None else stratify
        if seed is None:
            seed = getattr(getattr(cfg, "project", None), "seed", 42)
    n_folds = 5 if n_folds is None else int(n_folds)
    seed = 42 if seed is None else int(seed)
    stratify = True if stratify is None else bool(stratify)

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(ids) < n_folds:
        raise ValueError(f"only {len(ids)} entities for n_folds={n_folds}")

    rng = random.Random(seed)
    lab = labels or {}
    assignment: Dict[str, int] = {}
    if stratify and lab:
        strata: Dict[str, List[str]] = {"0": [], "1": [], "2": [], "3_plus": []}
        for sid in ids:
            strata[_stratum(len(lab.get(sid, ())))].append(sid)
        offset = 0  # rotate start so small strata don't all land in fold 0
        for key in ("0", "1", "2", "3_plus"):
            members = sorted(strata[key])
            rng.shuffle(members)
            for i, sid in enumerate(members):
                assignment[sid] = (i + offset) % n_folds
            offset = (offset + len(members)) % n_folds
    else:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        assignment = {sid: i % n_folds for i, sid in enumerate(shuffled)}

    folds: List[Fold] = []
    for k in range(n_folds):
        val = sorted(s for s, f in assignment.items() if f == k)
        train = sorted(s for s, f in assignment.items() if f != k)
        val_strata: Dict[str, int] = {}
        if lab:
            for sid in val:
                b = _stratum(len(lab.get(sid, ())))
                val_strata[b] = val_strata.get(b, 0) + 1
        folds.append(Fold(k, train, val, val_strata))
    return folds


def split_labels(labels: Dict[str, Set[str]],
                 fold: Fold) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Split a full label dict into (train, val) for one fold (copies)."""
    val = {s: set(labels[s]) for s in fold.val_ids if s in labels}
    train = {s: set(labels[s]) for s in fold.train_ids if s in labels}
    return train, val
