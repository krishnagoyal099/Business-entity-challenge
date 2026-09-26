"""Formal ground-truth parser and label validation (Phase 3).

Confirmed from samples: columns source1_entity_id / matched_entity_ids; one row
per S1 entity; comma-separated matched ids prefixed S2-/S3-.

Zero-match encoding is not yet observed. Explicit markers and absent rows are
both handled; the full audit's GT-row-count vs |S1| delta decides the policy.
Pure parsing/validation: no model, no S3, no features.

Note: labels whose id is not in the pool are KEPT here (so they are visible) and
reported by validate_labels; exclude them before scoring if desired.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .data_loader import TsvReader, ensure_source

log = logging.getLogger(__name__)

DEFAULT_SEPARATOR = ","
SOURCE_PREFIXES = {"source2": "S2-", "source3": "S3-"}

ZERO_TOKENS = frozenset({
    "", "na", "n/a", "null", "none", "nan", "nil", "-", "--", "missing",
    "unknown", "empty", "blank", "not available", "not applicable",
    "no match", "no matches",
})


def match_source(token: str) -> Optional[str]:
    """Return the pool a matched id belongs to, by ID prefix (S2-/S3-)."""
    for source, prefix in SOURCE_PREFIXES.items():
        if token.startswith(prefix):
            return source
    return None


def _detect_columns(header: Sequence[str]) -> Tuple[str, str]:
    s1_col: Optional[str] = None
    match_col: Optional[str] = None
    for name in header:
        low = str(name).lower()
        if s1_col is None and ("source1" in low or ("s1" in low and "id" in low)):
            s1_col = name
        elif match_col is None and "match" in low:
            match_col = name
    if s1_col is None or match_col is None:
        raise ValueError(
            f"could not detect ground-truth columns from header "
            f"{list(header)}; pass s1_column/match_column explicitly")
    return s1_col, match_col


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _read_source(source: Any) -> Tuple[List[str], List[List[str]]]:
    if isinstance(source, (str, Path)):
        reader = TsvReader(source)
        header = reader.read_header()
        return header, reader.read_all()
    if hasattr(source, "columns") and hasattr(source, "itertuples"):
        header = [str(c) for c in source.columns]
        rows = [[_cell(v) for v in row]
                for row in source[header].itertuples(index=False)]
        return header, rows
    raise TypeError(f"parse_ground_truth accepts a path or a DataFrame, "
                    f"got {type(source).__name__}")


@dataclass
class GroundTruth:
    """Parsed labels: s1_id -> set of matched pool ids (prefix-qualified)."""
    matches: Dict[str, Set[str]]
    stats: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    @property
    def entities(self) -> List[str]:
        return sorted(self.matches)

    def cardinality(self) -> Dict[str, int]:
        counts = {"0": 0, "1": 0, "2": 0, "3_plus": 0}
        for toks in self.matches.values():
            n = len(toks)
            counts["0" if n == 0 else "1" if n == 1 else "2" if n == 2
                   else "3_plus"] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {"matches": {k: sorted(v) for k, v in self.matches.items()},
                "stats": self.stats, "warnings": self.warnings}

    def label_sets(self, s1_ids: Iterable[str],
                   absent_is_zero: bool = True) -> Dict[str, Set[str]]:
        """Expand to a label dict over the S1 id universe.

        absent_is_zero=True: S1 ids absent from the GT get empty sets.
        absent_is_zero=False: absent ids are excluded from scoring.
        """
        labels: Dict[str, Set[str]] = {}
        n_absent = 0
        for sid in s1_ids:
            sid = str(sid).strip()
            if not sid:
                continue
            if sid in self.matches:
                labels[sid] = set(self.matches[sid])
            elif absent_is_zero:
                labels[sid] = set()
            else:
                n_absent += 1
        if n_absent:
            log.warning("label_sets: %d S1 ids absent from GT are excluded "
                        "(absent_is_zero=False)", n_absent)
        return labels


def parse_ground_truth(source: Any, s1_column: Optional[str] = None,
                       match_column: Optional[str] = None,
                       separator: str = DEFAULT_SEPARATOR) -> GroundTruth:
    """Parse the ground truth from a file path or a DataFrame."""
    header, rows = _read_source(source)
    if s1_column is None or match_column is None:
        auto_s1, auto_match = _detect_columns(header)
        s1_column = s1_column or auto_s1
        match_column = match_column or auto_match
    for col in (s1_column, match_column):
        if col not in header:
            raise ValueError(f"column {col!r} not in GT header {header}")
    s1_idx = header.index(s1_column)
    match_idx = header.index(match_column)

    matches: Dict[str, Set[str]] = {}
    pair_seen: Set[Tuple[str, str]] = set()
    marker_entities: Set[str] = set()
    markers: Dict[str, int] = {}
    invalid_prefix_examples: List[str] = []
    invalid_prefix_count = duplicate_pairs = duplicate_s1_rows = 0
    blank_s1_rows = data_rows = 0

    for r in rows:
        s1v = r[s1_idx].strip() if s1_idx < len(r) else ""
        if not s1v:
            blank_s1_rows += 1
            continue
        data_rows += 1
        if s1v in matches:
            duplicate_s1_rows += 1
        else:
            matches[s1v] = set()
        v = r[match_idx].strip() if match_idx < len(r) else ""
        if v.lower() in ZERO_TOKENS:
            markers[v.lower()] = markers.get(v.lower(), 0) + 1
            marker_entities.add(s1v)
            continue
        for t in (x.strip() for x in v.split(separator)):
            if not t:
                continue
            if t.lower() in ZERO_TOKENS:
                markers[t.lower()] = markers.get(t.lower(), 0) + 1
                continue
            if match_source(t) is None:
                invalid_prefix_count += 1
                if len(invalid_prefix_examples) < 10:
                    invalid_prefix_examples.append(t)
            if (s1v, t) in pair_seen:
                duplicate_pairs += 1
            else:
                pair_seen.add((s1v, t))
                matches[s1v].add(t)

    explicit_zero = sorted(s for s in marker_entities if not matches.get(s))
    gt = GroundTruth(matches=matches, stats={}, warnings=[])
    n_entities = len(matches)
    gt.stats = {
        "s1_column": s1_column, "match_column": match_column,
        "separator": separator, "rows": data_rows,
        "blank_s1_rows": blank_s1_rows, "entities": n_entities,
        "duplicate_s1_rows": duplicate_s1_rows,
        "duplicate_pairs": duplicate_pairs, "unique_pairs": len(pair_seen),
        "zero_marker_rows": sum(markers.values()),
        "zero_marker_entities": len(explicit_zero),
        "markers_seen": markers,
        "invalid_prefix_tokens": {"count": invalid_prefix_count,
                                  "examples": invalid_prefix_examples},
        "cardinality": gt.cardinality(),
        "mean_cardinality": (round(sum(len(v) for v in matches.values())
                                   / n_entities, 3) if n_entities else 0.0),
    }
    if duplicate_s1_rows:
        gt.warnings.append(f"{duplicate_s1_rows} duplicate s1 rows (aggregated)")
    if duplicate_pairs:
        gt.warnings.append(f"{duplicate_pairs} duplicate (s1, match) pairs (deduped)")
    if invalid_prefix_count:
        gt.warnings.append(
            f"{invalid_prefix_count} matched ids without S2-/S3- prefix: "
            f"{invalid_prefix_examples}")
    if explicit_zero:
        gt.warnings.append(
            f"{len(explicit_zero)} entities with explicit zero-match markers")
    return gt


def load_ground_truth(cfg, split: str = "train") -> GroundTruth:
    """Load + parse the GT for a split using config schema overrides."""
    ov = (cfg.schema.overrides or {}).get("ground_truth", {})
    path = ensure_source(cfg, split, "ground_truth")
    return parse_ground_truth(
        path,
        s1_column=ov.get("s1_id") or ov.get("s1_column"),
        match_column=ov.get("matches") or ov.get("match_column"),
        separator=ov.get("separator", DEFAULT_SEPARATOR),
    )


def validate_labels(gt: GroundTruth, s1_ids: Iterable[str],
                    s2_ids: Iterable[str], s3_ids: Iterable[str]) -> Dict[str, Any]:
    """Cross-check parsed labels against the actual source id sets."""
    s1 = {str(x).strip() for x in s1_ids if str(x).strip()}
    s2 = {str(x).strip() for x in s2_ids if str(x).strip()}
    s3 = {str(x).strip() for x in s3_ids if str(x).strip()}
    pools = {"source2": s2, "source3": s3}

    gt_entities = set(gt.matches)
    unknown_s1 = sorted(gt_entities - s1)
    absent = sorted(s1 - gt_entities)

    unknown_matches: Set[str] = set()
    ambiguous: Set[str] = set()
    prefix_mismatch: Set[str] = set()
    for toks in gt.matches.values():
        for t in toks:
            src = match_source(t)
            in_s2, in_s3 = t in s2, t in s3
            if in_s2 and in_s3:
                ambiguous.add(t)
            if src is None:
                continue
            if t not in pools[src]:
                unknown_matches.add(t)
            if (src == "source2" and not in_s2 and in_s3) or \
               (src == "source3" and not in_s3 and in_s2):
                prefix_mismatch.add(t)

    rec = "labels consistent"
    if absent:
        rec = (f"{len(absent)} S1 ids are absent from the GT. If the official "
               f"metric scores every S1 entity, treat absent as zero-match "
               f"(validation.absent_is_zero: true); verify against the "
               f"official evaluator.")
    if unknown_matches:
        rec += (f" {len(unknown_matches)} matched ids do not exist in their "
                f"pool (invalid labels).")

    def block(items) -> Dict[str, Any]:
        items = sorted(items)
        return {"count": len(items), "examples": items[:10]}

    return {
        "n_s1": len(s1), "n_s2": len(s2), "n_s3": len(s3),
        "n_gt_entities": len(gt_entities),
        "unknown_s1_ids": block(unknown_s1),
        "absent_from_gt": block(absent),
        "unknown_match_ids": block(unknown_matches),
        "ambiguous_ids": block(ambiguous),
        "prefix_mismatches": block(prefix_mismatch),
        "recommendation": rec,
    }
