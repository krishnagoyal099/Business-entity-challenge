"""Submission writers for the locked official output format.

Contract (from the official validator, utils/validate_submission.py):

matching_results.tsv (scored):
- header exactly: source1_entity_id<TAB>matched_entity_ids
- ONE row per test S1 entity (missing row = hard error); empty second column
  means zero matches
- ids comma-separated, NO whitespace, only S2-/S3- prefixed
- no duplicate S1 rows; no duplicate ids within a list
- plain UTF-8, tab-separated, not compressed

candidate_pairs.tsv:
- header exactly: source1_entity_id<TAB>candidate_entity_ids
- same row rules; final matches should be a subset of candidates

Unknown ids only lower the score (count as false positives); this pipeline only
emits ids taken from pool records.

Writers are STRICT: locally detectable violations raise ValueError. The only
auto-repair is filling missing S1 entities with empty rows, with a loud warning.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

log = logging.getLogger(__name__)

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
REQUIRED_PREFIXES = ("S2-", "S3-")
_FORBIDDEN_ID_CHARS = ("\t", "\n", "\r", ",")


def _clean_ids(ids: Iterable[str], context: str) -> List[str]:
    """Strip, validate, dedupe, sort. Raises on self-match / malformed id."""
    cleaned: List[str] = []
    seen: Set[str] = set()
    stripped = 0
    for raw in (ids or ()):
        mid = str(raw).strip()
        if not mid:
            continue
        if mid != str(raw):
            stripped += 1
        if any(ch in mid for ch in _FORBIDDEN_ID_CHARS):
            raise ValueError(f"{context}: id {mid!r} contains a forbidden "
                             f"character (tab/newline/comma).")
        if not mid.startswith(REQUIRED_PREFIXES):
            raise ValueError(f"{context}: id {mid!r} does not start with "
                             f"S2-/S3- (self-match or malformed id).")
        if mid not in seen:
            seen.add(mid)
            cleaned.append(mid)
    if stripped:
        log.warning("%s: stripped whitespace from %d id(s)", context, stripped)
    return sorted(cleaned)


def _clean_s1_key(s1: Any, context: str) -> str:
    key = str(s1).strip()
    if not key:
        raise ValueError(f"{context}: empty source1_entity_id key")
    if any(ch in key for ch in _FORBIDDEN_ID_CHARS):
        raise ValueError(f"{context}: s1 id {key!r} contains a forbidden "
                         f"character (tab/newline/comma).")
    return key


def write_id_list_file(preds: Dict[str, Iterable[str]], path,
                       header: Sequence[str], context: str,
                       required_ids: Optional[Iterable[str]] = None,
                       max_ids_per_row: Optional[int] = None) -> Dict[str, Any]:
    """Shared writer: sorted rows with sorted id lists (deterministic files)."""
    rows: Dict[str, List[str]] = {}
    for s1, ids in preds.items():
        key = _clean_s1_key(s1, context)
        if key in rows:
            raise ValueError(f"{context}: duplicate s1 key {key!r} after "
                             f"normalization")
        rows[key] = _clean_ids(ids, context)

    filled_missing: List[str] = []
    if required_ids is not None:
        required = sorted({str(x).strip() for x in required_ids if str(x).strip()})
        unexpected = sorted(set(rows) - set(required))
        if unexpected:
            raise ValueError(
                f"{context}: {len(unexpected)} s1 ids are not in the required "
                f"test S1 set, e.g. {unexpected[:5]}.")
        filled_missing = sorted(set(required) - set(rows))
        if filled_missing:
            log.warning("%s: %d required S1 entities are MISSING from "
                        "predictions - writing empty (zero-match) rows; this "
                        "usually indicates a pipeline bug; e.g. %s",
                        context, len(filled_missing), filled_missing[:5])
        ordered = required
    else:
        ordered = sorted(rows)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_nonempty = total_ids = max_row = 0
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for s1 in ordered:
            ids = rows.get(s1, [])
            if max_ids_per_row is not None and len(ids) > max_ids_per_row:
                ids = ids[:max_ids_per_row]  # safety cap; pre-cap by rank upstream
            fh.write(f"{s1}\t{','.join(ids)}\n")
            n_rows += 1
            total_ids += len(ids)
            if ids:
                n_nonempty += 1
                max_row = max(max_row, len(ids))
    stats = {"path": str(p), "rows": n_rows, "non_empty_rows": n_nonempty,
             "empty_rows": n_rows - n_nonempty, "total_ids": total_ids,
             "max_ids_per_row_observed": max_row,
             "filled_missing_rows": len(filled_missing),
             "size_bytes": p.stat().st_size}
    log.info("%s: written %s", context, stats)
    return stats


def write_matching_results(preds: Dict[str, Iterable[str]], path,
                           required_ids: Optional[Iterable[str]] = None
                           ) -> Dict[str, Any]:
    """Write matching_results.tsv in the exact official format."""
    return write_id_list_file(preds, path, MATCHING_HEADER,
                              "matching_results.tsv", required_ids)


def write_candidate_pairs(candidates: Dict[str, Iterable[str]], path,
                          required_ids: Optional[Iterable[str]] = None,
                          max_ids_per_row: Optional[int] = None
                          ) -> Dict[str, Any]:
    """Write candidate_pairs.tsv; max_ids_per_row caps file size at scale."""
    return write_id_list_file(candidates, path, CANDIDATE_HEADER,
                              "candidate_pairs.tsv", required_ids,
                              max_ids_per_row)


def read_id_list_file(path) -> Dict[str, Set[str]]:
    """Read back a matching/candidate TSV written by this module."""
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()  # header
        for line in fh:
            if not line.strip():
                continue
            s1, tab, rest = line.partition("\t")
            if not tab:
                continue
            out[s1.strip()] = {t.strip() for t in rest.strip().split(",")
                               if t.strip()}
    return out


def check_matches_subset_of_candidates(
        preds: Dict[str, Iterable[str]],
        candidates: Dict[str, Iterable[str]]) -> List[str]:
    """Mirror of the validator's soft check; returns offending s1 ids."""
    offenders: List[str] = []
    for s1, mids in preds.items():
        matched = {str(x).strip() for x in mids}
        cands = {str(x).strip() for x in (candidates.get(s1) or ())}
        if matched - cands:
            offenders.append(str(s1))
    return sorted(offenders)
