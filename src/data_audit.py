"""Phase 2 data forensics: schema-agnostic profiling of S1/S2/S3 + ground truth.

Nothing about the dataset is assumed. ID columns, name/address/country columns,
GT encoding, zero-match convention and ID namespace collisions are detected and
reported, then locked by you via configs: schema.overrides.

The ground-truth analysis is a PROVISIONAL parse used to measure the data; the
formal parser is src/ground_truth.py (Phase 3).
"""
from __future__ import annotations

import csv
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .aws_utils import sha256_file, write_json
from .data_loader import (TsvReader, canonical_source, ensure_source,
                          extract_column, local_source_path, source_available)

MISSING_TOKENS = frozenset({
    "", "na", "n/a", "n.a.", "null", "none", "nil", "nan", "-", "--", "---",
    "?", "??", "missing", "unknown", "empty", "blank", "not available",
    "not applicable", "no address", "na.",
})

SEPARATORS: Tuple[Optional[str], ...] = (None, ",", ";", "|", " ")

ROLE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "id": ("id", "identifier", "uuid", "guid"),
    "country": ("country", "nation"),
    "address": ("address", "addr", "street", "road", "avenue", "block", "city",
                "state", "province", "zip", "zipcode", "postal", "postcode",
                "pin", "pincode", "locality", "district", "region", "building",
                "suite", "floor", "lane", "area"),
    "phone": ("phone", "telephone", "mobile", "cell", "fax"),
    "url": ("url", "website", "site", "web", "domain", "email"),
    "name": ("name", "business", "company", "entity", "firm", "brand", "store",
             "shop", "merchant", "vendor", "title"),
}

# Detection aid only; countries remain open-set.
COUNTRY_HINTS = frozenset({
    "usa", "us", "u.s.", "u.s", "united states", "united states of america",
    "america", "india", "bharat", "canada", "ca", "uk", "u.k.", "united kingdom",
    "great britain", "britain", "england", "gb", "scotland", "wales",
    "northern ireland", "australia", "austria", "belgium", "brazil", "china",
    "france", "germany", "deutschland", "greece", "hungary", "iceland",
    "ireland", "italy", "japan", "mexico", "netherlands", "the netherlands",
    "holland", "new zealand", "norway", "poland", "portugal", "russia",
    "russian federation", "spain", "sweden", "switzerland", "turkey",
    "turkiye", "south africa", "argentina", "chile", "colombia", "peru",
    "venezuela", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan",
    "myanmar", "burma", "thailand", "vietnam", "viet nam", "philippines",
    "malaysia", "singapore", "indonesia", "hong kong", "taiwan", "south korea",
    "korea", "republic of korea", "israel", "saudi arabia",
    "united arab emirates", "uae", "qatar", "kuwait", "bahrain", "oman",
    "jordan", "lebanon", "iraq", "iran", "egypt", "nigeria", "kenya", "ghana",
    "ethiopia", "morocco", "algeria", "tunisia", "libya", "sudan", "tanzania",
    "uganda", "zimbabwe", "zambia", "botswana", "namibia", "mozambique",
    "angola", "czech republic", "czechia", "slovakia", "slovenia", "croatia",
    "serbia", "bosnia and herzegovina", "bulgaria", "romania", "ukraine",
    "belarus", "lithuania", "latvia", "estonia", "finland", "denmark",
    "ind", "gbr", "aus", "deu", "fra", "esp", "ita", "nld", "bra", "mex",
    "chn", "jpn", "kor", "sgp", "mys", "idn", "tha", "vnm", "phl", "are",
    "sau", "nzl", "zaf", "pak", "bgd", "lka", "hkg", "twn", "che", "swe",
    "nor", "dnk", "irl", "pol", "prt",
})


def _keyword_hit(column_name: str, keywords: Sequence[str]) -> bool:
    norm = re.sub(r"[^a-z0-9]+", " ", column_name.lower()).strip()
    tokens = norm.split()
    for kw in keywords:
        if kw in tokens:
            return True
        if len(kw) >= 4 and kw in norm:
            return True
    return False


# ---------------------------------------------------------------------------
# column profiling
# ---------------------------------------------------------------------------

@dataclass
class ColumnStats:
    name: str
    total: int = 0
    non_empty: int = 0
    empty: int = 0
    missing_tokens: Dict[str, int] = field(default_factory=dict)
    whitespace_padded: int = 0
    unique_est: int = 0
    unique_capped: bool = False
    len_min: Optional[int] = None
    len_max: int = 0
    len_avg: Optional[float] = None
    tokens_avg: Optional[float] = None
    tokens_max: int = 0
    top_values: List[List[Any]] = field(default_factory=list)
    sampled_rows: int = 0
    digit_only_frac: float = 0.0
    alpha_only_frac: float = 0.0
    alnum_only_frac: float = 0.0
    numeric_frac: float = 0.0
    non_ascii_frac: float = 0.0
    has_digit_frac: float = 0.0
    fffd_cells: int = 0
    fffd_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _ColumnAccumulator:
    """Incremental per-column statistics (memory-bounded by unique_cap)."""

    __slots__ = ("name", "total", "non_empty", "empty", "ws_padded", "missing",
                 "len_sum", "len_min", "len_max", "tok_sum", "tok_max",
                 "fffd_cells", "fffd_chars", "sampled_n", "digit_only",
                 "alpha_only", "alnum_only", "numeric", "non_ascii", "has_digit")

    def __init__(self, name: str):
        self.name = name
        self.total = self.non_empty = self.empty = self.ws_padded = 0
        self.missing: Counter = Counter()
        self.len_sum = 0
        self.len_min: Optional[int] = None
        self.len_max = 0
        self.tok_sum = self.tok_max = 0
        self.fffd_cells = self.fffd_chars = 0
        self.sampled_n = self.digit_only = self.alpha_only = 0
        self.alnum_only = self.numeric = self.non_ascii = self.has_digit = 0

    def add(self, val: str, sampled: bool) -> None:
        self.total += 1
        s = val.strip()
        if not s:
            self.empty += 1
            self.missing[""] += 1
            return
        self.non_empty += 1
        if val != s:
            self.ws_padded += 1
        length = len(s)
        self.len_sum += length
        if self.len_min is None or length < self.len_min:
            self.len_min = length
        if length > self.len_max:
            self.len_max = length
        toks = s.split()
        self.tok_sum += len(toks)
        if len(toks) > self.tok_max:
            self.tok_max = len(toks)
        if "�" in val:
            self.fffd_cells += 1
            self.fffd_chars += val.count("�")
        low = s.lower()
        if low in MISSING_TOKENS:
            self.missing[low] += 1
        if sampled:
            self.sampled_n += 1
            if s.isdigit():
                self.digit_only += 1
            if s.isalpha():
                self.alpha_only += 1
            if s.isalnum():
                self.alnum_only += 1
            if any(ch.isdigit() for ch in s):
                self.has_digit += 1
            if any(ord(ch) > 127 for ch in s):
                self.non_ascii += 1
            if low not in MISSING_TOKENS:
                try:
                    float(s)
                    self.numeric += 1
                except ValueError:
                    pass

    def finalize(self, counter: Counter, capped: bool, top_k: int) -> ColumnStats:
        ne = self.non_empty
        sampled = self.sampled_n

        def frac(n: int) -> float:
            return round(n / sampled, 4) if sampled else 0.0

        return ColumnStats(
            name=self.name, total=self.total, non_empty=ne, empty=self.empty,
            missing_tokens=dict(self.missing),
            whitespace_padded=self.ws_padded,
            unique_est=len(counter), unique_capped=bool(capped),
            len_min=self.len_min, len_max=self.len_max,
            len_avg=round(self.len_sum / ne, 2) if ne else None,
            tokens_avg=round(self.tok_sum / ne, 2) if ne else None,
            tokens_max=self.tok_max,
            top_values=[[v, c] for v, c in counter.most_common(top_k)],
            sampled_rows=sampled,
            digit_only_frac=frac(self.digit_only),
            alpha_only_frac=frac(self.alpha_only),
            alnum_only_frac=frac(self.alnum_only),
            numeric_frac=frac(self.numeric),
            non_ascii_frac=frac(self.non_ascii),
            has_digit_frac=frac(self.has_digit),
            fffd_cells=self.fffd_cells, fffd_chars=self.fffd_chars,
        )


# ---------------------------------------------------------------------------
# source profiling
# ---------------------------------------------------------------------------

@dataclass
class SourceProfile:
    source: str
    split: str
    path: str
    row_count: int
    header: List[str]
    columns: Dict[str, ColumnStats]
    head_rows: List[Dict[str, str]] = field(default_factory=list)
    roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    leading_blank_lines: int = 0
    blank_rows: int = 0
    ragged_rows: int = 0
    short_rows: int = 0
    ragged_examples: List[Tuple[int, int]] = field(default_factory=list)
    duplicate_header_names: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source, "split": self.split, "path": self.path,
            "row_count": self.row_count, "column_count": len(self.header),
            "header": list(self.header), "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "leading_blank_lines": self.leading_blank_lines,
            "blank_rows": self.blank_rows, "ragged_rows": self.ragged_rows,
            "short_rows": self.short_rows,
            "ragged_examples": [list(x) for x in self.ragged_examples],
            "duplicate_header_names": [list(x) for x in self.duplicate_header_names],
            "roles": self.roles,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }


def profile_source(cfg, split: str, source) -> SourceProfile:
    """Full single-pass profile of one source file (memory-bounded)."""
    key = canonical_source(source)
    path = ensure_source(cfg, split, source)
    reader = TsvReader(path)
    header = reader.read_header()
    acc = {name: _ColumnAccumulator(name) for name in header}
    counters: Dict[str, Counter] = {name: Counter() for name in header}
    capped: Dict[str, bool] = {name: False for name in header}
    cap = max(1, int(cfg.audit.unique_cap))
    top_k = max(1, int(cfg.audit.top_k))
    head_n = max(0, int(cfg.audit.save_head_rows))
    sample_n = max(0, int(cfg.audit.pattern_sample_rows))
    head_rows: List[Dict[str, str]] = []
    rows = 0
    for i, row in enumerate(reader.iter_rows()):
        rows += 1
        if i < head_n:
            head_rows.append({header[j]: row[j] for j in range(len(header))})
        sampled = i < sample_n
        for j, name in enumerate(header):
            val = row[j]
            acc[name].add(val, sampled)
            counter = counters[name]
            k = val.strip()
            if len(counter) < cap or k in counter:
                counter[k] += 1
            else:
                capped[name] = True
    columns = {name: acc[name].finalize(counters[name], capped[name], top_k)
               for name in header}
    prof = SourceProfile(
        source=key, split=split, path=str(path), row_count=rows, header=header,
        columns=columns, head_rows=head_rows,
        size_bytes=path.stat().st_size, sha256=sha256_file(path),
        leading_blank_lines=reader.leading_blank_lines,
        blank_rows=reader.blank_rows, ragged_rows=reader.ragged_rows,
        short_rows=reader.short_rows,
        ragged_examples=list(reader.ragged_examples),
        duplicate_header_names=list(reader.duplicate_header_names),
    )
    detect_roles(prof, (cfg.schema.overrides or {}).get(key, {}))
    return prof


# ---------------------------------------------------------------------------
# role detection (advisory)
# ---------------------------------------------------------------------------

def detect_roles(profile: SourceProfile,
                 overrides: Optional[Dict[str, Any]] = None) -> None:
    header = profile.header
    stats = profile.columns
    roles: Dict[str, Dict[str, Any]] = {}
    assigned: Set[str] = set()

    for role, cols in (overrides or {}).items():
        cols_list = [cols] if isinstance(cols, str) else list(cols)
        for c in cols_list:
            if c not in header:
                raise ValueError(
                    f"schema.overrides[{profile.source}][{role}]: column {c!r} "
                    f"not in header {header}")
        roles[role] = {"columns": cols_list, "method": "config_override",
                       "confidence": "explicit"}
        assigned.update(cols_list)

    # id: keyword + fully populated + near-unique. When the unique counter hit
    # its cap the ratio is unmeasurable, so a capped, fully-populated column with
    # an id-like name is accepted (exact uniqueness is verified in run_audit).
    if "id" not in roles:
        cands: List[Tuple[float, str, bool]] = []
        for name in header:
            if name in assigned:
                continue
            st = stats[name]
            if not (st.total > 0 and st.non_empty == st.total
                    and _keyword_hit(name, ROLE_KEYWORDS["id"])):
                continue
            ratio = st.unique_est / st.total
            if ratio >= 0.98 or st.unique_capped:
                cands.append((ratio, name, st.unique_capped))
        if cands:
            cands.sort(reverse=True)
            capped_pick = cands[0][2]
            roles["id"] = {
                "columns": [cands[0][1]],
                "method": "name_keyword+uniqueness"
                          + ("(capped; verified in id_columns)" if capped_pick else ""),
                "confidence": "medium" if capped_pick else "high",
                "candidates": [n for _, n, _ in cands]}
            assigned.add(cands[0][1])

    if "country" not in roles:
        hits = [n for n in header if n not in assigned
                and _keyword_hit(n, ROLE_KEYWORDS["country"])]
        if hits:
            roles["country"] = {"columns": hits, "method": "name_keyword",
                                "confidence": "high" if len(hits) == 1 else "medium"}
            assigned.update(hits)
        else:
            for n in header:
                if n in assigned:
                    continue
                st = stats[n]
                if 0 < st.unique_est <= 350:
                    tops = [str(v) for v, _ in st.top_values[:25]]
                    n_hint = sum(1 for v in tops if v.strip().lower() in COUNTRY_HINTS)
                    if tops and n_hint / len(tops) >= 0.3:
                        roles["country"] = {
                            "columns": [n],
                            "method": "content_cardinality+value_hints",
                            "confidence": "low"}
                        assigned.add(n)
                        break

    for role in ("address", "phone", "url"):
        if role in roles:
            continue
        hits = [n for n in header if n not in assigned
                and _keyword_hit(n, ROLE_KEYWORDS[role])]
        if hits:
            roles[role] = {"columns": hits, "method": "name_keyword",
                           "confidence": "high" if len(hits) == 1 else "medium"}
            assigned.update(hits)

    if "name" not in roles:
        hits = [n for n in header if n not in assigned
                and _keyword_hit(n, ROLE_KEYWORDS["name"])]
        if hits:
            roles["name"] = {"columns": hits, "method": "name_keyword",
                             "confidence": "high" if len(hits) == 1 else "medium"}
            assigned.update(hits)
        else:
            best: Optional[Tuple[float, str]] = None
            for n in header:
                if n in assigned:
                    continue
                st = stats[n]
                if (st.tokens_avg is not None and 1 <= st.tokens_avg <= 8
                        and st.has_digit_frac <= 0.2 and (st.len_avg or 0) >= 3):
                    score = 1.0 - st.has_digit_frac
                    if best is None or score > best[0]:
                        best = (score, n)
            if best is not None:
                roles["name"] = {"columns": [best[1]],
                                 "method": "content_heuristic",
                                 "confidence": "low"}
                assigned.add(best[1])

    profile.roles = roles


# ---------------------------------------------------------------------------
# ground truth profiling (provisional)
# ---------------------------------------------------------------------------

def _split_tokens(v: str, sep: Optional[str]) -> List[str]:
    toks = [t.strip() for t in (v.split(sep) if sep else [v])]
    return [t for t in toks if t]


def _sep_stats(rows, ci: int, sep: Optional[str], s2_ids: Set[str],
               s3_ids: Set[str]) -> Dict[str, Any]:
    total = valid = s2c = s3c = multi = missing_vals = 0
    for r in rows:
        v = r[ci].strip()
        if v.lower() in MISSING_TOKENS:
            missing_vals += 1
            continue
        toks = _split_tokens(v, sep)
        if len(toks) > 1:
            multi += 1
        for t in toks:
            total += 1
            if t in s2_ids or t in s3_ids:
                valid += 1
            if t in s2_ids:
                s2c += 1
            if t in s3_ids:
                s3c += 1
    return {
        "token_count": total,
        "valid_token_frac": round(valid / total, 4) if total else 0.0,
        "s2_token_frac": round(s2c / total, 4) if total else 0.0,
        "s3_token_frac": round(s3c / total, 4) if total else 0.0,
        "rows_with_multiple_tokens": multi,
        "missing_value_rows": missing_vals,
    }


def profile_ground_truth(cfg, split: str, id_sets: Dict[str, Set[str]],
                         save_head_rows: int) -> Dict[str, Any]:
    files = getattr(cfg.data, split)
    if not getattr(files, "ground_truth", ""):
        return {"present": False, "status": "not_configured",
                "reason": "no ground_truth filename configured for this split "
                          "(expected for test)"}
    if not source_available(cfg, split, "ground_truth"):
        return {"present": False, "status": "not_found",
                "reason": "configured but not found locally or in S3",
                "path": str(local_source_path(cfg, split, "ground_truth"))}
    path = ensure_source(cfg, split, "ground_truth")
    reader = TsvReader(path)
    header = reader.read_header()
    rows = reader.read_all()
    out: Dict[str, Any] = {
        "present": True, "status": "unparsed", "path": str(path),
        "row_count": len(rows), "columns": list(header),
        "head_rows": [dict(zip(header, r)) for r in rows[:save_head_rows]],
        "ragged_rows": reader.ragged_rows, "short_rows": reader.short_rows,
        "blank_rows": reader.blank_rows,
    }
    s1_ids = id_sets.get("source1") or set()
    s2_ids = id_sets.get("source2") or set()
    s3_ids = id_sets.get("source3") or set()
    if not s1_ids or not (s2_ids or s3_ids):
        out["reason"] = ("source id columns not detected; set "
                         "schema.overrides.<sourceN>.id then rerun the audit")
        return out

    membership: Dict[str, Dict[str, float]] = {}
    for ci, col in enumerate(header):
        nonblank = [r[ci].strip() for r in rows if r[ci].strip()]
        if nonblank:
            def frac(pool: Set[str]) -> float:
                return round(sum(1 for v in nonblank if v in pool) / len(nonblank), 4)
            membership[col] = {"s1_frac": frac(s1_ids), "s2_frac": frac(s2_ids),
                               "s3_frac": frac(s3_ids)}
        else:
            membership[col] = {"s1_frac": 0.0, "s2_frac": 0.0, "s3_frac": 0.0}
    out["column_membership"] = membership

    s1_col: Optional[str] = None
    best_frac = 0.0
    for col in header:
        f = membership[col]["s1_frac"]
        if f > best_frac:
            best_frac, s1_col = f, col
    if s1_col is None or best_frac < 0.9:
        out["status"] = "no_s1_column"
        out["reason"] = (f"no column with >=90% S1 id membership "
                         f"(best: {s1_col}={best_frac})")
        return out
    out["s1_column"] = s1_col
    s1_idx = header.index(s1_col)
    match_cols = [c for c in header if c != s1_col]
    col_idx = {c: header.index(c) for c in match_cols}

    match_info: Dict[str, Dict[str, Any]] = {}
    for col in match_cols:
        best_sep: Optional[str] = None
        best_stats: Optional[Dict[str, Any]] = None
        for sep in SEPARATORS:
            st = _sep_stats(rows, col_idx[col], sep, s2_ids, s3_ids)
            if (best_stats is None
                    or st["valid_token_frac"] > best_stats["valid_token_frac"]):
                best_sep, best_stats = sep, st
        match_info[col] = {"separator": best_sep, **best_stats}  # type: ignore[arg-type]
    out["match_columns"] = match_info

    pair_list: List[Tuple[str, str]] = []
    per_row_s1: List[str] = []
    markers: Counter = Counter()
    zero_marker_rows = 0
    for r in rows:
        s1v = r[s1_idx].strip()
        per_row_s1.append(s1v)
        row_any_match = False
        for col in match_cols:
            v = r[col_idx[col]].strip()
            if v.lower() in MISSING_TOKENS:
                markers[v.lower()] += 1
                continue
            toks = _split_tokens(v, match_info[col]["separator"])
            if toks:
                row_any_match = True
            for t in toks:
                pair_list.append((s1v, t))
        if not row_any_match:
            zero_marker_rows += 1

    unique_pairs = set(pair_list)
    dup_pairs = len(pair_list) - len(unique_pairs)
    duplicate_s1_rows = sum(c - 1 for c in Counter(per_row_s1).values())

    # Only labels that exist in S2/S3 count as matches: unknown ids cannot be
    # predicted, so they must not inflate match-set sizes.
    matches: Dict[str, Set[str]] = defaultdict(set)
    for s, t in unique_pairs:
        if t in s2_ids or t in s3_ids:
            matches[s].add(t)
    entities = sorted(set(per_row_s1))
    entities_in_gt = len(entities)
    explicit_zero = [s for s in entities if not matches.get(s)]
    absent = sorted(s1_ids - set(entities))
    not_in_s1 = sorted(set(entities) - s1_ids)

    sizes = {s: len(matches.get(s, ())) for s in entities}
    card_counter = Counter(sizes.values())
    cardinality = {
        "0": card_counter.get(0, 0), "1": card_counter.get(1, 0),
        "2": card_counter.get(2, 0),
        "3_plus": sum(v for k, v in card_counter.items() if k >= 3),
        "max": max(sizes.values(), default=0),
        "mean": round(sum(sizes.values()) / entities_in_gt, 3) if entities_in_gt else 0.0,
    }

    all_tokens: Set[str] = {t for _, t in unique_pairs}
    invalid = sorted(t for t in all_tokens if t not in s2_ids and t not in s3_ids)
    ambiguous = sorted(t for t in all_tokens if t in s2_ids and t in s3_ids)
    s2_tokens = sum(1 for t in all_tokens if t in s2_ids and t not in s3_ids)
    s3_tokens = sum(1 for t in all_tokens if t in s3_ids and t not in s2_ids)

    def has_s2(s: str) -> bool:
        return any(t in s2_ids and t not in s3_ids for t in matches.get(s, ()))

    def has_s3(s: str) -> bool:
        return any(t in s3_ids and t not in s2_ids for t in matches.get(s, ()))

    ent_any = sum(1 for s in entities if matches.get(s))
    ent_s2 = sum(1 for s in entities if has_s2(s))
    ent_s3 = sum(1 for s in entities if has_s3(s))
    ent_both = sum(1 for s in entities if has_s2(s) and has_s3(s))

    def rate(cnt: int) -> Dict[str, Any]:
        return {
            "count": cnt,
            "frac_of_s1_file": round(cnt / len(s1_ids), 4) if s1_ids else None,
            "frac_of_gt_entities": (round(cnt / entities_in_gt, 4)
                                    if entities_in_gt else None),
        }

    out.update({
        "status": "parsed",
        "unique_pairs": len(unique_pairs),
        "duplicate_pairs": dup_pairs,
        "duplicate_s1_rows": duplicate_s1_rows,
        "zero_marker_rows": zero_marker_rows,
        "markers_seen": dict(markers),
        "entities_in_gt": entities_in_gt,
        "entities_with_explicit_zero_marker": len(explicit_zero),
        "s1_ids_absent_from_gt": {"count": len(absent), "examples": absent[:10]},
        "gt_s1_ids_not_in_source1": {"count": len(not_in_s1),
                                     "examples": not_in_s1[:10]},
        "cardinality": cardinality,
        "matched_tokens": {"unique": len(all_tokens),
                           "s2_unambiguous": s2_tokens,
                           "s3_unambiguous": s3_tokens},
        "invalid_tokens": {"count": len(invalid), "examples": invalid[:10]},
        "ambiguous_tokens": {"count": len(ambiguous), "examples": ambiguous[:10]},
        "match_rate": {
            "denominators": {"s1_file": len(s1_ids), "gt_entities": entities_in_gt},
            "any": rate(ent_any), "s2": rate(ent_s2), "s3": rate(ent_s3),
            "both": rate(ent_both),
        },
    })
    return out


def cross_source_summary(id_sets: Dict[str, Set[str]],
                         gt: Dict[str, Any]) -> Dict[str, Any]:
    s1 = id_sets.get("source1", set())
    s2 = id_sets.get("source2", set())
    s3 = id_sets.get("source3", set())
    overlap = sorted(s2 & s3)
    return {
        "s1_ids": len(s1), "s2_ids": len(s2), "s3_ids": len(s3),
        "s2_s3_id_overlap": {"count": len(overlap), "examples": overlap[:10]},
        "s2_s3_jaccard": (round(len(overlap) / len(s2 | s3), 4)
                          if (s2 or s3) else None),
        "ground_truth_parsed": gt.get("status") == "parsed",
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_audit(cfg, split: str = "train") -> Dict[str, Any]:
    """Profile all sources + ground truth for a split. Returns the full report."""
    t0 = time.time()
    warnings: List[str] = []
    profiles: Dict[str, SourceProfile] = {}
    for src in (1, 2, 3):
        key = canonical_source(src)
        if not source_available(cfg, split, src):
            warnings.append(f"{key}: file not found locally or in S3; skipped")
            continue
        profiles[key] = profile_source(cfg, split, src)

    id_sets: Dict[str, Set[str]] = {}
    id_info: Dict[str, Dict[str, Any]] = {}
    for key, prof in profiles.items():
        id_cols = prof.roles.get("id", {}).get("columns", [])
        if not id_cols:
            warnings.append(f"{key}: no id column detected; set "
                            f"schema.overrides.{key}.id and rerun")
            continue
        vals = extract_column(Path(prof.path), id_cols[0])
        nonempty = [v.strip() for v in vals if v.strip()]
        id_sets[key] = set(nonempty)
        id_info[key] = {"column": id_cols[0], "rows": len(vals),
                        "unique": len(set(nonempty)),
                        "duplicates": len(nonempty) - len(set(nonempty))}
        if id_info[key]["duplicates"]:
            warnings.append(f"{key}: id column {id_cols[0]!r} has "
                            f"{id_info[key]['duplicates']} duplicate ids")

    gt = profile_ground_truth(cfg, split, id_sets, cfg.audit.save_head_rows)
    if gt.get("status") in ("not_found", "unparsed", "no_s1_column"):
        warnings.append(f"ground_truth: {gt['status']}: {gt.get('reason', '')}")

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "tool": "src/data_audit.py",
            "elapsed_s": round(time.time() - t0, 2),
            "audit_config": asdict(cfg.audit),
        },
        "sources": {k: p.to_dict() for k, p in profiles.items()},
        "id_columns": id_info,
        "ground_truth": gt,
        "cross_source": cross_source_summary(id_sets, gt),
        "samples": {k: p.head_rows for k, p in profiles.items()},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# rendering + persistence
# ---------------------------------------------------------------------------

def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n} TB"


def _f2(x) -> str:
    return "-" if x is None else f"{x}"


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.0f}"


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    meta = report["meta"]
    lines.append("=" * 78)
    lines.append("BUSINESS ENTITY RESOLUTION — DATA AUDIT")
    lines.append(f"generated: {meta['generated_at']} | split: {meta['split']} | "
                 f"elapsed: {meta['elapsed_s']}s")
    lines.append("=" * 78)
    if report["warnings"]:
        lines.append("")
        lines.append(f"WARNINGS ({len(report['warnings'])})")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    for key, src in report["sources"].items():
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"SOURCE {key}  ({src['path']})")
        lines.append(f"  size: {_fmt_bytes(src['size_bytes'])} | "
                     f"sha256: {src['sha256'][:16]}")
        lines.append(f"  rows: {src['row_count']} | columns: "
                     f"{src['column_count']} | ragged: {src['ragged_rows']} | "
                     f"short: {src['short_rows']} | blank: {src['blank_rows']} | "
                     f"leading blank: {src['leading_blank_lines']}")
        if src["duplicate_header_names"]:
            lines.append(f"  duplicate header names renamed: "
                         f"{src['duplicate_header_names']}")
        lines.append("  roles (advisory — confirm via schema.overrides):")
        if src["roles"]:
            for role, info in sorted(src["roles"].items()):
                lines.append(f"    {role:<9}: {', '.join(info['columns']):<28} "
                             f"({info['method']}, {info['confidence']})")
        else:
            lines.append("    (none detected)")
        lines.append("  columns:")
        lines.append(f"    {'column':<22}{'nonempty':>9}{'empty':>7}"
                     f"{'unique':>9}{'avglen':>8}{'avgtok':>8}{'dig%':>6}"
                     f"{'alp%':>6}{'num%':>6}{'nasc%':>6}  top values")
        for name, st in src["columns"].items():
            uniq = (f">{st['unique_est']}" if st["unique_capped"]
                    else str(st["unique_est"]))
            tops = "; ".join(f"{(v if v else '(empty)')[:18]}({c})"
                             for v, c in st["top_values"][:3])
            lines.append(f"    {name[:22]:<22}{st['non_empty']:>9}"
                         f"{st['empty']:>7}{uniq:>9}{_f2(st['len_avg']):>8}"
                         f"{_f2(st['tokens_avg']):>8}"
                         f"{_pct(st['digit_only_frac']):>6}"
                         f"{_pct(st['alpha_only_frac']):>6}"
                         f"{_pct(st['numeric_frac']):>6}"
                         f"{_pct(st['non_ascii_frac']):>6}  {tops}")
    gt = report["ground_truth"]
    lines.append("")
    lines.append("-" * 78)
    lines.append("GROUND TRUTH")
    if not gt.get("present"):
        lines.append(f"  present: no (status: {gt.get('status')}; "
                     f"{gt.get('reason', '')})")
    else:
        lines.append(f"  present: yes | status: {gt.get('status')} | "
                     f"rows: {gt['row_count']} | columns: {gt['columns']}")
        if gt.get("status") == "parsed":
            lines.append(f"  s1 column: {gt['s1_column']}")
            for col, info in gt["match_columns"].items():
                lines.append(
                    f"  match column {col!r}: separator={info['separator']!r} "
                    f"tokens={info['token_count']} "
                    f"valid={info['valid_token_frac']} "
                    f"s2={info['s2_token_frac']} s3={info['s3_token_frac']} "
                    f"multi-token rows={info['rows_with_multiple_tokens']}")
            lines.append(f"  zero-marker rows: {gt['zero_marker_rows']} | "
                         f"markers seen: {gt['markers_seen']}")
            lines.append(f"  entities in GT: {gt['entities_in_gt']} | "
                         f"duplicate s1 rows: {gt['duplicate_s1_rows']} | "
                         f"duplicate pairs: {gt['duplicate_pairs']}")
            c = gt["cardinality"]
            lines.append(f"  cardinality (valid labels only): 0 -> {c['0']} | "
                         f"1 -> {c['1']} | 2 -> {c['2']} | 3+ -> {c['3_plus']} | "
                         f"max {c['max']} | mean {c['mean']}")
            lines.append(f"  invalid tokens: {gt['invalid_tokens']['count']} "
                         f"{gt['invalid_tokens']['examples']}")
            lines.append(f"  ambiguous tokens (id in both S2 and S3): "
                         f"{gt['ambiguous_tokens']['count']} "
                         f"{gt['ambiguous_tokens']['examples']}")
            lines.append(f"  S1 ids absent from GT (zero-by-absence "
                         f"candidates): {gt['s1_ids_absent_from_gt']['count']} "
                         f"{gt['s1_ids_absent_from_gt']['examples']}")
            mr = gt["match_rate"]
            lines.append(
                f"  match rate (denominator |S1|={mr['denominators']['s1_file']}): "
                f"any={mr['any']['count']} s2={mr['s2']['count']} "
                f"s3={mr['s3']['count']} both={mr['both']['count']}")
        else:
            lines.append(f"  not parsed: {gt.get('reason', '')}")
    cs = report["cross_source"]
    lines.append("")
    lines.append("-" * 78)
    lines.append("CROSS-SOURCE")
    lines.append(f"  |S1|={cs['s1_ids']} |S2|={cs['s2_ids']} |S3|={cs['s3_ids']} "
                 f"| S2^S3 id overlap: {cs['s2_s3_id_overlap']['count']} "
                 f"{cs['s2_s3_id_overlap']['examples']}")
    questions: List[str] = [
        "Confirm detected role columns; lock them via configs: "
        "schema.overrides (id / name / address / country per source).",
    ]
    if gt.get("status") == "parsed" and (gt["zero_marker_rows"]
                                         or gt["s1_ids_absent_from_gt"]["count"]):
        questions.append(
            f"Zero-match encoding: explicit marker rows={gt['zero_marker_rows']} "
            f"vs absent-from-GT count={gt['s1_ids_absent_from_gt']['count']}; "
            f"verify how the official evaluator treats both.")
    if cs["s2_s3_id_overlap"]["count"]:
        questions.append(
            f"S2/S3 ID namespace overlap={cs['s2_s3_id_overlap']['count']}: "
            f"decide disambiguation (e.g. prefix ids with source) before Phase 3.")
    if gt.get("status") == "parsed" and gt["invalid_tokens"]["count"]:
        questions.append(
            f"Ground-truth labels referencing unknown ids: "
            f"{gt['invalid_tokens']['count']} — excluded from match sets; "
            f"confirm this policy.")
    questions.append(
        "Verify metric edge conventions (empty prediction vs empty truth) "
        "against the official evaluator before Phase 3 decision work.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("OPEN QUESTIONS (resolve before Phase 3)")
    for i, q in enumerate(questions, start=1):
        lines.append(f"  {i}. {q}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[Dict[str, str]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def write_profile(report: Dict[str, Any], out_dir) -> List[str]:
    """Write profile.json, profile.txt and samples/ into out_dir.

    Returns the filenames written (relative to out_dir) for the stage manifest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    write_json(out / "profile.json", report)
    written.append("profile.json")
    (out / "profile.txt").write_text(render_text(report), encoding="utf-8")
    written.append("profile.txt")
    split = report["meta"]["split"]
    for key, rows in report.get("samples", {}).items():
        rel = f"samples/{split}_{key}_head.csv"
        _write_csv(out / rel, rows, report["sources"][key]["header"])
        written.append(rel)
    gt_rows = report.get("ground_truth", {}).get("head_rows") or []
    if gt_rows:
        rel = f"samples/{split}_ground_truth_head.csv"
        _write_csv(out / rel, gt_rows, report["ground_truth"]["columns"])
        written.append(rel)
    return written
