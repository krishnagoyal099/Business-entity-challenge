"""Pair feature computation (Phase 7-lite).

- Side tables (per-row fields incl. derived house/digits/states) are built ONCE
  in the parent; per-pair work is pure lookups + rapidfuzz calls.
- Missing fields never fake similarity: ratio("", "") == 100 in rapidfuzz, so an
  empty name/address zeroes the sims and clears the *_known flag.
- Retrieval provenance (ranks/scores/channels) is passed through as features.
- Labels (train only) are vectorized int64 key lookups against the GT.
"""
from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import iter_column
from .normalization import _extract_states

log = logging.getLogger(__name__)

RETRIEVAL_COLUMNS = (
    "rank_exact", "rank_char_name", "rank_word_name", "rank_rare",
    "rank_word_addr", "rank_char_addr",
    "score_char_name", "score_word_name", "score_rare",
    "score_word_addr", "score_char_addr",
)

CONTEXT_COLUMNS = (
    "ctx_rank_best", "ctx_rank_wa", "ctx_n_cand",
    "ctx_top1", "ctx_margin1", "ctx_rel_best",
)

_SCORE_COLS = ("score_char_name", "score_word_name", "score_rare",
               "score_word_addr", "score_char_addr")

COMPUTED_COLUMNS = (
    "name_exact", "name_core_exact", "name_sort", "name_set", "name_jw",
    "name_len_ratio", "name_known",
    "addr_exact", "addr_sort", "addr_set", "addr_len_ratio", "addr_known",
    "house_match", "house_known", "digits_match", "digits_known",
    "state_match", "state_known", "country_match", "country_known",
)

# v1 feature list (kept so the v1 model and old feature parts stay usable)
LEGACY_FEATURE_COLUMNS = COMPUTED_COLUMNS + RETRIEVAL_COLUMNS + ("n_channels",)

FEATURE_COLUMNS = COMPUTED_COLUMNS + CONTEXT_COLUMNS + RETRIEVAL_COLUMNS + (
    "n_channels",)

# 2-letter state codes that are also common words in other languages
# ("rue de la ..." -> DE, LA; "in", "or", "me" ...). Mid-address they are read
# as words; only an address-initial/final code or one before a zip counts.
_AMBIGUOUS_STATE_TOKENS = frozenset({"de", "la", "in", "or", "me", "co", "al",
                                     "ma", "pa", "hi", "id", "ok", "oh", "ne"})


def address_states(toks: List[str]) -> frozenset:
    """US/India states of an address, ignoring ambiguous mid-address words."""
    n = len(toks)
    keep = [t for i, t in enumerate(toks)
            if t not in _AMBIGUOUS_STATE_TOKENS or i == 0 or i == n - 1
            or (i + 1 < n and toks[i + 1].isdigit() and len(toks[i + 1]) == 5)]
    return frozenset(_extract_states(keep))


_POPCOUNT = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def build_side_table(paths, limit_rows: Optional[int] = None) -> Dict[str, list]:
    """Per-row fields used by pair features; row order == row index."""
    def col(name: str) -> List[str]:
        it = iter_column(paths, name)
        if limit_rows is not None:
            it = itertools.islice(it, limit_rows)
        return list(it)

    name_core = col("name_core_joined")
    name_sorted = col("name_core_sorted")
    name_alnum = col("name_alnum")
    addr_core = col("addr_core_joined")
    country = col("country_norm")
    house: List[str] = []
    digits: List[str] = []
    states: List[frozenset] = []
    for a in addr_core:
        toks = a.split()
        ht = next((t for t in toks if any(c.isdigit() for c in t)), "")
        house.append("".join(c for c in ht if c.isdigit()))
        digits.append("".join(c for c in a if c.isdigit()))
        states.append(address_states(toks))
    return {"name_core": name_core, "name_sorted": name_sorted,
            "name_alnum": name_alnum, "addr_core": addr_core,
            "country": country, "house": house, "digits": digits,
            "states": states}


def label_for_keys(keys: np.ndarray, gt_keys: Optional[np.ndarray]) -> np.ndarray:
    """Vectorized membership: 1 if the (s1,pool) key is a true pair."""
    if gt_keys is None or gt_keys.size == 0:
        return np.zeros(keys.size, dtype=np.uint8)
    pos = np.searchsorted(gt_keys, keys)
    pos_c = np.minimum(pos, gt_keys.size - 1)
    return ((pos < gt_keys.size) & (gt_keys[pos_c] == keys)).astype(np.uint8)


def _entity_context(s1_idx: np.ndarray,
                    retrieval: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Within-entity context from retrieval scores (entity rows never span parts)."""
    n = int(s1_idx.size)
    ctx = {c: np.zeros(n, dtype=np.float32) for c in CONTEXT_COLUMNS}
    if n == 0:
        return ctx
    best = np.maximum.reduce([retrieval[c].astype(np.float32) for c in _SCORE_COLS])
    order = np.lexsort((-best, s1_idx))
    s_sorted = s1_idx[order]
    starts = np.r_[0, np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    rep = np.repeat(starts, sizes)              # entity start per row (sorted)
    rank_s = (np.arange(n) - rep + 1).astype(np.float32)
    n_cand_s = np.repeat(sizes, sizes).astype(np.float32)
    best_s = best[order]
    top1_s = best_s[rep]
    margin_s = (top1_s - best_s).astype(np.float32)
    rel_s = np.where(top1_s > 1e-9, best_s / np.maximum(top1_s, 1e-9),
                     0.0).astype(np.float32)
    # rank by word_addr specifically; both sorts group entities identically,
    # so starts/sizes/rep are reusable
    wa = retrieval["score_word_addr"].astype(np.float32)
    order_wa = np.lexsort((-wa, s1_idx))
    ctx["ctx_rank_wa"][order_wa] = (np.arange(n) - rep + 1).astype(np.float32)
    for name, arr_s in (("ctx_rank_best", rank_s), ("ctx_n_cand", n_cand_s),
                        ("ctx_top1", top1_s.astype(np.float32)),
                        ("ctx_margin1", margin_s), ("ctx_rel_best", rel_s)):
        ctx[name][order] = arr_s
    return ctx


def compute_part_features(s1_idx: np.ndarray, pool_idx: np.ndarray,
                          channel_bits: np.ndarray,
                          retrieval: Dict[str, np.ndarray],
                          s1_side: Dict[str, list],
                          pool_side: Dict[str, list]) -> Dict[str, np.ndarray]:
    """Compute all FEATURE_COLUMNS for one block of candidate pairs."""
    n = int(s1_idx.size)
    F = {c: np.zeros(n, dtype=np.float32) for c in COMPUTED_COLUMNS}
    s_na, p_na = s1_side["name_alnum"], pool_side["name_alnum"]
    s_nc, p_nc = s1_side["name_core"], pool_side["name_core"]
    s_ns, p_ns = s1_side["name_sorted"], pool_side["name_sorted"]
    s_ac, p_ac = s1_side["addr_core"], pool_side["addr_core"]
    s_ct, p_ct = s1_side["country"], pool_side["country"]
    s_ho, p_ho = s1_side["house"], pool_side["house"]
    s_di, p_di = s1_side["digits"], pool_side["digits"]
    s_st, p_st = s1_side["states"], pool_side["states"]
    for i in range(n):
        a = int(s1_idx[i])
        b = int(pool_idx[i])
        fa, fb = s_na[a], p_na[b]
        if fa and fb:
            F["name_known"][i] = 1.0
            F["name_exact"][i] = 1.0 if fa == fb else 0.0
            F["name_core_exact"][i] = 1.0 if (s_ns[a] and s_ns[a] == p_ns[b]) else 0.0
            F["name_sort"][i] = fuzz.token_sort_ratio(s_nc[a], p_nc[b])
            F["name_set"][i] = fuzz.token_set_ratio(s_nc[a], p_nc[b])
            F["name_jw"][i] = 100.0 * JaroWinkler.similarity(fa, fb)
            F["name_len_ratio"][i] = len(fa) / len(fb)
        aa, ab = s_ac[a], p_ac[b]
        if aa and ab:
            F["addr_known"][i] = 1.0
            F["addr_exact"][i] = 1.0 if aa == ab else 0.0
            F["addr_sort"][i] = fuzz.token_sort_ratio(aa, ab)
            F["addr_set"][i] = fuzz.token_set_ratio(aa, ab)
            F["addr_len_ratio"][i] = len(aa) / len(ab)
        ha, hb = s_ho[a], p_ho[b]
        if ha and hb:
            F["house_known"][i] = 1.0
            F["house_match"][i] = 1.0 if ha == hb else 0.0
        da, db = s_di[a], p_di[b]
        if da and db:
            F["digits_known"][i] = 1.0
            F["digits_match"][i] = 1.0 if (da == db or da in db or db in da) else 0.0
        sa, sb = s_st[a], p_st[b]
        if sa and sb:
            F["state_known"][i] = 1.0
            F["state_match"][i] = 1.0 if (sa & sb) else 0.0
        ca, cb = s_ct[a], p_ct[b]
        if ca and cb:
            F["country_known"][i] = 1.0
            F["country_match"][i] = 1.0 if ca == cb else 0.0
    out = dict(F)
    out.update(_entity_context(s1_idx, retrieval))
    for c in RETRIEVAL_COLUMNS:
        out[c] = retrieval[c].astype(np.float32)
    out["n_channels"] = _POPCOUNT[channel_bits].astype(np.float32)
    return out
