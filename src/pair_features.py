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

COMPUTED_COLUMNS = (
    "name_exact", "name_core_exact", "name_sort", "name_set", "name_jw",
    "name_len_ratio", "name_known",
    "addr_exact", "addr_sort", "addr_set", "addr_len_ratio", "addr_known",
    "house_match", "house_known", "digits_match", "digits_known",
    "state_match", "state_known", "country_match", "country_known",
)

FEATURE_COLUMNS = COMPUTED_COLUMNS + RETRIEVAL_COLUMNS + ("n_channels",)

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
        states.append(frozenset(_extract_states(toks)))
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
    for c in RETRIEVAL_COLUMNS:
        out[c] = retrieval[c].astype(np.float32)
    out["n_channels"] = _POPCOUNT[channel_bits].astype(np.float32)
    return out
