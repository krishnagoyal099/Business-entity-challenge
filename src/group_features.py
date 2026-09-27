"""Stage-2 group features: link a candidate to the S1 entity's confident matches.

A true S2/S3 record whose name is garbage ("Dovacira") usually still shares its
address with another, confidently matched record of the same entity. Stage 1
scores pairs in isolation; stage 2 adds, per candidate j of entity i, how j
relates to i's "anchors" (other candidates of i with stage-1 prob >= ANCHOR_P).

Only the top-K candidates per entity (by stage-1 prob) enter stage 2; all
features are S1-group local (no pool-side competition counts), so they do not
shift between the train sample and the full test set.
"""
from __future__ import annotations

import multiprocessing
from typing import Dict, Tuple

import numpy as np
from rapidfuzz import fuzz

GROUP_COLUMNS = (
    "stage1_p", "g_rank", "g_n_anchor", "g_p_max_other", "g_p_sum_other",
    "g_addr_eq", "g_addr_max", "g_num_eq", "g_name_skel_max", "g_anchor_addr_share",
)
ANCHOR_P = 0.5
MAX_ANCHORS = 5

_G: Dict[str, object] = {}


def topk_per_entity(s1_idx: np.ndarray, prob: np.ndarray, k: int) -> np.ndarray:
    """Row indices of the top-k pairs per S1 entity, grouped, best first."""
    order = np.lexsort((-prob, s1_idx))
    s = s1_idx[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    sizes = np.diff(np.r_[starts, s.size])
    pos = np.arange(s.size) - np.repeat(starts, sizes)
    return order[pos < k]


def _group_range(bounds: Tuple[int, int]) -> Tuple[int, Dict[str, np.ndarray]]:
    st = _G
    lo_g, hi_g = bounds
    starts, sizes = st["starts"], st["sizes"]
    pool, prob = st["pool"], st["prob"]
    addr, nums, nskel = st["addr"], st["nums"], st["nskel"]
    r0 = int(starts[lo_g])
    r1 = int(starts[hi_g - 1] + sizes[hi_g - 1])
    n = r1 - r0
    out = {c: np.zeros(n, dtype=np.float32) for c in GROUP_COLUMNS}
    for g in range(lo_g, hi_g):
        a, m = int(starts[g]), int(sizes[g])
        ps = prob[a:a + m]                       # sorted desc within group
        tot = float(ps.sum())
        anchors = [a + t for t in range(min(m, MAX_ANCHORS)) if ps[t] >= ANCHOR_P]
        # how many anchors share one address: a consistent entity address
        share = 0.0
        if len(anchors) >= 2:
            ad = [addr[int(pool[x])] for x in anchors]
            share = max(sum(1 for y in ad if y and y == z) for z in ad) / len(ad)
        for t in range(m):
            row = a + t
            o = row - r0
            pj = float(ps[t])
            out["stage1_p"][o] = pj
            out["g_rank"][o] = t + 1
            out["g_p_max_other"][o] = float(ps[1]) if t == 0 and m > 1 else (
                float(ps[0]) if t > 0 else 0.0)
            out["g_p_sum_other"][o] = tot - pj
            out["g_anchor_addr_share"][o] = share
            others = [x for x in anchors if x != row]
            if not others:
                continue
            out["g_n_anchor"][o] = len(others)
            j = int(pool[row])
            aj, nj, kj = addr[j], nums[j], nskel[j]
            best_addr = best_name = 0.0
            eq = neq = 0.0
            for x in others:
                q = int(pool[x])
                aq = addr[q]
                if aj and aq:
                    if aj == aq:
                        eq = 1.0
                        best_addr = 100.0
                    elif best_addr < 100.0:
                        best_addr = max(best_addr, fuzz.token_set_ratio(aj, aq))
                if nj and nj == nums[q]:
                    neq = 1.0
                if kj and nskel[q]:
                    best_name = max(best_name, fuzz.token_set_ratio(kj, nskel[q]))
            out["g_addr_eq"][o] = eq
            out["g_addr_max"][o] = best_addr
            out["g_num_eq"][o] = neq
            out["g_name_skel_max"][o] = best_name
    return r0, out


def compute_group_features(s1_idx: np.ndarray, pool_idx: np.ndarray,
                           prob: np.ndarray, pool_side: Dict[str, list],
                           n_jobs: int = 1) -> Dict[str, np.ndarray]:
    """Rows must be grouped by s1 and sorted by prob desc within each group
    (as returned by `topk_per_entity`). Returns GROUP_COLUMNS arrays."""
    n = int(s1_idx.size)
    out = {c: np.zeros(n, dtype=np.float32) for c in GROUP_COLUMNS}
    if n == 0:
        return out
    starts = np.r_[0, np.flatnonzero(s1_idx[1:] != s1_idx[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    n_g = int(starts.size)
    step = 20000
    bounds = [(i, min(i + step, n_g)) for i in range(0, n_g, step)]
    _G.update({"starts": starts, "sizes": sizes, "pool": pool_idx, "prob": prob,
               "addr": pool_side["addr_core"], "nums": pool_side["nums"],
               "nskel": pool_side["name_skel"]})
    try:
        if n_jobs > 1 and "fork" in multiprocessing.get_all_start_methods():
            with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
                results = pool.imap_unordered(_group_range, bounds)
                for r0, part in results:
                    for c in GROUP_COLUMNS:
                        out[c][r0:r0 + part[c].size] = part[c]
        else:
            for b in bounds:
                r0, part = _group_range(b)
                for c in GROUP_COLUMNS:
                    out[c][r0:r0 + part[c].size] = part[c]
    finally:
        _G.clear()
    return out
