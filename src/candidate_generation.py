"""High-recall candidate generation (Phase 5 / 5b).

- word_addr channel (word TF-IDF over addr_core_joined): links pairs whose
  names differ (renamed / cross-script) but whose addresses match.
- Variant channels (word_name, rare): alias segments become separate pool AND
  query documents; results are mapped back and min-rank deduped per pair.
  Exact channel indexes variant keys.
- Parallel chunk workers (fork): per-chunk spread budget = max_spread //
  n_jobs, so TOTAL transient memory stays bounded; chunk CONTENT is
  independent of n_jobs (only part grouping changes), guarded by a chunk
  signature in the resume manifest. Workers never touch boto3.
- Parts are aggregated to part_target_pairs rows per file.

Leakage contract unchanged: pool statistics are label-free; candidates are
generated once per split and shared across OOF folds.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import logging
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from .aws_utils import local_artifact_path, publish_artifact, read_json, write_json
from .blocking import (code_fingerprint, ensure_normalized, iter_column,
                       iter_variants, load_ids)
from .data_loader import source_available
from .logging_utils import stage_timer

log = logging.getLogger(__name__)

CHANNELS = ("exact", "char_name", "word_name", "rare", "word_addr", "char_addr")
CHANNEL_BIT = {c: 1 << i for i, c in enumerate(CHANNELS)}
RANK_COLUMNS = {c: f"rank_{c}" for c in CHANNELS}
SCORE_COLUMNS = {c: f"score_{c}" for c in
                 ("char_name", "word_name", "rare", "word_addr", "char_addr")}

# worker state, populated before fork and read inside _process_chunk
_CH_STATE: Dict[str, Any] = {}


def candidates_rel(split: str, dry_run: bool = False) -> str:
    return f"candidates/{split}_dryrun" if dry_run else f"candidates/{split}"


def candidates_dir(cfg, split: str, dry_run: bool = False) -> Path:
    d = local_artifact_path(cfg, candidates_rel(split, dry_run))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# engine pieces (unit-tested)
# ---------------------------------------------------------------------------


def _pack_chunks(spread: np.ndarray, max_spread: float,
                 max_rows: int) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    start = 0
    acc = 0.0
    for i in range(len(spread)):
        s = float(spread[i])
        if i > start and (acc + s > max_spread or (i - start) >= max_rows):
            chunks.append((start, i))
            start = i
            acc = s
        else:
            acc += s
    if start < len(spread):
        chunks.append((start, len(spread)))
    return chunks


def _empty_result():
    z32 = np.empty(0, np.int32)
    return z32, z32.copy(), np.empty(0, np.uint8), np.empty(0, np.float32)


def _topk_rows(S, k: int, row_offset: int):
    indptr, indices, data = S.indptr, S.indices, S.data
    qs, ps, rs, ss = [], [], [], []
    for i in range(S.shape[0]):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        if hi <= lo:
            continue
        pi, pv = indices[lo:hi], data[lo:hi]
        if hi - lo > k:
            sel = np.argpartition(pv, hi - lo - k)[hi - lo - k:]
            pi, pv = pi[sel], pv[sel]
        order = np.argsort(-pv, kind="stable")[:k]
        pi, pv = pi[order], pv[order]
        qs.append(np.full(pi.size, row_offset + i, dtype=np.int32))
        ps.append(pi.astype(np.int32))
        rs.append(np.arange(1, pi.size + 1, dtype=np.uint8))
        ss.append(pv.astype(np.float32))
    if not qs:
        return _empty_result()
    return (np.concatenate(qs), np.concatenate(ps),
            np.concatenate(rs), np.concatenate(ss))


def _reduce_pairs(q, p, r, s, n_pool: int):
    """Collapse duplicate (row, pool) pairs from variant docs: min rank, max score."""
    if q.size <= 1:
        return q, p, r, s
    keys = q.astype(np.int64) * n_pool + p.astype(np.int64)
    order = np.lexsort((r, keys))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    sel = order[first]
    order2 = np.lexsort((-s.astype(np.float32), keys))
    first2 = np.r_[True, keys[order2][1:] != keys[order2][:-1]]
    return q[sel], p[sel], r[sel], s[order2[first2]]


def _merge_channel_results(n_pool: int, per_channel: Dict[str, Tuple]
                           ) -> Optional[Dict[str, Any]]:
    """Dedupe per-chunk channel results into one table with provenance."""
    if not per_channel:
        return None
    keys = np.concatenate([q.astype(np.int64) * n_pool + p.astype(np.int64)
                           for q, p, _r, _s in per_channel.values()])
    if keys.size == 0:
        return None
    uniq, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = int(uniq.size)
    bits = np.zeros(n, dtype=np.uint8)
    ranks = {c: np.zeros(n, dtype=np.uint8) for c in CHANNELS}
    scores = {c: np.zeros(n, dtype=np.float32) for c in SCORE_COLUMNS}
    offset = 0
    for ch, (_q, _p, r, s) in per_channel.items():
        inv = inverse[offset:offset + r.size]
        bits[inv] |= CHANNEL_BIT[ch]
        ranks[ch][inv] = r
        if ch in scores:
            scores[ch][inv] = s
        offset += r.size
    return {"n": n, "s1_idx": (uniq // n_pool).astype(np.int32),
            "pool_idx": (uniq % n_pool).astype(np.int32),
            "channel_bits": bits, "ranks": ranks, "scores": scores}


def _concat_merged(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "s1_idx": np.concatenate([m["s1_idx"] for m in items]),
        "pool_idx": np.concatenate([m["pool_idx"] for m in items]),
        "channel_bits": np.concatenate([m["channel_bits"] for m in items]),
        "ranks": {c: np.concatenate([m["ranks"][c] for m in items])
                  for c in CHANNELS},
        "scores": {c: np.concatenate([m["scores"][c] for m in items])
                   for c in SCORE_COLUMNS},
    }
    out["n"] = int(out["s1_idx"].size)
    return out


def _make_tfidf(kind: str, params: Dict[str, Any], pool_docs, query_docs):
    if kind == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3),
                              max_df=float(params.get("max_df", 0.02)),
                              dtype=np.float32, sublinear_tf=True)
    else:
        abs_df = params.get("max_df_abs")
        max_df = (int(abs_df) if abs_df is not None
                  else float(params.get("max_df", 0.02)))
        vec = TfidfVectorizer(analyzer=str.split, lowercase=False,
                              max_df=max_df, dtype=np.float32,
                              sublinear_tf=True)
    P = vec.fit_transform(pool_docs)
    Q = vec.transform(query_docs)
    return vec, P, Q, int(params.get("k", 30))


def _expand_variant_docs(paths, limit: Optional[int] = None
                         ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """(docs, row_map, offsets): every row contributes >= 1 doc, so offsets
    are strictly increasing."""
    docs: List[str] = []
    row_map: List[int] = []
    counts: List[int] = []
    it = iter_variants(paths)
    if limit is not None:
        it = itertools.islice(it, limit)
    n = 0
    for i, variants in enumerate(it):
        kept = [v for v in variants if v] or [""]
        for v in kept:
            docs.append(v)
            row_map.append(i)
        counts.append(len(kept))
        n = i + 1
    offsets = np.zeros(n + 1, dtype=np.int64)
    if n:
        offsets[1:] = np.cumsum(counts)
    return docs, np.asarray(row_map, dtype=np.int32), offsets


def _variant_key(v: str) -> str:
    return " ".join(sorted(v.split()))


def _build_exact_index_variants(pool_paths, cap: int
                                ) -> Tuple[Dict[str, List[int]], int]:
    index: Dict[str, List[int]] = {}
    truncated = 0
    for i, variants in enumerate(iter_variants(pool_paths)):
        for v in variants:
            if not v:
                continue
            key = _variant_key(v)
            posts = index.get(key)
            if posts is None:
                index[key] = [i]
            elif posts[-1] != i:
                if len(posts) < cap:
                    posts.append(i)
                else:
                    truncated += 1
    return index, truncated


def _exact_channel_chunk(index: Dict[str, List[int]],
                         query_key_variants: List[List[str]], k: int,
                         start: int, end: int):
    qs, ps, rs = [], [], []
    for i in range(start, end):
        seen: set = set()
        for key in query_key_variants[i]:
            posts = index.get(key) if key else None
            if not posts:
                continue
            take = [t for t in posts[:k] if t not in seen]
            if not take:
                continue
            seen.update(take)
            qs.append(np.full(len(take), i, dtype=np.int32))
            ps.append(np.asarray(take, dtype=np.int32))
            rs.append(np.arange(1, len(take) + 1, dtype=np.uint8))
    if not qs:
        return _empty_result()
    r = np.concatenate(rs)
    return np.concatenate(qs), np.concatenate(ps), r, np.ones(r.size, np.float32)


def _chunk_signature(chunks: List[Tuple[int, int]]) -> str:
    blob = json.dumps([list(c) for c in chunks], separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _write_candidate_part(out_dir: Path, name: str, merged: Dict[str, Any]) -> int:
    cols: Dict[str, Any] = {"s1_idx": merged["s1_idx"],
                            "pool_idx": merged["pool_idx"],
                            "channel_bits": merged["channel_bits"]}
    for c in CHANNELS:
        cols[RANK_COLUMNS[c]] = merged["ranks"][c]
    for c in SCORE_COLUMNS:
        cols[SCORE_COLUMNS[c]] = merged["scores"][c]
    table = pa.table({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(table, out_dir / name, compression="snappy")
    return table.num_rows


# ---------------------------------------------------------------------------
# chunk worker (parent process or forked child; reads only _CH_STATE)
# ---------------------------------------------------------------------------


def _process_chunk(ci: int) -> Dict[str, Any]:
    ch = _CH_STATE
    start, end = ch["chunks"][ci]
    per_channel: Dict[str, Tuple] = {}
    for name, m in ch["matrices"].items():
        if m["offsets"] is not None:          # variant channel: expanded rows
            es, ee = int(m["offsets"][start]), int(m["offsets"][end])
            if ee <= es:
                continue
            S = (m["Q"][es:ee] @ m["PT"]).tocsr()
            q, p, r, s = _topk_rows(S, m["k"], es)
            del S
            if q.size:
                q = m["row_map"][q]
                p = m["pool_map"][p]
                q, p, r, s = _reduce_pairs(q, p, r, s, ch["n_pool"])
        else:                                 # identity channel
            S = (m["Q"][start:end] @ m["PT"]).tocsr()
            q, p, r, s = _topk_rows(S, m["k"], start)
            del S
        if q.size:
            per_channel[name] = (q, p, r, s)
    if ch["exact_index"] is not None:
        q, p, r, s = _exact_channel_chunk(ch["exact_index"],
                                          ch["query_exact_keys"],
                                          ch["k_exact"], start, end)
        if q.size:
            per_channel["exact"] = (q, p, r, s)
    merged = _merge_channel_results(ch["n_pool"], per_channel)
    return {"ci": ci, "start": start, "end": end,
            "rows": merged["n"] if merged else 0, "merged": merged}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _write_ids_parquet(path: Path, ids: np.ndarray) -> None:
    pq.write_table(pa.table({"entity_id": pa.array(ids.tolist(), type=pa.string())}),
                   path, compression="snappy")


def run_retrieval(cfg, split: str = "train", limit_rows: Optional[int] = None,
                  dry_run: bool = False,
                  channel_subset: Optional[Sequence[str]] = None,
                  n_jobs: Optional[int] = None, max_spread: Optional[int] = None,
                  log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    t0 = time.time()
    log = log or logging.getLogger(__name__)
    if limit_rows:
        dry_run = True
    if dry_run:
        shutil.rmtree(candidates_dir(cfg, split, True), ignore_errors=True)

    norm = ensure_normalized(cfg, split)
    s1_paths = norm["source1"]
    pool_paths = norm["source2"] + norm["source3"]
    s1_ids = load_ids(s1_paths)
    pool_ids = load_ids(pool_paths)
    n_pool = int(pool_ids.size)
    n_s1 = int(s1_ids.size if not limit_rows else min(s1_ids.size, limit_rows))
    log.info("pool: %d records; s1 rows: %d (split=%s)", n_pool, n_s1, split)

    out_dir = candidates_dir(cfg, split, dry_run)
    publish = not dry_run
    rel = candidates_rel(split, dry_run)
    _write_ids_parquet(out_dir / "s1_ids.parquet", s1_ids[:n_s1])
    _write_ids_parquet(out_dir / "pool_ids.parquet", pool_ids)
    if publish:
        publish_artifact(cfg, f"{rel}/s1_ids.parquet")
        publish_artifact(cfg, f"{rel}/pool_ids.parquet")

    channels_cfg = {c: dict(p) for c, p in dict(cfg.retrieval.channels).items()
                    if p.get("enabled", True)}
    if channel_subset:
        channels_cfg = {c: p for c, p in channels_cfg.items()
                        if c in set(channel_subset)}

    def qdocs(column: str) -> List[str]:
        return list(itertools.islice(iter_column(s1_paths, column), n_s1))

    q_alnum = qdocs("name_alnum")
    q_addr = qdocs("addr_core_joined")
    q_addr_alnum = qdocs("addr_alnum") if "char_addr" in channels_cfg else []
    q_variants, q_row_map, q_offsets = _expand_variant_docs(s1_paths, n_s1)
    need_variants = any(c in channels_cfg for c in ("word_name", "rare"))
    pool_variants, pool_map = ([], None)
    if need_variants:
        pool_variants, pool_map, _ = _expand_variant_docs(pool_paths)

    matrices: Dict[str, Dict[str, Any]] = {}
    spread = np.zeros(n_s1, dtype=np.float64)
    plain = {"pool_map": None, "row_map": None, "offsets": None}
    for name in ("char_name", "word_name", "rare", "word_addr", "char_addr"):
        if name not in channels_cfg:
            continue
        params = channels_cfg[name]
        if name == "char_name":
            vec, P, Q, k = _make_tfidf("char", params,
                                       iter_column(pool_paths, "name_alnum"),
                                       q_alnum)
            info = plain
        elif name == "word_addr":
            vec, P, Q, k = _make_tfidf(
                "word", params, iter_column(pool_paths, "addr_core_joined"),
                q_addr)
            info = plain
        elif name == "char_addr":
            vec, P, Q, k = _make_tfidf(
                "char", params, iter_column(pool_paths, "addr_alnum"),
                q_addr_alnum)
            info = plain
        else:                                   # word_name / rare: variants
            vec, P, Q, k = _make_tfidf("word", params, pool_variants, q_variants)
            info = {"pool_map": pool_map, "row_map": q_row_map,
                    "offsets": q_offsets}
        df = np.asarray(P.getnnz(axis=0)).ravel().astype(np.float64)  # per TERM
        PT = P.T.tocsr()                        # built once, reused per chunk
        del P
        Qb = Q.copy()
        Qb.data = np.ones_like(Qb.data)
        spread_ch = np.asarray(Qb @ df).ravel()
        if info["offsets"] is not None:
            spread_ch = np.add.reduceat(spread_ch, info["offsets"][:-1])
        spread = np.maximum(spread, spread_ch[:n_s1])
        matrices[name] = {"Q": Q, "PT": PT, "k": k, **info}
        log.info("channel %s: vocab=%d pool_nnz=%d empty_queries=%d", name,
                 len(vec.vocabulary_), PT.nnz, int((Q.getnnz(axis=1) == 0).sum()))
        gc.collect()
    pool_variants = []

    exact_index, exact_trunc, k_exact = None, 0, 0
    if "exact" in channels_cfg:
        k_exact = int(channels_cfg["exact"].get("k", 100))
        exact_index, exact_trunc = _build_exact_index_variants(
            pool_paths, int(channels_cfg["exact"].get("max_postings", 300)))
        log.info("channel exact: %d keys (truncated postings: %d)",
                 len(exact_index), exact_trunc)
    query_exact_keys: List[List[str]] = [
        [_variant_key(v) for v in variants if v] or [""]
        for variants in itertools.islice(iter_variants(s1_paths), n_s1)]

    can_fork = "fork" in multiprocessing.get_all_start_methods()
    n_jobs_eff = max(1, int(n_jobs if n_jobs is not None
                            else getattr(cfg.execution, "n_jobs", 1)))
    parallel = n_jobs_eff > 1 and can_fork
    budget = max(1, int((max_spread if max_spread is not None
                         else cfg.retrieval.max_spread)
                        // (n_jobs_eff if parallel else 1)))
    chunks = _pack_chunks(spread, float(budget), int(cfg.retrieval.s1_chunk_rows))
    sig = _chunk_signature(chunks)
    log.info("retrieval plan: %d chunks, budget=%d/chunk, n_jobs=%d",
             len(chunks), budget, n_jobs_eff)

    # ---- resume state: content identity + chunking signature ------------
    # run_key captures CONTENT-affecting inputs (code + channel params + split).
    # Layout-only knobs (max_spread, s1_chunk_rows, part_target, n_jobs) are
    # excluded: they change grouping, never content.
    here = Path(__file__).resolve().parent
    content_blob = json.dumps({"split": split, "channels": cfg.retrieval.channels},
                              sort_keys=True)
    run_key = (code_fingerprint([here / "normalization.py", here / "blocking.py",
                                 here / "candidate_generation.py",
                                 here / "candidate_metrics.py",
                                 here / "config.py"])
               + ":" + hashlib.sha1(content_blob.encode()).hexdigest()[:12])

    chunks_path = out_dir / "chunks.json"
    parts_path = out_dir / "parts.json"
    done_entries: List[Dict[str, Any]] = []
    part_files: List[str] = []
    stale = True
    if chunks_path.exists() and parts_path.exists():
        stored = read_json(chunks_path) or {}
        if stored.get("run_key") == run_key and stored.get("signature") == sig:
            done_entries = [e for e in stored.get("entries", []) if "merged" not in e]
            part_files = list(read_json(parts_path) or [])
            stale = False
        else:
            reason = ("content changed" if stored.get("run_key") != run_key
                      else "chunking changed")
            log.warning("candidate resume state discarded (%s); regenerating "
                        "all parts", reason)
    if stale:
        part_files = []
        for f in out_dir.glob("part_*.parquet"):
            f.unlink()
        for f in (parts_path, chunks_path):
            if f.exists():
                f.unlink()
    else:
        known = set(part_files)
        for f in out_dir.glob("part_*.parquet"):
            if f.name not in known:
                f.unlink()
    done_ids = {int(e["ci"]) for e in done_entries}
    pending = [ci for ci in range(len(chunks)) if ci not in done_ids]
    part_target = max(1, int(cfg.retrieval.part_target_pairs))

    _CH_STATE.clear()
    _CH_STATE.update({"chunks": chunks, "matrices": matrices, "n_pool": n_pool,
                      "exact_index": exact_index, "k_exact": k_exact,
                      "query_exact_keys": query_exact_keys})

    state = {"buffer": [], "buffer_pairs": 0, "unflushed": [], "completed":
             len(done_entries), "pairs": sum(int(e.get("rows", 0))
                                             for e in done_entries)}

    def _save_chunks() -> None:
        write_json(chunks_path, {"signature": sig, "run_key": run_key,
                                 "entries": done_entries})
        if publish:
            publish_artifact(cfg, f"{rel}/chunks.json")

    def _flush() -> None:
        if state["buffer"]:
            name = f"part_{len(part_files):04d}.parquet"
            _write_candidate_part(out_dir, name, _concat_merged(state["buffer"]))
            part_files.append(name)
            write_json(parts_path, part_files)
            if publish:
                publish_artifact(cfg, f"{rel}/{name}")
                publish_artifact(cfg, f"{rel}/parts.json")
            state["buffer"], state["buffer_pairs"] = [], 0
        done_entries.extend(state["unflushed"])
        state["unflushed"] = []
        _save_chunks()

    def _record(res: Dict[str, Any]) -> None:
        merged = res.pop("merged", None)
        rows = int(res.get("rows", 0))
        state["unflushed"].append({"ci": res["ci"], "start": res["start"],
                                   "end": res["end"], "rows": rows})
        if merged is not None and merged.get("n"):
            state["buffer"].append(merged)
            state["buffer_pairs"] += merged["n"]
        state["completed"] += 1
        state["pairs"] += rows
        if state["buffer_pairs"] >= part_target:
            _flush()
        if state["completed"] % 50 == 0 or state["completed"] == len(chunks):
            el = time.time() - t0
            log.info("chunks %d/%d: pairs=%d elapsed=%.0fs eta=%.0fs",
                     state["completed"], len(chunks), state["pairs"], el,
                     el / max(1, state["completed"]) *
                     (len(chunks) - state["completed"]))

    timer = stage_timer(f"candidates[{split}]", cfg)
    timer.start()
    if parallel:
        try:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=n_jobs_eff) as pool:
                for res in pool.imap_unordered(_process_chunk, pending):
                    _record(res)
        except Exception as exc:            # keep resume state, go serial
            log.warning("parallel retrieval failed (%s); serial fallback", exc)
    finished = ({int(e["ci"]) for e in done_entries}
                | {int(e["ci"]) for e in state["unflushed"]})
    for ci in pending:
        if ci not in finished:
            _record(_process_chunk(ci))
    _flush()
    timer.stop()
    _CH_STATE.clear()
    del matrices, exact_index, query_exact_keys
    gc.collect()

    summary: Dict[str, Any] = {
        "split": split, "n_pool": n_pool, "n_s1": n_s1,
        "total_pairs": sum(int(e.get("rows", 0)) for e in done_entries),
        "parts": len(part_files),
        "channels": sorted(channels_cfg.keys()),
        "exact_truncated_keys": exact_trunc,
        "n_jobs": n_jobs_eff, "elapsed_s": round(time.time() - t0, 1),
        "dry_run": dry_run, "peak_rss_mb": timer.peak_rss_mb,
    }

    gt_file = getattr(getattr(cfg.data, split), "ground_truth", "")
    if gt_file and source_available(cfg, split, "ground_truth"):
        from .candidate_metrics import retrieval_report, write_report
        from .ground_truth import load_ground_truth
        gt = load_ground_truth(cfg, split)
        report = retrieval_report(cfg, split, gt, dry_run=dry_run, log=log)
        write_report(cfg, split, report, dry_run=dry_run)
        summary["pair_recall"] = report.get("pair_recall")
        summary["avg_candidates_per_entity"] = report.get("avg_candidates_per_entity")
        summary["missed_pairs_total"] = report.get("missed_pairs_total")
    return summary
