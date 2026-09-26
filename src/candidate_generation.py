"""High-recall candidate generation (Phase 5).

Unified engine:
- Vector channels (char/word TF-IDF): fit_transform on the pool's normalized
  parquet column (streamed); queries transformed from S1 columns. Retrieval =
  chunked sparse matmul (Q_chunk @ P.T) + per-row top-k. Chunks are packed by
  SPREAD = summed pool df of the chunk's retained terms, bounding transient
  memory regardless of common-name queries.
- Exact channel: capped hash join on name_core_sorted.
- Union with provenance: channel bitmask + per-channel rank/score, deduped per
  chunk via combined int64 keys, written as parquet parts.

Leakage contract: pool statistics (df/IDF, postings) derive from pool TEXT only
- no labels - and the identical procedure runs at test time. Candidates are
generated once per split and shared across OOF folds.
"""
from __future__ import annotations

import gc
import itertools
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from .aws_utils import local_artifact_path, publish_artifact, read_json, write_json
from .blocking import build_exact_index, ensure_normalized, iter_column, load_ids
from .data_loader import source_available
from .logging_utils import stage_timer

log = logging.getLogger(__name__)

CHANNELS = ("exact", "char_name", "word_name", "rare", "char_addr")
CHANNEL_BIT = {c: 1 << i for i, c in enumerate(CHANNELS)}
RANK_COLUMNS = {c: f"rank_{c}" for c in CHANNELS}
SCORE_COLUMNS = {c: f"score_{c}" for c in
                 ("char_name", "word_name", "rare", "char_addr")}
CHANNEL_COLUMN = {"char_name": "name_alnum", "word_name": "name_core_joined",
                  "rare": "name_core_joined", "char_addr": "addr_alnum"}


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
    """Greedily pack contiguous query rows into (start, end) chunks whose summed
    spread stays within budget; a single row always forms a chunk."""
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
    """Per-row top-k from a chunk similarity CSR -> (q_idx, pool_idx, rank, score)."""
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


def _exact_channel_chunk(exact_index: Dict[str, List[int]], query_keys: List[str],
                         k: int, start: int, end: int):
    qs, ps, rs = [], [], []
    for i in range(start, end):
        key = query_keys[i]
        posts = exact_index.get(key) if key else None
        if not posts:
            continue
        take = posts[:k]
        qs.append(np.full(len(take), i, dtype=np.int32))
        ps.append(np.asarray(take, dtype=np.int32))
        rs.append(np.arange(1, len(take) + 1, dtype=np.uint8))
    if not qs:
        return _empty_result()
    r = np.concatenate(rs)
    return np.concatenate(qs), np.concatenate(ps), r, np.ones(r.size, np.float32)


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


def _build_tfidf_channel(name: str, pool_paths, params: Dict[str, Any],
                         query_docs: List[str]):
    column = CHANNEL_COLUMN[name]
    k = int(params.get("k", 30))
    if name in ("char_name", "char_addr"):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3),
                              max_df=float(params.get("max_df", 0.01)),
                              dtype=np.float32, sublinear_tf=True)
    else:
        abs_df = params.get("max_df_abs")
        max_df = (int(abs_df) if abs_df is not None
                  else float(params.get("max_df", 0.02)))
        vec = TfidfVectorizer(analyzer=str.split, lowercase=False, max_df=max_df,
                              dtype=np.float32, sublinear_tf=True)
    pool_mat = vec.fit_transform(iter_column(pool_paths, column))
    q_mat = vec.transform(query_docs)
    return vec, pool_mat, q_mat, k


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _write_ids_parquet(path: Path, ids: np.ndarray) -> None:
    pq.write_table(pa.table({"entity_id": pa.array(ids.tolist(), type=pa.string())}),
                   path, compression="snappy")


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


def run_retrieval(cfg, split: str = "train", limit_rows: Optional[int] = None,
                  dry_run: bool = False,
                  channel_subset: Optional[Sequence[str]] = None,
                  log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Generate candidates for a split; resumable at part granularity."""
    t0 = time.time()
    log = log or logging.getLogger(__name__)
    if limit_rows:
        dry_run = True                 # limited runs are disposable by design
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
    rel = candidates_rel(split, dry_run)
    publish = not dry_run
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

    query_docs = {c: qdocs(c) for c in ("name_alnum", "name_core_joined",
                                        "addr_alnum", "name_core_sorted")}

    matrices: Dict[str, Dict[str, Any]] = {}
    spread = np.zeros(n_s1, dtype=np.float64)
    for name in ("char_name", "word_name", "rare", "char_addr"):
        if name not in channels_cfg:
            continue
        vec, P, Q, k = _build_tfidf_channel(name, pool_paths, channels_cfg[name],
                                            query_docs[CHANNEL_COLUMN[name]])
        term_df = np.asarray(P.getnnz(axis=0), dtype=np.float64)   # per-term df
        Qb = Q.copy()
        Qb.data = np.ones_like(Qb.data)
        spread = np.maximum(spread, np.asarray(Qb @ term_df).reshape(-1))
        PT = P.T.tocsr()               # (V x n_pool), built once for all chunks
        matrices[name] = {"PT": PT, "Q": Q, "k": k}
        del P, Qb
        log.info("channel %s: vocab=%d nnz=%d empty_queries=%d", name,
                 len(vec.vocabulary_), PT.nnz, int((Q.getnnz(axis=1) == 0).sum()))
        gc.collect()

    exact_index, exact_trunc = None, 0
    if "exact" in channels_cfg:
        exact_index, exact_trunc = build_exact_index(
            pool_paths, "name_core_sorted",
            int(channels_cfg["exact"].get("max_postings", 300)))
        log.info("channel exact: %d keys (truncated postings: %d)",
                 len(exact_index), exact_trunc)

    chunks = _pack_chunks(spread, float(cfg.retrieval.max_spread),
                          int(cfg.retrieval.s1_chunk_rows))
    log.info("retrieval plan: %d chunks (avg rows/chunk=%.0f)", len(chunks),
             n_s1 / max(1, len(chunks)))

    chunks_path = out_dir / "chunks.json"
    done: List[Dict[str, Any]] = (read_json(chunks_path)
                                  if chunks_path.exists() else [])
    known = {c.get("file") for c in done if c.get("file")}
    for f in out_dir.glob("part_*.parquet"):
        if f.name not in known:
            f.unlink()
    total_pairs = sum(int(c.get("rows", 0)) for c in done)

    timer = stage_timer(f"candidates[{split}]", cfg)
    timer.start()
    for ci, (start, end) in enumerate(chunks):
        if ci < len(done):
            continue
        per_channel: Dict[str, Tuple] = {}
        for name, m in matrices.items():
            S = (m["Q"][start:end] @ m["PT"]).tocsr()
            res = _topk_rows(S, m["k"], start)
            del S
            if res[0].size:
                per_channel[name] = res
        if exact_index is not None:
            res = _exact_channel_chunk(
                exact_index, query_docs["name_core_sorted"],
                int(channels_cfg["exact"].get("k", 100)), start, end)
            if res[0].size:
                per_channel["exact"] = res
        merged = _merge_channel_results(n_pool, per_channel)
        part_name = f"part_{ci:04d}.parquet"
        rows = 0
        if merged is not None:
            rows = _write_candidate_part(out_dir, part_name, merged)
            if publish:
                publish_artifact(cfg, f"{rel}/{part_name}")
        done.append({"start": start, "end": end,
                     "file": part_name if rows else None, "rows": rows})
        write_json(chunks_path, done)
        if publish:
            publish_artifact(cfg, f"{rel}/chunks.json")
        total_pairs += rows
        if (ci + 1) % 5 == 0 or ci == len(chunks) - 1:
            el = time.time() - t0
            log.info("chunk %d/%d: pairs=%d elapsed=%.0fs eta=%.0fs",
                     ci + 1, len(chunks), total_pairs, el,
                     el / (ci + 1) * (len(chunks) - ci - 1))
    timer.stop()

    del matrices, exact_index
    gc.collect()

    summary: Dict[str, Any] = {
        "split": split, "n_pool": n_pool, "n_s1": n_s1,
        "total_pairs": total_pairs,
        "parts": sum(1 for c in done if c.get("rows")),
        "channels": sorted(channels_cfg.keys()),
        "exact_truncated_keys": exact_trunc,
        "elapsed_s": round(time.time() - t0, 1), "dry_run": dry_run,
        "peak_rss_mb": timer.peak_rss_mb,
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
    return summary
