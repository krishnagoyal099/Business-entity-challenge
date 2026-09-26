#!/usr/bin/env python
"""Phase 7-lite: compute pair features + labels over the candidate parts.

Usage (full train, on the big instance, after candidates land):
  python scripts/build_features.py --config configs/aws_cpu.yaml \
      --split train --n-jobs 90
Dry-run slice:
  python scripts/build_features.py --config configs/local.yaml \
      --split train --limit-rows 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import (local_artifact_path, mark_stage, publish_artifact,
                           read_json, stage_complete, write_json)
from src.blocking import _list_parts, code_fingerprint, normalized_dir
from src.candidate_generation import candidates_dir
from src.config import config_hash, load_config
from src.data_loader import source_available
from src.ground_truth import load_ground_truth
from src.logging_utils import setup_logging, stage_timer
from src.pair_features import (RETRIEVAL_COLUMNS, build_side_table,
                               compute_part_features, label_for_keys)

_FE_STATE: Dict[str, Any] = {}

DIAG_FEATURES = ("name_sort", "name_set", "addr_sort", "country_match")


def _process_part(pi: int) -> Dict[str, Any]:
    st = _FE_STATE
    t = pq.read_table(st["cand_parts"][pi])
    s1_idx = t.column("s1_idx").to_numpy()
    pool_idx = t.column("pool_idx").to_numpy()
    bits = t.column("channel_bits").to_numpy()
    retrieval = {c: t.column(c).to_numpy() for c in RETRIEVAL_COLUMNS}
    feats = compute_part_features(s1_idx, pool_idx, bits, retrieval,
                                  st["s1_side"], st["pool_side"])
    cols: Dict[str, Any] = {"s1_idx": s1_idx, "pool_idx": pool_idx}
    n_pos = 0
    label = None
    if st["gt_keys"] is not None:
        keys = s1_idx.astype(np.int64) * st["n_pool"] + pool_idx.astype(np.int64)
        label = label_for_keys(keys, st["gt_keys"])
        cols["label"] = label
        n_pos = int(label.sum())
    cols.update(feats)
    name = f"part_{pi:04d}.parquet"
    pq.write_table(pa.table({k: pa.array(v) for k, v in cols.items()}),
                   st["out_dir"] / name, compression="snappy")
    diag: Dict[str, List[float]] = {c: [0.0, 0.0] for c in DIAG_FEATURES}
    if label is not None and label.size:
        lab = label.astype(bool)
        for c in DIAG_FEATURES:
            diag[c] = [float(feats[c][lab].sum()), float(feats[c][~lab].sum())]
    return {"pi": pi, "rows": int(s1_idx.size), "positives": n_pos,
            "file": name, "neg": int(s1_idx.size - n_pos), "diag": diag}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build pair features (Phase 7).")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    dry = args.limit_rows is not None
    cand_base = candidates_dir(cfg, args.split, dry)
    parts_path = cand_base / "parts.json"
    if not parts_path.exists():
        log.error("no candidates at %s - run generate_candidates first", cand_base)
        return 2
    cand_parts = [cand_base / f for f in read_json(parts_path)]
    chunks = read_json(cand_base / "chunks.json") or {}
    cand_run_key = chunks.get("run_key", "unknown") if isinstance(chunks, dict) \
        else "unknown"

    here = Path(__file__).resolve().parents[1]
    key = hashlib.sha1(json.dumps({
        "code": code_fingerprint([here / "src" / "pair_features.py",
                                  here / "src" / "blocking.py",
                                  here / "src" / "normalization.py",
                                  here / "src" / "config.py"]),
        "cfg": config_hash(cfg, "candidates"),
        "cand": cand_run_key, "split": args.split,
        "limit": args.limit_rows}, sort_keys=True).encode()).hexdigest()[:12]

    stage = f"features_{args.split}"
    markable = args.limit_rows is None
    if not args.force and markable:
        if stage_complete(cfg, stage, expected_hash=key) is not None:
            log.info("features_%s already complete", args.split)
            return 0

    rel = f"features/{args.split}_dryrun" if dry else f"features/{args.split}"
    out_dir = local_artifact_path(cfg, rel)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_out = out_dir / "parts.json"
    entries: List[Dict[str, Any]] = []
    if parts_out.exists():
        stored = read_json(parts_out) or {}
        if stored.get("key") == key:
            entries = stored.get("entries", [])
        else:
            log.warning("features state discarded (inputs changed)")
            for f in out_dir.glob("part_*.parquet"):
                f.unlink()
    known = {e.get("file") for e in entries if e.get("file")}
    for f in out_dir.glob("part_*.parquet"):
        if f.name not in known:
            f.unlink()
    done_pi = {int(e["pi"]) for e in entries}

    s1_paths = _list_parts(normalized_dir(cfg, args.split, "source1"))
    pool_paths = (_list_parts(normalized_dir(cfg, args.split, "source2")) +
                  _list_parts(normalized_dir(cfg, args.split, "source3")))
    log.info("building side tables (pool=%d rows)...",
             sum(pq.ParquetFile(x).metadata.num_rows for x in pool_paths))
    t0 = time.time()
    s1_side = build_side_table(s1_paths, limit_rows=args.limit_rows)
    pool_side = build_side_table(pool_paths)
    log.info("side tables built in %.0fs", time.time() - t0)

    n_pool = len(pool_side["name_alnum"])
    gt_keys = None
    gt_file = getattr(getattr(cfg.data, args.split), "ground_truth", "")
    if gt_file and source_available(cfg, args.split, "ground_truth"):
        gt = load_ground_truth(cfg, args.split)
        s1_ids = pq.read_table(cand_base / "s1_ids.parquet").column(0).to_pylist()
        pool_ids = pq.read_table(cand_base / "pool_ids.parquet").column(0).to_pylist()
        s1_pos = {e: i for i, e in enumerate(s1_ids)}
        pool_pos = {e: i for i, e in enumerate(pool_ids)}
        keys = [s1_pos[s] * n_pool + pool_pos[t]
                for s, toks in gt.matches.items() if s in s1_pos
                for t in toks if t in pool_pos]
        gt_keys = np.unique(np.asarray(keys, dtype=np.int64))
        log.info("labels: %d true pairs", gt_keys.size)

    _FE_STATE.update({"cand_parts": cand_parts, "s1_side": s1_side,
                      "pool_side": pool_side, "gt_keys": gt_keys,
                      "n_pool": n_pool, "out_dir": out_dir})
    pending = [pi for pi in range(len(cand_parts)) if pi not in done_pi]
    n_jobs = max(1, int(args.n_jobs or getattr(cfg.execution, "n_jobs", 1)))
    can_fork = "fork" in multiprocessing.get_all_start_methods() and n_jobs > 1

    timer = stage_timer(f"features[{args.split}]", cfg)
    timer.start()
    total_rows = sum(int(e.get("rows", 0)) for e in entries)
    total_pos = sum(int(e.get("positives", 0)) for e in entries)
    diag_sum = {c: [0.0, 0.0] for c in DIAG_FEATURES}
    pos_n = neg_n = 0

    def _record(res: Dict[str, Any]) -> None:
        nonlocal total_rows, total_pos, pos_n, neg_n
        entries.append(res)
        total_rows += res["rows"]
        total_pos += res["positives"]
        pos_n += res["positives"]
        neg_n += res["neg"]
        for c in DIAG_FEATURES:
            diag_sum[c][0] += res["diag"][c][0]
            diag_sum[c][1] += res["diag"][c][1]
        write_json(parts_out, {"key": key, "entries": entries})
        if not dry:
            publish_artifact(cfg, f"{rel}/{res['file']}")
            publish_artifact(cfg, f"{rel}/parts.json")

    if can_fork:
        try:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=n_jobs) as pool:
                for res in pool.imap_unordered(_process_part, pending):
                    _record(res)
        except Exception as exc:
            log.warning("parallel features failed (%s); serial fallback", exc)
    finished = {int(e["pi"]) for e in entries}
    for pi in [x for x in pending if x not in finished]:
        _record(_process_part(pi))
    timer.stop()
    _FE_STATE.clear()

    if markable:
        mark_stage(cfg, stage,
                   details={"split": args.split, "rows": total_rows,
                            "positives": total_pos, "parts": len(entries)},
                   artifacts=[f"{rel}/{e['file']}" for e in entries
                              if e.get("file")] + [f"{rel}/parts.json"],
                   config_hash=key, duration_s=timer.seconds,
                   peak_rss_mb=timer.peak_rss_mb)

    print("\n=== FEATURES SUMMARY ===")
    print(f"  rows: {total_rows} | positives: {total_pos} "
          f"({100 * total_pos / max(1, total_rows):.2f}%) | parts: {len(entries)}")
    if pos_n:
        print(f"  {'feature':<16}{'pos_mean':>10}{'neg_mean':>10}")
        for c in DIAG_FEATURES:
            print(f"  {c:<16}{diag_sum[c][0] / max(1, pos_n):>10.1f}"
                  f"{diag_sum[c][1] / max(1, neg_n):>10.1f}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
