#!/usr/bin/env python
"""Phase 5 stage entrypoint: candidate generation + retrieval report.

Cost discipline: ALWAYS run the dry-run first:
  python scripts/generate_candidates.py --config configs/local.yaml --limit-rows 20000
(measures runtime, peak RSS and early recall; writes candidates/<split>_dryrun/,
marks no stage). Full run:
  python scripts/generate_candidates.py --config configs/local.yaml
Channel subset: --channels exact,char_name,rare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import mark_stage, stage_complete
from src.blocking import code_fingerprint
from src.candidate_generation import candidates_dir, run_retrieval
from src.config import config_hash, load_config
from src.logging_utils import setup_logging


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate candidates (Phase 5).")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit-rows", type=int, default=None,
                   help="dry-run on the first N S1 entities (no stage marking)")
    p.add_argument("--channels", default=None, help="comma-separated channel subset")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="worker processes (default: cfg.execution.n_jobs)")
    p.add_argument("--max-spread", type=int, default=None,
                   help="TOTAL transient nnz budget across workers")
    args = p.parse_args(argv)
    args.channel_subset = args.channels.split(",") if args.channels else None
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg)
    src = Path(__file__).resolve().parents[1] / "src"
    key = hashlib.sha1(json.dumps({
        "cfg": config_hash(cfg, "candidates"),
        "code": code_fingerprint([src / n for n in (
            "normalization.py", "blocking.py", "candidate_generation.py",
            "candidate_metrics.py", "config.py")]),
        "split": args.split, "channels": args.channel_subset,
    }, sort_keys=True).encode()).hexdigest()[:12]

    stage = f"candidates_{args.split}"
    full_run = args.limit_rows is None and args.channel_subset is None
    if not args.force and full_run:
        done = stage_complete(cfg, stage, expected_hash=key)
        if done is not None:
            log.info("%s already complete (hash=%s); use --force to rerun", stage, key)
            return 0

    summary = run_retrieval(cfg, split=args.split, limit_rows=args.limit_rows,
                            dry_run=args.limit_rows is not None,
                            channel_subset=args.channel_subset,
                            n_jobs=args.n_jobs, max_spread=args.max_spread,
                            log=log)
    if full_run:
        base = candidates_dir(cfg, args.split)
        rels = [f"candidates/{args.split}/{f.name}"
                for f in sorted(base.glob("*")) if f.is_file()]
        mark_stage(cfg, stage,
                   details={k: summary.get(k) for k in (
                       "n_pool", "n_s1", "total_pairs", "channels", "pair_recall")},
                   artifacts=rels, config_hash=key,
                   duration_s=summary.get("elapsed_s"),
                   peak_rss_mb=summary.get("peak_rss_mb"))
    print("\n=== CANDIDATES SUMMARY ===")
    for k in ("n_pool", "n_s1", "total_pairs", "parts", "channels", "pair_recall",
              "avg_candidates_per_entity", "missed_pairs_total",
              "elapsed_s", "n_jobs", "peak_rss_mb"):
        print(f"  {k}: {summary.get(k)}")
    print(f"  report: {candidates_dir(cfg, args.split, bool(args.limit_rows))}"
          f"/retrieval_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
