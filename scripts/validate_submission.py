#!/usr/bin/env python
"""Run the OFFICIAL submission validator with paths resolved from config.

utils/validate_submission.py is executed verbatim via subprocess; it is never
re-implemented here.

Examples:
  python scripts/validate_submission.py --config configs/local.yaml
  python scripts/validate_submission.py --config configs/aws_cpu.yaml --check-ids
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_loader import ensure_source
from src.logging_utils import setup_logging


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the official submission validator.")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--matching", default=None,
                   help="default: <output_dir>/matching_results.tsv")
    p.add_argument("--candidate", default=None,
                   help="default: <output_dir>/candidate_pairs.tsv")
    p.add_argument("--no-candidate", action="store_true")
    p.add_argument("--test-dir", default=None, help="default: <data_dir>/test")
    p.add_argument("--check-ids", action="store_true",
                   help="enable the validator's memory-heavy ID-existence check")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log = setup_logging(cfg)

    root = Path(__file__).resolve().parents[1]
    validator = root / "utils" / "validate_submission.py"
    if not validator.exists():
        log.error("official validator not found at %s - save the "
                  "competition-provided file there, unmodified", validator)
        return 2

    sources = (1, 2, 3) if args.check_ids else (1,)
    for src in sources:
        try:
            ensure_source(cfg, "test", src)
        except Exception as exc:
            log.debug("test source%d not locally available (%s)", src, exc)

    matching = (Path(args.matching) if args.matching
                else Path(cfg.paths.output_dir) / "matching_results.tsv")
    test_dir = (Path(args.test_dir) if args.test_dir
                else Path(cfg.paths.data_dir) / "test")
    cmd = [sys.executable, str(validator), "--matching", str(matching),
           "--test-dir", str(test_dir)]
    if not args.no_candidate:
        candidate = (Path(args.candidate) if args.candidate
                     else Path(cfg.paths.output_dir) / "candidate_pairs.tsv")
        cmd += ["--candidate", str(candidate)]
    if args.check_ids:
        cmd += ["--check-ids"]
    log.info("running official validator: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        log.error("official validator FAILED (exit %d)", result.returncode)
    else:
        log.info("official validator PASS")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
