#!/usr/bin/env python
"""Phase 2 stage entrypoint: data audit.

Examples:
  python scripts/audit_data.py --config configs/local.yaml
  python scripts/audit_data.py --config configs/aws_cpu.yaml --upload
  python scripts/audit_data.py --config configs/local.yaml --split test
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aws_utils import local_artifact_path, mark_stage, stage_complete
from src.config import config_hash, load_config
from src.data_audit import run_audit, write_profile
from src.data_loader import local_source_path
from src.logging_utils import setup_logging, stage_timer


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Audit the entity-resolution source data (Phase 2).")
    p.add_argument("--config", required=True,
                   help="Path or s3:// URI of a YAML config")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--force", action="store_true",
                   help="Rerun even if the audit stage manifest matches")
    p.add_argument("--upload", action="store_true",
                   help="Force upload of artifacts to S3 (requires S3_BUCKET)")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip S3 upload even if configured")
    return p.parse_args(argv)


def input_fingerprint(cfg, split: str) -> str:
    """Size + mtime of each local input file, so changed data invalidates the stage."""
    parts = []
    for src in (1, 2, 3, "ground_truth"):
        try:
            p = local_source_path(cfg, split, src)
        except ValueError:
            continue
        if p.exists():
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return ";".join(parts)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.upload:
        cfg.aws.use_s3 = True
    if args.no_upload:
        cfg.aws.use_s3 = False
    log = setup_logging(cfg)

    stage = "audit" if args.split == "train" else f"audit_{args.split}"
    art_rel = "data_audit" if args.split == "train" else f"data_audit_{args.split}"
    chash = hashlib.sha1(
        (config_hash(cfg, "audit") + "|" + args.split + "|"
         + input_fingerprint(cfg, args.split)).encode("utf-8")).hexdigest()[:12]

    if not args.force:
        manifest = stage_complete(cfg, stage, expected_hash=chash)
        if manifest is not None:
            log.info("audit stage already complete (hash=%s); artifacts: %s "
                     "— use --force to rerun", chash, manifest.get("artifacts"))
            return 0

    timer = stage_timer(f"audit[{args.split}]", cfg)
    timer.start()
    report = run_audit(cfg, split=args.split)
    timer.stop()

    out_dir = local_artifact_path(cfg, art_rel)
    written = write_profile(report, out_dir)
    rels = [f"{art_rel}/{name}" for name in written]

    details = {
        "split": args.split,
        "sources_profiled": sorted(report["sources"].keys()),
        "row_counts": {k: v["row_count"] for k, v in report["sources"].items()},
        "id_columns": {k: v["column"] for k, v in report["id_columns"].items()},
        "ground_truth_status": report["ground_truth"].get("status"),
        "cardinality": report["ground_truth"].get("cardinality"),
        "warnings": report["warnings"],
    }
    mark_stage(cfg, stage, details=details, artifacts=rels, config_hash=chash,
               duration_s=timer.seconds, peak_rss_mb=timer.peak_rss_mb)
    log.info("audit report written to %s", out_dir)
    _print_summary(report, out_dir)
    return 0


def _print_summary(report, out_dir) -> None:
    print("\n=== AUDIT SUMMARY ===")
    for k, v in report["sources"].items():
        print(f"  {k}: rows={v['row_count']} cols={v['column_count']} "
              f"id={report['id_columns'].get(k, {}).get('column', '?')}")
    gt = report["ground_truth"]
    if gt.get("status") == "parsed":
        mr = gt["match_rate"]
        print(f"  GT: entities={gt['entities_in_gt']} cardinality={gt['cardinality']}")
        print(f"  match rate (of |S1|): any={mr['any']['frac_of_s1_file']} "
              f"s2={mr['s2']['frac_of_s1_file']} s3={mr['s3']['frac_of_s1_file']}")
    else:
        print(f"  GT: {gt.get('status')} ({gt.get('reason', '')})")
    if report["warnings"]:
        print(f"  warnings: {len(report['warnings'])}")
    print(f"  full report: {Path(out_dir) / 'profile.txt'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
