"""Shared fixtures: a tiny synthetic dataset exercising every forensic path."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402

S1_HEADER = ["entity_id", "business_name", "business_address", "country"]
S1_ROWS = [
    ["S1-101", "Acme Corporation", "123 Main Street, Springfield", "US"],
    [],  # blank line
    ["S1-102", " Zenith Traders ", "45 River Road, Riverton", "India"],
    ["S1-103", "Blue Ocean Foods", "9 Beach Avenue, Coastville", "US"],
    ["S1-104", "Missing Data Co", "", "N/A"],
]
S2_HEADER = ["entity_id", "business_name", "business_address", "country"]
S2_ROWS = [
    ["S2-201", "Acme Corporation", "123 Main Street, Springfield", "US"],
    ["S2-202", "Acme Corp", "123 Main St, Springfield", "US"],
    ["S2-203", "Blue Ocean Foods", "9 Beach Avenue, Coastville", "US"],
    ["S2-204", "Zephyr Retail", "77 Lake Drive, Lakeside", "Canada", "EXTRA_FIELD"],
]
S3_HEADER = ["entity_id", "business_name", "business_address", "country"]
S3_ROWS = [
    ["S3-301", "ACME CORPORATION", "123 Main Street, Springfield", "US"],
    ["S3-303", "Blue Ocean Foods Pvt", "9 Beach Ave, Coastville", "US"],
    ["S2-204", "Duplicate Id Foods", "1 Collision Lane, Nowhere", "Canada"],
    ["S3-304"],  # short row (padded)
]
GT_HEADER = ["source1_entity_id", "matched_entity_ids"]
GT_ROWS = [
    ["S1-101", "S2-201,S3-301"],
    ["S1-103", "S2-203,S3-303,S2-204"],
    ["S1-102", "NA"],      # explicit zero-match marker
    ["S1-101", "S2-201"],    # duplicate pair
    ["S1-103", "S2-999"],    # invalid label (id not in any pool)
]


S1_IDS = {"S1-101", "S1-102", "S1-103", "S1-104"}
S2_IDS = {"S2-201", "S2-202", "S2-203", "S2-204"}
S3_IDS = {"S3-301", "S3-303", "S3-304", "S2-204"}


def _write_tsv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        if header:
            w.writerow(header)
        for r in rows:
            if r:
                w.writerow(r)
            else:
                fh.write("\n")


def make_config(tmp_path):
    cfg = Config()
    cfg.paths.data_dir = str(tmp_path / "dataset")
    cfg.paths.artifact_dir = str(tmp_path / "artifacts")
    cfg.paths.output_dir = str(tmp_path / "output")
    cfg.aws.use_s3 = False
    cfg.audit.pattern_sample_rows = 1000
    cfg.audit.top_k = 5
    cfg.audit.unique_cap = 500
    cfg.audit.save_head_rows = 10
    cfg.validation.n_folds = 2
    return cfg


@pytest.fixture
def synth(tmp_path):
    cfg = make_config(tmp_path)
    data = Path(cfg.paths.data_dir) / "train"
    _write_tsv(data / "train_source1.tsv", S1_HEADER, S1_ROWS)
    _write_tsv(data / "train_source2.tsv", S2_HEADER, S2_ROWS)
    _write_tsv(data / "train_source3.tsv", S3_HEADER, S3_ROWS)
    _write_tsv(data / "train_ground_truth.tsv", GT_HEADER, GT_ROWS)
    facts = {
        "row_counts": {"source1": 4, "source2": 4, "source3": 4},
        "id_cols": {"source1": "entity_id", "source2": "entity_id",
                    "source3": "entity_id"},
        "gt": {
            "entities": 3, "dup_s1_rows": 2, "dup_pairs": 1,
            "zero_marker_rows": 1, "absent": 1, "invalid": 1, "ambiguous": 1,
            "card": {"0": 1, "1": 0, "2": 1, "3_plus": 1, "max": 3,
                     "mean": 1.667},
            "any": 2, "s2": 2, "s3": 2, "both": 2, "s1_total": 4,
        },
        "overlap": 1,
    }
    return cfg, facts
