import subprocess
import sys
from pathlib import Path

import pytest

from src.submission import (check_matches_subset_of_candidates,
                            read_id_list_file, write_candidate_pairs,
                            write_matching_results)

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "utils" / "validate_submission.py"
REQUIRED = ["S1-1", "S1-2", "S1-3", "S1-4"]
NO_VALIDATOR = "official validator not placed at utils/validate_submission.py"


def _make_test_dir(tmp_path):
    d = tmp_path / "dataset_test"
    d.mkdir(parents=True)

    def w(name, rows):
        with open(d / name, "w", encoding="utf-8", newline="") as fh:
            fh.write("entity_id\tbusiness_name\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")

    w("test_source1.tsv", [[f"S1-{i}", f"Business {i}"] for i in (1, 2, 3, 4)])
    w("test_source2.tsv", [[f"S2-{i}", f"Pool2 {i}"] for i in (201, 202, 203, 204)])
    w("test_source3.tsv", [[f"S3-{i}", f"Pool3 {i}"] for i in (301, 302, 303)])
    return d


def test_writer_format_and_roundtrip(tmp_path):
    out = tmp_path / "output"
    preds = {"S1-1": {"S2-201", "S3-301"}, "S1-2": set(),
             "S1-3": {"S2-203"}, "S1-4": {"S3-302"}}
    stats = write_matching_results(preds, out / "matching_results.tsv",
                                   required_ids=REQUIRED)
    assert stats["rows"] == 4 and stats["empty_rows"] == 1
    lines = (out / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert "S1-2\t" in lines
    assert "S1-1\tS2-201,S3-301" in lines

    write_candidate_pairs({"S1-1": {"S2-201", "S3-301", "S2-202"}, "S1-2": set(),
                           "S1-3": {"S2-203"}, "S1-4": {"S3-302"}},
                          out / "candidate_pairs.tsv", required_ids=REQUIRED)
    header = (out / "candidate_pairs.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "source1_entity_id\tcandidate_entity_ids"
    back = read_id_list_file(out / "matching_results.tsv")
    assert back["S1-1"] == {"S2-201", "S3-301"} and back["S1-2"] == set()


def test_our_output_passes_official_validator(tmp_path):
    if not VALIDATOR.exists():
        pytest.skip(NO_VALIDATOR)
    out = tmp_path / "output"
    write_matching_results({"S1-1": {"S2-201", "S3-301"}, "S1-2": set(),
                            "S1-3": {"S2-203"}, "S1-4": {"S3-302"}},
                           out / "matching_results.tsv", required_ids=REQUIRED)
    write_candidate_pairs({"S1-1": {"S2-201", "S3-301", "S2-202"}, "S1-2": set(),
                           "S1-3": {"S2-203"}, "S1-4": {"S3-302"}},
                          out / "candidate_pairs.tsv", required_ids=REQUIRED)
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
         "--candidate", str(out / "candidate_pairs.tsv"),
         "--test-dir", str(_make_test_dir(tmp_path)), "--check-ids"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


def test_official_validator_rejects_bad_files(tmp_path):
    if not VALIDATOR.exists():
        pytest.skip(NO_VALIDATOR)
    bad = tmp_path / "matching_results.tsv"
    bad.write_text("source1_entity_id\tmatched_entity_ids\n"
                   "S1-1\tS2-201\nS1-1\tS2-202\n", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(VALIDATOR), "--matching", str(bad),
         "--test-dir", str(_make_test_dir(tmp_path))],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert r.returncode == 1
    assert "duplicate" in r.stdout.lower()


def test_writer_fills_missing_entities_loudly(tmp_path):
    out = tmp_path / "output"
    stats = write_matching_results({"S1-1": {"S2-201"}},
                                   out / "matching_results.tsv", required_ids=REQUIRED)
    assert stats["filled_missing_rows"] == 3
    lines = (out / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and "S1-2\t" in lines and "S1-4\t" in lines


def test_writer_rejects_invalid_content(tmp_path):
    out = tmp_path / "output"
    with pytest.raises(ValueError):
        write_matching_results({"S1-1": {"S1-2"}}, out / "m.tsv")
    with pytest.raises(ValueError):
        write_matching_results({"S1-1": {"XX-9"}}, out / "m.tsv")
    with pytest.raises(ValueError):
        write_matching_results({"S1-1": {"S2-20\t1"}}, out / "m.tsv")
    with pytest.raises(ValueError):
        write_matching_results({"S1-1": {"S2-201"}}, out / "m.tsv",
                               required_ids=["S1-9"])


def test_writer_strips_and_dedupes(tmp_path):
    out = tmp_path / "output"
    write_matching_results({"S1-1": [" S2-201 ", "S2-201", " S3-301"]}, out / "m.tsv")
    assert (out / "m.tsv").read_text(encoding="utf-8").splitlines()[1] == \
        "S1-1\tS2-201,S3-301"


def test_candidate_cap(tmp_path):
    out = tmp_path / "output"
    stats = write_candidate_pairs({"S1-1": {"S2-1", "S2-2", "S2-3"}},
                                  out / "c.tsv", max_ids_per_row=2)
    assert stats["max_ids_per_row_observed"] == 2


def test_subset_check():
    preds = {"S1-1": {"S2-201", "S2-999"}, "S1-2": {"S3-301"}}
    cands = {"S1-1": {"S2-201"}, "S1-2": {"S3-301"}}
    assert check_matches_subset_of_candidates(preds, cands) == ["S1-1"]
