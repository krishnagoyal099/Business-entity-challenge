import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from src.blocking import ensure_normalized
from src.candidate_generation import run_retrieval
from src.pair_features import (FEATURE_COLUMNS, RETRIEVAL_COLUMNS,
                               build_side_table, compute_part_features,
                               label_for_keys)

TINY_CHANNELS = {
    "exact": {"enabled": True, "k": 10, "max_postings": 10},
    "char_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "word_name": {"enabled": True, "k": 3, "max_df": 1.0},
    "rare": {"enabled": True, "k": 3, "max_df_abs": 100},
    "word_addr": {"enabled": True, "k": 3, "max_df": 1.0},
}


def _sides(synth):
    cfg, _ = synth
    norm = ensure_normalized(cfg, "train")
    return (build_side_table(norm["source1"]),
            build_side_table(norm["source2"] + norm["source3"]))


def _retrieval_zeros(n):
    return {c: np.zeros(n, dtype=np.float32) for c in RETRIEVAL_COLUMNS}


def test_side_table_derivations(synth):
    s1, _ = _sides(synth)
    assert s1["house"][0] == "123"
    assert s1["digits"][0] == "123"
    assert s1["country"][0] == "us"
    assert s1["states"][0] == frozenset()
    assert s1["house"][3] == ""               # S1-104: missing address


def test_features_exact_pair(synth):
    s1, pool = _sides(synth)
    f = compute_part_features(np.array([0], np.int32), np.array([0], np.int32),
                              np.array([3], np.uint8), _retrieval_zeros(1),
                              s1, pool)
    assert f["name_exact"][0] == 1.0 and f["name_core_exact"][0] == 1.0
    assert f["name_sort"][0] == 100.0 and f["addr_sort"][0] == 100.0
    assert f["house_match"][0] == 1.0 and f["country_match"][0] == 1.0
    assert f["n_channels"][0] == 2.0          # popcount(3)


def test_features_mismatch_pair(synth):
    s1, pool = _sides(synth)
    f = compute_part_features(np.array([0], np.int32), np.array([3], np.int32),
                              np.array([1], np.uint8), _retrieval_zeros(1),
                              s1, pool)                   # S2-204: other business
    assert f["name_exact"][0] == 0.0
    assert f["name_sort"][0] < 60.0 and f["addr_sort"][0] < 60.0
    assert f["country_known"][0] == 1.0 and f["country_match"][0] == 0.0  # Canada


def test_missing_address_is_not_perfect_match(synth):
    s1, pool = _sides(synth)
    f = compute_part_features(np.array([3], np.int32), np.array([0], np.int32),
                              np.array([1], np.uint8), _retrieval_zeros(1),
                              s1, pool)                   # S1-104 has no address
    assert f["addr_known"][0] == 0.0
    assert f["addr_sort"][0] == 0.0           # NOT 100: the empty-string trap
    assert f["house_known"][0] == 0.0


def test_label_for_keys():
    gt = np.array([10, 30, 50], dtype=np.int64)
    keys = np.array([10, 11, 30, 99], dtype=np.int64)
    assert list(label_for_keys(keys, gt)) == [1, 0, 1, 0]
    assert list(label_for_keys(keys, np.empty(0, np.int64))) == [0, 0, 0, 0]


def test_feature_columns_complete(synth):
    s1, pool = _sides(synth)
    f = compute_part_features(np.array([0], np.int32), np.array([0], np.int32),
                              np.array([1], np.uint8), _retrieval_zeros(1),
                              s1, pool)
    assert set(f.keys()) == set(FEATURE_COLUMNS)


def test_build_features_script_end_to_end(synth, tmp_path):
    cfg, _ = synth
    cfg.retrieval.channels = {k: dict(v) for k, v in TINY_CHANNELS.items()}
    cfg.retrieval.normalize_part_rows = 3
    cfg.retrieval.s1_chunk_rows = 2
    cfg.execution.n_jobs = 1
    run_retrieval(cfg, "train")
    conf = tmp_path / "cfg.yaml"
    conf.write_text(yaml.safe_dump({
        "paths": {"data_dir": cfg.paths.data_dir,
                  "artifact_dir": cfg.paths.artifact_dir,
                  "output_dir": cfg.paths.output_dir},
        "retrieval": {"channels": TINY_CHANNELS,
                      "normalize_part_rows": 3, "s1_chunk_rows": 2},
    }), encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_features
    assert build_features.main(["--config", str(conf), "--split", "train",
                                "--n-jobs", "1"]) == 0
    feat_dir = Path(cfg.paths.artifact_dir) / "features" / "train"
    parts = sorted(feat_dir.glob("part_*.parquet"))
    assert parts
    t = pq.read_table(parts[0])
    for c in ("s1_idx", "pool_idx", "label") + FEATURE_COLUMNS:
        assert c in t.column_names
    total_pos = sum(int(pq.read_table(p, columns=["label"]).column(0)
                        .to_numpy().sum()) for p in parts)
    assert total_pos >= 4          # 5 mappable true pairs, recall >= 0.8


def test_address_states_ignore_french_function_words():
    from src.normalization import build_address_views
    from src.pair_features import address_states

    def st(a):
        return address_states(list(build_address_views(a).core))
    assert st("106 Rue de la Gaudiniere, Nantes, Pays de la Loire") == frozenset()
    assert st("N 212 RUE DE LA BENAUGE, BORDEAUX") == frozenset()
    assert st("12 Elm St, Dover, DE") == {"DE"}
    assert st("OH, Columbus, 5559 Orville Avenue") == {"OH"}
    assert st("5415 ARMOUR DRIVE, TN, SOMERVILLE") == {"TN"}
    assert st("1 Main St, La Crosse, WI") == {"WI"}
    assert st("Columbus OH 43215 Main") == {"OH"}
