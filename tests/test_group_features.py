import numpy as np

from src.group_features import (GROUP_COLUMNS, compute_group_features,
                                topk_per_entity)


def test_topk_groups_and_sorts():
    s1 = np.array([1, 0, 1, 0, 1], np.int32)
    pr = np.array([0.2, 0.9, 0.8, 0.1, 0.5], np.float32)
    k = topk_per_entity(s1, pr, 2)
    assert s1[k].tolist() == [0, 0, 1, 1]
    assert pr[k].tolist() == [np.float32(0.9), np.float32(0.1),
                              np.float32(0.8), np.float32(0.5)]


def test_garbage_name_linked_through_anchor_address():
    side = {
        "addr_core": ["187 p 13 bajrang dham kota", "187 p 13 bajrang dham kota",
                      "9 other road jaipur"],
        "nums": [frozenset({"187", "13"}), frozenset({"187", "13"}), frozenset({"9"})],
        "name_skel": ["kb mls prbt ltd", "dbkr", "kb mls prbt ltd"],
    }
    # entity 0: confident anchor (pool 0), garbage-name twin (pool 1), decoy (pool 2)
    s1 = np.array([0, 0, 0], np.int32)
    pool = np.array([0, 1, 2], np.int32)
    pr = np.array([0.95, 0.2, 0.15], np.float32)
    g = compute_group_features(s1, pool, pr, side)
    assert set(g) == set(GROUP_COLUMNS)
    assert g["g_addr_eq"].tolist() == [0.0, 1.0, 0.0]   # anchor excludes itself
    assert g["g_num_eq"][1] == 1.0 and g["g_num_eq"][2] == 0.0
    assert g["g_name_skel_max"][2] == 100.0             # decoy copies the name
    assert g["g_rank"].tolist() == [1.0, 2.0, 3.0]
    assert g["g_n_anchor"].tolist() == [0.0, 1.0, 1.0]
    assert abs(g["g_p_max_other"][0] - 0.2) < 1e-6
