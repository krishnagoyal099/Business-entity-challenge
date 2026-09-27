import numpy as np

from src import pair_features as pf
from src.transliterate import romanize, skeleton


def test_romanize_passes_latin_through():
    assert romanize("Best Infotech") == "Best Infotech"
    assert romanize("") == ""


def test_cross_script_names_share_skeleton():
    assert skeleton("बेस्ट इंफोटेक लिमिटेड") == skeleton("Best Infotech Limited")
    assert skeleton("राम मार्केटिंग") == skeleton("Ram Marketing")
    assert skeleton("ହ୍ୱାଇଟ୍ ଫୁଡ୍ସ୍") == skeleton("White Foods")


def _side(names, addrs):
    n = len(names)
    return {
        "name_core": names, "name_sorted": [" ".join(sorted(x.split())) for x in names],
        "name_alnum": names, "addr_core": addrs, "country": ["india"] * n,
        "house": [""] * n, "digits": [""] * n, "states": [frozenset()] * n,
        "nums": [frozenset(t.lstrip("0") or "0" for t in a.split() if t.isdigit())
                 for a in addrs],
        "alphas": [frozenset(t for t in a.split() if t.isalpha() and len(t) > 1)
                   for a in addrs],
        "addr_skel": [skeleton(a) for a in addrs],
        "name_skel": [skeleton(x) for x in names],
        "script": [not x.isascii() for x in names],
        "glued": [x.replace(" ", "") for x in names],
    }


def test_extra_features_separate_true_pair_from_number_decoy():
    s1 = _side(["shivam energy"], ["flat 6052 5 sector d vasant kunj delhi"])
    pool = _side(["शिवम एनर्जी", "shivamenergy"],
                 ["flat 6052 5 sector d vasant kunj delhi",
                  "flat 6052 6 sector d vasant kunj delhi"])
    n = 2
    ret = {c: np.zeros(n, np.float32) for c in pf.RETRIEVAL_COLUMNS}
    out = pf.compute_part_features(np.zeros(n, np.int32), np.arange(n, dtype=np.int32),
                                   np.zeros(n, np.uint8), ret, s1, pool)
    assert set(pf.FEATURE_COLUMNS) <= set(out)
    assert out["name_skel_set"][0] >= 80.0           # cross-script (soft g ~ j)
    assert out["name_script_mix"][0] == 1.0
    assert out["name_glued_ratio"][1] == 100.0       # glued/url-style name
    assert out["addr_num_equal"][0] == 1.0
    assert out["addr_num_equal"][1] == 0.0           # decoy: 6052/6 vs 6052/5
    assert out["addr_num_jacc"][1] < 1.0
