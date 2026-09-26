"""Regression tests built from the REAL noise observed in the data samples.

Revision 2: single-letter core stripping (with initials fallback), Indic legal
stop words, Indic tokenization integrity, ordinal/floor state guards, repeated
URL prefix stripping.
"""
import pytest

from src.data_loader import load_source
from src.normalization import (AddressViews, NameViews, build_address_views,
                               build_country, build_name_views, build_views,
                               char_ngrams, fold_latin_diacritics,
                               normalize_address, normalize_name, soundex,
                               strip_legal_suffix, strip_legal_tokens)


def test_legal_affixes_both_sides_and_plus():
    assert build_name_views("B+ Retail Inc").core == ("retail",)
    assert build_name_views("LLC Moncada Léarning Center").core == \
        ("moncada", "learning", "center")
    assert build_name_views("Pvt. EFS Print Ventures Ltd.").core == ("efs", "print")


def test_pure_initials_fallback():
    v = build_name_views("A B C")
    assert v.core == ("a", "b", "c") and v.acronym == "abc"


def test_indic_legal_words_stripped():
    assert build_name_views("राम मार्केटिंग प्राइवेट लिमिटेड").core == \
        ("राम", "मार्केटिंग")
    assert build_name_views("आदित्य प्रॉपर्टीज एलएलपी").core == \
        ("आदित्य", "प्रॉपर्टीज")


def test_indic_tokenization_intact():
    v = build_name_views("आदित्य प्रॉपर्टीज एलएलपी")
    assert "आदित्य" in v.tokens and "प्रॉपर्टीज" in v.tokens
    assert all(len(t) > 1 for t in v.tokens)


def test_url_extraction_and_pure_domain_name():
    v = build_name_views(
        "SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | "
        "[www.shivshakti.com](https://www.shivshakti.com)")
    assert v.urls == ("shivshakti.com",)
    assert "www" not in v.alnum and "com" not in v.alnum
    assert v.core == ("shivshakti", "vidyalaya", "overseas")
    v2 = build_name_views("wilfordhancock.com")
    assert v2.urls == ("wilfordhancock.com",) and v2.core == ()
    assert build_name_views("https://www.example.com").urls == ("example.com",)


def test_build_urls_toggle():
    assert build_name_views("acme.com", build_urls=False).urls == ()


def test_leading_junk_and_dba():
    assert build_name_views("-- Holloway Peak Inc Seafood").core == \
        ("holloway", "peak", "seafood")
    assert build_name_views("Ectolumdrex dba X+ Madison Inc").core == \
        ("ectolumdrex", "madison")


def test_apostrophe_and_duplication():
    assert build_name_views("Moyna's Coffee").core == ("moynas", "coffee")


def test_address_component_order_invariance():
    a = build_address_views("OH, Columbus, 5559 Orville Avenue")
    b = build_address_views("5559 Orville Avenue, Columbus, OH")
    assert a.core_sorted == b.core_sorted and a.sorted == b.sorted


def test_address_house_numbers_corrupted():
    v = build_address_views("##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri")
    assert v.house == "8" and v.states == ("MO",)   # "Fl. 0" is not Florida
    assert build_address_views("19 1/2 STARDUST TRAIL").house == "19"
    v = build_address_views("S03575 Cty Tk M, Town Of Buffalo, WI")
    assert "WI" in v.states and v.house == "03575"


def test_real_florida_still_matches():
    assert "FL" in build_address_views("12 Bay St, Miami, FL 33101").states


def test_ordinals_do_not_fire_states():
    assert build_address_views("22Nd Main Street, Dundalk, MD").states == ("MD",)


def test_ordinals_and_states_indic():
    v = build_address_views(
        "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, "
        "Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ")
    assert "41" in v.tokens and "st" in v.tokens
    assert "KA" in v.states
    assert "183" in v.tokens and "door" not in v.core


def test_west_bengal_bigram_and_plot_numbers():
    v = build_address_views("797, Lake Town Block A, Kolkata, Howrah, West Bengal")
    assert "WB" in v.states and "797" in v.tokens
    assert "MP" in build_address_views(
        "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh").states


def test_missing_address():
    v = build_address_views("")
    assert v.core == () and v.house == "" and v.states == () and v.digits == ""


def test_country_open_set_and_missing():
    assert build_country("US") == "us"
    assert build_country("India") == "india"
    assert build_country("United States") == "us"
    assert build_country("  CANADA ") == "canada"
    assert build_country("") == "" and build_country("N/A") == ""


def test_soundex_known_pairs():
    assert soundex("Robert") == "R163" and soundex("Rupert") == "R163"
    assert soundex("") == ""


def test_diacritic_fold_keeps_indic():
    assert fold_latin_diacritics("Léarning") == "Learning"
    kn = "ಕರ್ನಾಟಕ"
    assert fold_latin_diacritics(kn) == kn


def test_char_ngrams_padded():
    assert char_ngrams("ab", 3) == [" ab", "ab "]
    assert char_ngrams("", 3) == []


def test_strip_legal_helpers():
    assert strip_legal_suffix("ltd") == ""
    assert strip_legal_suffix("LTD.") == ""
    assert strip_legal_suffix("consulting") == "consulting"
    assert strip_legal_tokens(["acme", "inc", "corp"]) == ["acme"]


def test_view_dispatch():
    assert normalize_name("B+ Retail Inc", "core") == ("retail",)
    a = normalize_address("5559 Orville Avenue, Columbus, OH", "core_sorted")
    b = normalize_address("OH, Columbus, 5559 Orville Avenue", "core_sorted")
    assert a == b
    with pytest.raises(ValueError):
        normalize_name("x", "bogus")


def test_pure_and_deterministic():
    v1 = build_name_views("Delta Tetlecommunication Inc")
    assert v1 == build_name_views("Delta Tetlecommunication Inc")
    assert isinstance(v1, NameViews)
    assert isinstance(build_address_views("x"), AddressViews)


def test_build_views_dataframe(synth):
    cfg, _ = synth
    out = build_views(load_source(cfg, "train", 1), cfg, source_key="source1")
    assert "name_core_sorted" in out.columns
    assert "addr_house" in out.columns and "country_norm" in out.columns
    row = out.iloc[0]
    assert row["name_core"] == ["acme"] and row["country_norm"] == "us"
    assert out.iloc[3]["addr_core_sorted"] == ""
