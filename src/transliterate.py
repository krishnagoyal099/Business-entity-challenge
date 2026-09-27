"""Rule-based Indic -> Latin romanization + consonant skeletons (stdlib only).

The major Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
Tamil, Telugu, Kannada, Malayalam) share one ISCII-derived layout: the same
letter sits at the same offset inside each 128-codepoint block. One table keyed
by offset therefore romanizes all nine scripts.

`skeleton` maps both romanized Indic and Latin spellings onto a coarse
consonant key ("इंफोटेक" -> "infotek" -> "nftk" == skeleton("Infotech")),
so cross-script names and typo'd names become comparable by fuzzy ratio.
"""
from __future__ import annotations

import re
from typing import Dict

_BLOCK_STARTS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00,
                 0x0C80, 0x0D00)

_VOWELS: Dict[int, str] = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "rii", 0x61: "lii",
}
_CONSONANTS: Dict[int, str] = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch",
    0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th",
    0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
    0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h", 0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_SIGNS: Dict[int, str] = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri",
    0x44: "rii", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "lii",
}
_NASAL = {0x01: "n", 0x02: "n", 0x70: "n"}          # candrabindu, anusvara, tippi
_VIRAMA = 0x4D
_SILENT = {0x03, 0x3C, 0x71, 0x51, 0x52}             # visarga, nukta, addak, accents
# Malayalam chillu letters and Bengali khanda-ta are standalone final consonants
# (and Oriya wa, which sits outside the shared layout)
_EXTRA = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l",
          0x0D7F: "k", 0x09CE: "t", 0x0B71: "v"}


def _offset(ch: str):
    cp = ord(ch)
    if 0x0900 <= cp < 0x0D80:
        for start in _BLOCK_STARTS:
            if start <= cp < start + 0x80:
                return cp - start
    return None


def romanize(s: str) -> str:
    """Romanize Indic letters; everything else passes through unchanged."""
    if not s or s.isascii():
        return s or ""
    s = s.replace("റ്റ", "ട്ട")  # Malayalam rra-rra = tta
    out = []
    pending_a = False                      # inherent vowel of last consonant
    for ch in s:
        cp = ord(ch)
        if cp in _EXTRA:
            if pending_a:
                out.append("a")
            out.append(_EXTRA[cp])
            pending_a = False
            continue
        off = _offset(ch)
        if off is None:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(ch)
            continue
        if off in _CONSONANTS:
            if pending_a:
                out.append("a")
            out.append(_CONSONANTS[off])
            pending_a = True
        elif off in _SIGNS:
            out.append(_SIGNS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off in _NASAL:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_NASAL[off])
        elif off in _VOWELS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_VOWELS[off])
        elif 0x66 <= off <= 0x6F:          # native digits
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(str(off - 0x66))
        elif off in _SILENT:
            continue
        else:
            if pending_a:
                out.append("a")
                pending_a = False
    if pending_a:
        out.append("a")
    return "".join(out)


_REPEAT_RE = re.compile(r"(.)\1+")
_SKEL_DROP = str.maketrans("", "", "aeiouyhw")


def skeleton_token(tok: str) -> str:
    t = tok.replace("ph", "f").replace("x", "ks").replace("w", "v")
    t = t.replace("c", "k").replace("q", "k").replace("z", "j").replace("v", "b")
    t = t.replace("nb", "mb").replace("np", "mp")
    t = t.translate(_SKEL_DROP)
    return _REPEAT_RE.sub(r"\1", t)


def skeleton(s: str) -> str:
    """Space-joined consonant keys of the romanized, lowercased tokens."""
    r = romanize(s).lower()
    toks = (skeleton_token(t) for t in re.split(r"[^a-z0-9]+", r) if t)
    return " ".join(t for t in toks if t)


def has_non_latin_letters(s: str) -> bool:
    return (not s.isascii()) and any(_offset(ch) is not None for ch in s)
