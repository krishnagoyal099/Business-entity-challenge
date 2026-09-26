"""Multi-view normalization (Phase 4).

Targets the noise confirmed by the audit + samples:

Names: leading junk, embedded/whole URLs (extracted to a separate view, never
discarded), legal affixes on both sides, dba/aka segments, token duplication,
case variance, apostrophes, "+" tokens, Latin diacritics (Indic scripts are
never mangled), non-Latin scripts preserved.

Addresses: shuffled component order (sorted-token views are order-invariant),
missing values, corrupted house numbers ("##8", "19 1/2"), unit/PO-box
decorators, glued ordinals ("41St"), plot formats ("63/2275/7"), state
full-name vs abbreviation vs Indic script, duplicated city/state.

Country: open-set; unknown values pass through.

All functions are pure and deterministic: no fitted state, no leakage surface.
Raw strings are always preserved upstream; views are additive.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

NAME_STOP_TOKENS = frozenset({
    "inc", "llc", "ltd", "limited", "plc", "pvt", "private", "llp", "lp",
    "corp", "corporation", "co", "company", "enterprises", "enterprise",
    "ventures", "venture", "group", "dba", "aka", "fka", "the", "and", "of",
    "ltda", "gmbh", "bv", "nv", "sa", "sas", "srl", "spa", "pte", "sdn",
    "bhd", "oy", "ab", "as", "aps", "og",
    # Indic legal words (Devanagari / Kannada)
    "लिमिटेड", "प्राइवेट", "एलएलपी", "एलएलसी", "कंपनी", "प्रा", "लि",
    "ಪ್ರೈವೇಟ್", "ಲಿಮಿಟೆಡ್", "ಕಂಪನಿ",
})

ADDR_STOP_TOKENS = frozenset({
    "unit", "apt", "apartment", "suite", "ste", "floor", "fl", "no", "nos",
    "number", "num", "po", "box", "pob", "door", "h", "hn", "bldg",
    "building", "blk", "of", "the",
})

_TLDS = frozenset({
    "com", "net", "org", "in", "co", "io", "biz", "info", "edu", "gov",
    "us", "uk", "ai", "app", "shop", "store", "online", "site", "xyz",
    "me", "tv", "cc", "pro", "firm", "ind",
})

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")

_US_FULL = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "wisconsin": "WI",
    "wyoming": "WY",
}
_US_BIGRAMS = {
    ("new", "hampshire"): "NH", ("new", "jersey"): "NJ",
    ("new", "mexico"): "NM", ("new", "york"): "NY",
    ("north", "carolina"): "NC", ("north", "dakota"): "ND",
    ("south", "carolina"): "SC", ("south", "dakota"): "SD",
    ("rhode", "island"): "RI", ("west", "virginia"): "WV",
    ("district", "columbia"): "DC",
}
_US_ABBR = {a: a.upper() for a in (
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc")}
_IN_SINGLE = {
    "karnataka": "KA", "ಕರ್ನಾಟಕ": "KA", "kerala": "KL", "assam": "AS",
    "bihar": "BR", "goa": "GA", "gujarat": "GJ", "haryana": "HR",
    "jharkhand": "JH", "chhattisgarh": "CG", "manipur": "MN",
    "meghalaya": "ML", "mizoram": "MZ", "nagaland": "NL", "odisha": "OR",
    "orissa": "OR", "punjab": "PB", "panjab": "PB", "rajasthan": "RJ",
    "sikkim": "SK", "tripura": "TR", "uttarakhand": "UK",
    "uttaranchal": "UK", "delhi": "DL", "दिल्ली": "DL",
    "pondicherry": "PY", "puducherry": "PY", "chandigarh": "CH",
    "maharashtra": "MH", "महाराष्ट्र": "MH",
}
_IN_BIGRAMS = {
    ("west", "bengal"): "WB", ("uttar", "pradesh"): "UP",
    ("madhya", "pradesh"): "MP", ("andhra", "pradesh"): "AP",
    ("arunachal", "pradesh"): "AR", ("tamil", "nadu"): "TN",
    ("himachal", "pradesh"): "HP", ("jammu", "kashmir"): "JK",
}

_STATE_SINGLE: Dict[str, str] = {**_US_FULL, **_US_ABBR, **_IN_SINGLE}
_STATE_BIGRAM: Dict[Tuple[str, str], str] = {**_US_BIGRAMS, **_IN_BIGRAMS}

_COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u.s.": "us", "u.s": "us", "u.s.a.": "us",
    "united states": "us", "united states of america": "us", "america": "us",
    "india": "india", "bharat": "india", "republic of india": "india",
    "in": "india", "ind": "india",
}
_COUNTRY_MISSING = frozenset({
    "", "na", "n/a", "null", "none", "nan", "nil", "-", "--", "unknown",
    "missing", "blank",
})

_SOUND_CODES: Dict[str, str] = {}
for _chars, _code in (("bfpv", "1"), ("cgjkqsxz", "2"), ("dt", "3"),
                      ("l", "4"), ("mn", "5"), ("r", "6")):
    for _ch in _chars:
        _SOUND_CODES[_ch] = _code


def fold_latin_diacritics(s: str) -> str:
    """Drop combining marks only after LATIN bases (NFKD); keep Indic intact."""
    if s.isascii():
        return s
    out: List[str] = []
    for ch in unicodedata.normalize("NFKD", s):
        if (unicodedata.combining(ch) and out
                and out[-1].isascii() and out[-1].isalpha()):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _is_url_token(tok: str) -> bool:
    t = tok.lower().strip(".,;:!?'\"()[]")
    if len(t) < 4 or "." not in t:
        return False
    if "www" in t or "http" in t or t.startswith("//"):
        return True
    head, _, tail = t.rpartition(".")
    return bool(head) and tail in _TLDS


def _clean_url(tok: str) -> str:
    t = tok.lower().strip(".,;:!?'\"()[]")
    changed = True
    while changed:
        changed = False
        for prefix in ("https://", "http://", "//", "www."):
            if t.startswith(prefix):
                t = t[len(prefix):]
                changed = True
    return t


def _word_split(t: str) -> List[str]:
    """Split on non-word chars but keep combining marks (Indic vowel signs,
    viramas) attached to their base letters."""
    if t.isascii():
        return _NON_WORD_RE.sub(" ", t).split()
    chars = [ch if (ch.isalnum() or unicodedata.category(ch).startswith("M"))
             else " " for ch in t]
    return "".join(chars).split()


def _split_ordinal(tok: str) -> List[str]:
    m = _ORDINAL_RE.match(tok)
    return [m.group(1), m.group(2)] if m else [tok]


def _tokenize(raw: Any, field: str,
              build_urls: bool = True) -> Tuple[List[str], List[str]]:
    """Casefold + NFKC + diacritic fold + punctuation -> space.

    Returns (urls, tokens). Apostrophes and '+' are removed, not spaced.
    """
    s = unicodedata.normalize("NFKC", "" if raw is None else str(raw)).casefold()
    urls: List[str] = []
    tokens: List[str] = []
    for tok in s.split():
        # markdown-style "[www.x.com](https://www.x.com)": split into url pieces
        for part in re.split(r"[\[\]()|]", tok):
            if not part:
                continue
            if build_urls and _is_url_token(part):
                u = _clean_url(part)
                if u and u not in urls:
                    urls.append(u)
                continue
            t = fold_latin_diacritics(part)
            for ch in ("'", "’", "ʼ"):
                t = t.replace(ch, "")
            t = t.replace("+", "")
            for p in _word_split(t):
                if field == "address":
                    tokens.extend(_split_ordinal(p))
                else:
                    tokens.append(p)
    return urls, tokens


_ORDINAL_SUFFIXES = frozenset({"st", "nd", "rd", "th"})
_FLOOR_TOKENS = frozenset({"fl", "flr", "floor"})


def _extract_states(all_tokens: Sequence[str]) -> List[str]:
    # ordinal suffixes ("22nd" -> "22","nd") must not read as state codes (ND)
    n_all = len(all_tokens)
    tokens = []
    for i, t in enumerate(all_tokens):
        # ordinal suffixes are not states: "22Nd" -> "22","nd" != ND
        if t in _ORDINAL_SUFFIXES and i > 0 and all_tokens[i - 1].isdigit():
            continue
        # floor markers are not Florida: "Fl. 0" (zips are 5+ digits)
        if (t in _FLOOR_TOKENS and i + 1 < n_all
                and all_tokens[i + 1].isdigit() and len(all_tokens[i + 1]) <= 3):
            continue
        tokens.append(t)
    out: List[str] = []
    seen = set()
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            code = _STATE_BIGRAM.get((tokens[i], tokens[i + 1]))
            if code:
                if code not in seen:
                    seen.add(code)
                    out.append(code)
                i += 2
                continue
        code = _STATE_SINGLE.get(tokens[i])
        if code and code not in seen:
            seen.add(code)
            out.append(code)
        i += 1
    return out


def soundex(token: str) -> str:
    """Standard American Soundex (vendored, no license surface)."""
    if not token:
        return ""
    s = "".join(ch for ch in token.upper() if "A" <= ch <= "Z")
    if not s:
        return ""
    first = s[0]
    codes: List[str] = []
    prev = _SOUND_CODES.get(first.lower(), "")
    for ch in s[1:]:
        code = _SOUND_CODES.get(ch.lower())
        if code is None:
            if ch not in "HW":
                prev = ""
            continue
        if code != prev:
            codes.append(code)
        prev = code
    return (first + "".join(codes) + "000")[:4]


@dataclass(frozen=True)
class NameViews:
    raw: str
    alnum: str
    tokens: Tuple[str, ...]
    core: Tuple[str, ...]
    core_sorted: str
    sorted: str
    urls: Tuple[str, ...]
    soundex: Tuple[str, ...]
    acronym: str


@dataclass(frozen=True)
class AddressViews:
    raw: str
    alnum: str
    tokens: Tuple[str, ...]
    core: Tuple[str, ...]
    core_sorted: str
    sorted: str
    digits: str
    house: str
    postal: str
    states: Tuple[str, ...]


NAME_VIEWS = ("raw", "alnum", "tokens", "core", "core_sorted", "sorted",
              "urls", "soundex", "acronym")
ADDRESS_VIEWS = ("raw", "alnum", "tokens", "core", "core_sorted", "sorted",
                 "digits", "house", "postal", "states")


def build_name_views(raw: Any, build_phonetic: bool = True,
                     build_urls: bool = True) -> NameViews:
    urls, tokens = _tokenize(raw, "name", build_urls)
    dedup = list(dict.fromkeys(tokens))
    base = [t for t in dedup if t not in NAME_STOP_TOKENS]
    core = [t for t in base if len(t) > 1]   # drop single-letter noise
    if not core and base:
        core = list(base)                     # pure initials ("A B C")
    phon = (tuple(soundex(t) for t in core if t.isascii() and t.isalpha())
            if build_phonetic else tuple())
    return NameViews(
        raw="" if raw is None else str(raw),
        alnum=" ".join(dedup), tokens=tuple(tokens), core=tuple(core),
        core_sorted=" ".join(sorted(core)), sorted=" ".join(sorted(dedup)),
        urls=tuple(urls), soundex=phon,
        acronym="".join(t[0] for t in core if t[:1].isalpha()),
    )


def build_address_views(raw: Any) -> AddressViews:
    _urls, tokens = _tokenize(raw, "address")
    dedup = list(dict.fromkeys(tokens))
    core = [t for t in dedup if t not in ADDR_STOP_TOKENS]
    house_tok = next((t for t in dedup if any(c.isdigit() for c in t)), "")
    house = "".join(c for c in house_tok if c.isdigit())
    digits = "".join(c for c in " ".join(dedup) if c.isdigit())
    postal = next((t for t in dedup
                   if t != house_tok and t.isdigit() and len(t) in (5, 6)), "")
    return AddressViews(
        raw="" if raw is None else str(raw),
        alnum=" ".join(dedup), tokens=tuple(tokens), core=tuple(core),
        core_sorted=" ".join(sorted(core)), sorted=" ".join(sorted(dedup)),
        digits=digits, house=house, postal=postal,
        states=tuple(_extract_states(dedup)),
    )


def build_country(raw: Any) -> str:
    """Open-set country normalization; missing markers fold to ''."""
    s = unicodedata.normalize("NFKC", str(raw or "")).strip().casefold()
    if s in _COUNTRY_MISSING:
        return ""
    return _COUNTRY_ALIASES.get(s, s)


def normalize_name(s: Any, view: str, build_phonetic: bool = True) -> Any:
    if view not in NAME_VIEWS:
        raise ValueError(f"unknown name view {view!r}; valid: {NAME_VIEWS}")
    return getattr(build_name_views(s, build_phonetic), view)


def normalize_address(s: Any, view: str) -> Any:
    if view not in ADDRESS_VIEWS:
        raise ValueError(f"unknown address view {view!r}; valid: {ADDRESS_VIEWS}")
    return getattr(build_address_views(s), view)


def strip_legal_suffix(token: str) -> str:
    """Token-level legal-suffix stripping ('ltd' -> '')."""
    t = str(token).strip().casefold().strip(".,")
    if t in NAME_STOP_TOKENS:
        return ""
    for suffix in ("inc", "llc", "ltd", "limited", "corp", "llp", "pvt"):
        if t.endswith(suffix) and len(t) > len(suffix) + 1:
            return t[: -len(suffix)]
    return t


def strip_legal_tokens(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if t not in NAME_STOP_TOKENS]


def char_ngrams(text: Any, n: int = 3) -> List[str]:
    """Padded char n-grams for TF-IDF retrieval (Phase 5)."""
    s = re.sub(r"\s+", " ", f" {str(text if text is not None else '').strip()} ")
    if len(s.strip()) == 0:
        return []
    if len(s) < n:
        return [s]
    return [s[i:i + n] for i in range(len(s) - n + 1)]


_DEFAULT_COLS = {"name": "business_name", "address": "business_address",
                 "country": "country"}


def build_views(df: pd.DataFrame, cfg=None, source_key: Optional[str] = None,
                fields: Sequence[str] = ("name", "address", "country"),
                build_phonetic: Optional[bool] = None,
                build_urls: Optional[bool] = None) -> pd.DataFrame:
    """Add normalized view columns to df (in place) and return it.

    Columns: name_alnum, name_tokens, name_core, name_core_sorted, name_sorted,
    name_urls, name_soundex, name_acronym, addr_alnum, addr_tokens, addr_core,
    addr_core_sorted, addr_sorted, addr_digits, addr_house, addr_postal,
    addr_states, country_norm.
    """
    overrides: Dict[str, Any] = {}
    if cfg is not None and source_key:
        overrides = (cfg.schema.overrides or {}).get(source_key, {}) or {}
    cols = {f: overrides.get(f, _DEFAULT_COLS[f]) for f in fields}
    for f, c in cols.items():
        if c not in df.columns:
            raise ValueError(f"field '{f}' column {c!r} not in DataFrame "
                             f"columns {list(df.columns)}")
    norm = getattr(cfg, "normalization", None) if cfg is not None else None
    if build_phonetic is None:
        build_phonetic = norm.build_phonetic if norm is not None else True
    if build_urls is None:
        build_urls = norm.build_urls if norm is not None else True

    if "name" in fields:
        nv = [build_name_views(x, build_phonetic, build_urls)
              for x in df[cols["name"]].tolist()]
        df["name_alnum"] = [v.alnum for v in nv]
        df["name_tokens"] = [list(v.tokens) for v in nv]
        df["name_core"] = [list(v.core) for v in nv]
        df["name_core_sorted"] = [v.core_sorted for v in nv]
        df["name_sorted"] = [v.sorted for v in nv]
        df["name_urls"] = [list(v.urls) for v in nv]
        df["name_soundex"] = [list(v.soundex) for v in nv]
        df["name_acronym"] = [v.acronym for v in nv]
    if "address" in fields:
        av = [build_address_views(x) for x in df[cols["address"]].tolist()]
        df["addr_alnum"] = [v.alnum for v in av]
        df["addr_tokens"] = [list(v.tokens) for v in av]
        df["addr_core"] = [list(v.core) for v in av]
        df["addr_core_sorted"] = [v.core_sorted for v in av]
        df["addr_sorted"] = [v.sorted for v in av]
        df["addr_digits"] = [v.digits for v in av]
        df["addr_house"] = [v.house for v in av]
        df["addr_postal"] = [v.postal for v in av]
        df["addr_states"] = [list(v.states) for v in av]
    if "country" in fields:
        df["country_norm"] = [build_country(v) for v in df[cols["country"]].tolist()]
    return df
