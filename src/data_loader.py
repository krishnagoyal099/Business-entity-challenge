"""Schema-agnostic, robust TSV loading and compact integer ID maps.

- Files are read as raw text (no pandas NA coercion).
- Ragged rows are truncated and counted; short rows are padded.
- Files decode with errors='replace' so invalid UTF-8 never crashes a stage.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .aws_utils import (aws_enabled, read_json, s3_download, s3_exists, s3_uri,
                         write_json)

log = logging.getLogger(__name__)

# Long address/description fields must not crash the reader (default is 128 KB).
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

SourceRef = Union[int, str]


def canonical_source(source: SourceRef) -> str:
    """Normalize a source reference (1/'s1'/'source1'/'gt'/'ground_truth'/...)."""
    if isinstance(source, bool):
        raise ValueError(f"invalid source reference: {source!r}")
    if isinstance(source, int):
        if source in (1, 2, 3):
            return f"source{source}"
        raise ValueError(f"integer source must be 1, 2 or 3, got {source}")
    s = str(source).strip().lower().replace("-", "_")
    mapping = {"s1": "source1", "1": "source1", "source1": "source1",
               "s2": "source2", "2": "source2", "source2": "source2",
               "s3": "source3", "3": "source3", "source3": "source3",
               "gt": "ground_truth", "truth": "ground_truth",
               "labels": "ground_truth", "ground_truth": "ground_truth"}
    if s not in mapping:
        raise ValueError(f"unknown source reference: {source!r}")
    return mapping[s]


def source_filename(cfg, split: str, source: SourceRef) -> str:
    key = canonical_source(source)
    files = getattr(cfg.data, split, None)
    if files is None:
        raise ValueError(f"unknown split: {split!r}")
    fname = getattr(files, key, "")
    if not fname:
        raise ValueError(f"no filename configured for {split}/{key}")
    return fname


def local_source_path(cfg, split: str, source: SourceRef) -> Path:
    return Path(cfg.paths.data_dir).expanduser() / split / source_filename(
        cfg, split, source)


def s3_source_uri(cfg, split: str, source: SourceRef) -> Optional[str]:
    try:
        fname = source_filename(cfg, split, source)
    except ValueError:
        return None
    return s3_uri(cfg, f"data/{split}/{fname}")


def ensure_source(cfg, split: str, source: SourceRef) -> Path:
    """Return a local path for a source file, downloading from S3 if needed."""
    local = local_source_path(cfg, split, source)
    if local.exists():
        return local
    uri = s3_source_uri(cfg, split, source)
    if uri and aws_enabled(cfg) and s3_exists(
            uri, region=getattr(cfg.aws, "region", None)):
        log.info("downloading %s -> %s", uri, local)
        s3_download(uri, local, region=getattr(cfg.aws, "region", None))
        return local
    raise FileNotFoundError(
        f"[{split}/{canonical_source(source)}] not found locally at '{local}' "
        f"nor in S3 at '{uri}'")


def source_available(cfg, split: str, source: SourceRef) -> bool:
    try:
        if local_source_path(cfg, split, source).exists():
            return True
    except ValueError:
        return False
    uri = s3_source_uri(cfg, split, source)
    if uri and aws_enabled(cfg):
        return s3_exists(uri, region=getattr(cfg.aws, "region", None))
    return False


def _dedupe_header(names: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    used: set = set()
    out: List[str] = []
    renames: List[Tuple[str, str]] = []
    for n in names:
        cand = n if n else "column"
        if cand in used:
            k = 2
            while f"{cand}#{k}" in used:
                k += 1
            cand = f"{cand}#{k}"
            renames.append((n, cand))
        used.add(cand)
        out.append(cand)
    return out, renames


class TsvReader:
    """Streaming reader for competition TSVs with forensic counters."""

    def __init__(self, path, delimiter: str = "\t"):
        self.path = Path(path)
        self.delimiter = delimiter
        self.header: Optional[List[str]] = None
        self.original_header: Optional[List[str]] = None
        self.duplicate_header_names: List[Tuple[str, str]] = []
        self.leading_blank_lines = 0
        self.blank_rows = 0
        self.ragged_rows = 0
        self.short_rows = 0
        self.ragged_examples: List[Tuple[int, int]] = []
        self.rows_read = 0

    def _open(self):
        return open(self.path, "r", encoding="utf-8", errors="replace", newline="")

    def _set_header(self, fields: List[str]) -> None:
        self.original_header = list(fields)
        names, renames = _dedupe_header([f.strip() for f in fields])
        self.header = names
        self.duplicate_header_names = renames

    def read_header(self) -> List[str]:
        if self.header is not None:
            return self.header
        with self._open() as fh:
            for fields in csv.reader(fh, delimiter=self.delimiter,
                                     quoting=csv.QUOTE_NONE):
                if not fields or all(c == "" for c in fields):
                    self.leading_blank_lines += 1
                    continue
                self._set_header(fields)
                break
        if self.header is None:
            raise ValueError(f"file has no header row: {self.path}")
        return self.header

    def iter_rows(self, max_rows: Optional[int] = None) -> Iterator[List[str]]:
        header = self.read_header()
        n = len(header)
        with self._open() as fh:
            reader = csv.reader(fh, delimiter=self.delimiter,
                                quoting=csv.QUOTE_NONE)
            seen_header = False
            yielded = 0
            for line_no, fields in enumerate(reader, start=1):
                if not fields or all(c == "" for c in fields):
                    if not seen_header:
                        continue
                    self.blank_rows += 1
                    continue
                if not seen_header:
                    seen_header = True
                    continue
                if max_rows is not None and yielded >= max_rows:
                    break
                if len(fields) > n:
                    self.ragged_rows += 1
                    if len(self.ragged_examples) < 5:
                        self.ragged_examples.append((line_no, len(fields)))
                    fields = fields[:n]
                elif 0 < len(fields) < n:
                    self.short_rows += 1
                    fields = fields + [""] * (n - len(fields))
                yielded += 1
                self.rows_read += 1
                yield fields

    def read_all(self, max_rows: Optional[int] = None) -> List[List[str]]:
        return list(self.iter_rows(max_rows=max_rows))

    def as_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        rows = self.read_all(max_rows)
        return pd.DataFrame(rows, columns=self.read_header())

    def stats(self) -> Dict:
        return {
            "rows": self.rows_read,
            "ragged_rows": self.ragged_rows,
            "short_rows": self.short_rows,
            "blank_rows": self.blank_rows,
            "leading_blank_lines": self.leading_blank_lines,
            "ragged_examples": list(self.ragged_examples),
            "duplicate_header_names": list(self.duplicate_header_names),
        }


def read_tsv(path, max_rows: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
    reader = TsvReader(path)
    header = reader.read_header()
    return header, reader.read_all(max_rows)


def load_source(cfg, split: str, source: SourceRef,
                max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a source as a DataFrame of raw strings (no NA coercion)."""
    path = ensure_source(cfg, split, source)
    reader = TsvReader(path)
    df = reader.as_dataframe(max_rows)
    df.attrs["source_path"] = str(path)
    df.attrs["split"] = split
    df.attrs["source"] = canonical_source(source)
    df.attrs["tsv_stats"] = reader.stats()
    return df


def load_sources(cfg, split: str,
                 sources: Sequence[SourceRef] = (1, 2, 3)) -> Dict[str, pd.DataFrame]:
    return {canonical_source(s): load_source(cfg, split, s) for s in sources}


def extract_column(path, column: str) -> List[str]:
    """Extract one column as raw strings (single pass, memory-light)."""
    reader = TsvReader(path)
    header = reader.read_header()
    if column not in header:
        raise ValueError(f"column {column!r} not in header {header}")
    idx = header.index(column)
    return [row[idx] for row in reader.iter_rows()]


class IdMaps:
    """Compact integer IDs per source namespace; JSON-serializable artifact."""

    def __init__(self, ids: Dict[str, List[str]]):
        self._ids: Dict[str, List[str]] = {k: list(v) for k, v in ids.items()}
        self._maps: Dict[str, Dict[str, int]] = {
            k: {v: i for i, v in enumerate(vals)} for k, vals in self._ids.items()}
        self.duplicates: Dict[str, int] = {}

    @classmethod
    def build(cls, ids_by_source: Dict[str, Sequence[str]]) -> "IdMaps":
        ids: Dict[str, List[str]] = {}
        dups: Dict[str, int] = {}
        for key, vals in ids_by_source.items():
            nonempty = [v for v in (str(x).strip() for x in vals) if v]
            uniq = list(dict.fromkeys(nonempty))
            ids[key] = uniq
            dups[key] = len(nonempty) - len(uniq)
        obj = cls(ids)
        obj.duplicates = dups
        return obj

    @property
    def sources(self) -> List[str]:
        return sorted(self._ids)

    def size(self, source: str) -> int:
        return len(self._ids[source])

    def encode(self, source: str, raw_id: str) -> int:
        try:
            return self._maps[source][str(raw_id).strip()]
        except KeyError as exc:
            raise KeyError(f"unknown id {raw_id!r} for source {source!r}") from exc

    def decode(self, source: str, idx: int) -> str:
        return self._ids[source][idx]

    def to_dict(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._ids.items()}

    def save(self, path) -> str:
        return write_json(path, {"ids": self.to_dict(), "duplicates": self.duplicates})

    @classmethod
    def load(cls, path) -> "IdMaps":
        data = read_json(path)
        obj = cls(data["ids"])
        obj.duplicates = data.get("duplicates", {})
        return obj
