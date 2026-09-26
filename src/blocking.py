"""Pool blocking infrastructure (Phase 5).

- ensure_normalized: persistent multi-view parquet for S1/S2/S3, restartable per
  part. Row order in the source file IS the pool index used downstream.
- iter_column / load_ids / count_rows: streaming access to the parts.
- build_exact_index: hash index over an exact key column, postings capped.
- code_fingerprint: source-file hash for stage keys (code identity, not only
  config, so code changes invalidate cached stages).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .aws_utils import (local_artifact_path, mark_stage, publish_artifact,
                        read_json, stage_complete, write_json)
from .config import config_hash
from .data_loader import TsvReader, ensure_source
from .normalization import build_address_views, build_country, build_name_views

log = logging.getLogger(__name__)

NORM_COLUMNS = ("entity_id", "name_alnum", "name_core_joined",
                "name_core_sorted", "addr_alnum", "addr_core_joined",
                "country_norm")

_SOURCE_COLS = {"id": "entity_id", "name": "business_name",
                "address": "business_address", "country": "country"}


def code_fingerprint(paths) -> str:
    """Stable 12-hex hash of the given source files, content-based."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(Path(p).name.encode("utf-8"))
        h.update(b"\0")
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 16), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()[:12]


def _source_columns(cfg, source_key: str, header: List[str]) -> Dict[str, int]:
    overrides = (cfg.schema.overrides or {}).get(source_key, {}) or {}
    out = {}
    for role, default in _SOURCE_COLS.items():
        col = overrides.get(role, default)
        if col not in header:
            raise ValueError(f"{source_key}: column {col!r} ({role}) not in "
                             f"header {header}")
        out[role] = header.index(col)
    return out


def normalized_rel(split: str, source_key: str) -> str:
    return f"normalized/{split}/{source_key}"


def normalized_dir(cfg, split: str, source_key: str) -> Path:
    d = local_artifact_path(cfg, normalized_rel(split, source_key))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_parts(d: Path) -> List[Path]:
    return sorted(d.glob("part_*.parquet"))


def _norm_stage_hash(cfg, split: str, source_key: str) -> str:
    here = Path(__file__).resolve().parent
    code = code_fingerprint([here / "normalization.py", here / "blocking.py"])
    blob = json.dumps({"cfg": config_hash(cfg, "normalize"), "split": split,
                       "source": source_key, "code": code}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _write_part(cfg, rel_dir: str, index: int, columns: Dict[str, list],
                parts: List[Dict[str, Any]]) -> None:
    name = f"part_{index:04d}.parquet"
    table = pa.table({c: pa.array(v, type=pa.string()) for c, v in columns.items()})
    pq.write_table(table, local_artifact_path(cfg, rel_dir) / name,
                   compression="snappy")
    parts.append({"file": name, "rows": table.num_rows})
    publish_artifact(cfg, f"{rel_dir}/{name}")


def _write_normalized_source(cfg, split: str, source_key: str, stage: str,
                             chash: str) -> None:
    src_num = int(source_key.replace("source", ""))
    path = ensure_source(cfg, split, src_num)
    rel_dir = normalized_rel(split, source_key)
    out_dir = normalized_dir(cfg, split, source_key)
    manifest_p = out_dir / "parts.json"
    parts: List[Dict[str, Any]] = (read_json(manifest_p)
                                   if manifest_p.exists() else [])
    known = {p["file"] for p in parts}
    for f in out_dir.glob("part_*.parquet"):     # orphan cleanup
        if f.name not in known:
            f.unlink()
    skip = sum(p["rows"] for p in parts)
    part_rows = max(1, int(cfg.retrieval.normalize_part_rows))

    reader = TsvReader(path)
    header = reader.read_header()
    cols = _source_columns(cfg, source_key, header)
    buf: Dict[str, list] = {c: [] for c in NORM_COLUMNS}
    for row in reader.iter_rows():
        if skip:
            skip -= 1
            continue
        nv = build_name_views(row[cols["name"]])
        av = build_address_views(row[cols["address"]])
        buf["entity_id"].append(row[cols["id"]].strip())
        buf["name_alnum"].append(nv.alnum)
        buf["name_core_joined"].append(" ".join(nv.core))
        buf["name_core_sorted"].append(nv.core_sorted)
        buf["addr_alnum"].append(av.alnum)
        buf["addr_core_joined"].append(" ".join(av.core))
        buf["country_norm"].append(build_country(row[cols["country"]]))
        if len(buf["entity_id"]) >= part_rows:
            _write_part(cfg, rel_dir, len(parts), buf, parts)
            write_json(manifest_p, parts)          # checkpoint after every part
            buf = {c: [] for c in NORM_COLUMNS}
    if buf["entity_id"]:
        _write_part(cfg, rel_dir, len(parts), buf, parts)
    write_json(manifest_p, parts)
    publish_artifact(cfg, f"{rel_dir}/parts.json")
    total = sum(p["rows"] for p in parts)
    mark_stage(cfg, stage, details={"rows": total, "parts": len(parts)},
               artifacts=[f"{rel_dir}/{p['file']}" for p in parts]
                         + [f"{rel_dir}/parts.json"],
               config_hash=chash)
    log.info("normalized %s/%s: %d rows in %d parts", split, source_key,
             total, len(parts))


def ensure_normalized(cfg, split: str, sources=(1, 2, 3)) -> Dict[str, List[Path]]:
    """Write (or reuse) normalized parquet parts per source; resumable."""
    out: Dict[str, List[Path]] = {}
    for src in sources:
        key = f"source{src}"
        stage = f"normalize_{split}_{key}"
        chash = _norm_stage_hash(cfg, split, key)
        if stage_complete(cfg, stage, expected_hash=chash) is None:
            _write_normalized_source(cfg, split, key, stage, chash)
        out[key] = _list_parts(normalized_dir(cfg, split, key))
    return out


def iter_column(paths, column: str) -> Iterator[str]:
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=[column], batch_size=65536):
            for v in batch.column(0).to_pylist():
                yield v or ""


def load_ids(paths) -> np.ndarray:
    out: List[str] = []
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=["entity_id"], batch_size=65536):
            out.extend(v or "" for v in batch.column(0).to_pylist())
    return np.asarray(out, dtype=object)


def count_rows(paths) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def build_exact_index(paths, column: str = "name_core_sorted", cap: int = 300
                      ) -> Tuple[Dict[str, List[int]], int]:
    """key -> [pool indices] with capped postings; returns (index, truncated)."""
    index: Dict[str, List[int]] = {}
    truncated = 0
    idx = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=[column], batch_size=65536):
            for key in batch.column(0).to_pylist():
                k = key or ""
                if k:
                    posts = index.get(k)
                    if posts is None:
                        index[k] = [idx]
                    elif len(posts) < cap:
                        posts.append(idx)
                    else:
                        truncated += 1
                idx += 1
    return index, truncated
