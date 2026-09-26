"""S3 utilities, artifact path mapping, manifests and stage resume markers.

- Local filesystem paths are the unit of computation; S3 is touched only here.
- Artifact relpaths map to s3://<bucket>/<prefix>/artifacts/<rel>.
- Data files map to s3://<bucket>/<prefix>/data/<split>/<filename>.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

_S3_CLIENTS: Dict[str, Any] = {}

PathLike = Union[str, Path]


def is_s3_uri(path: PathLike) -> bool:
    return str(path).startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    s = str(uri)
    if not s.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = s[5:].partition("/")
    if not bucket:
        raise ValueError(f"malformed s3:// URI (no bucket): {uri!r}")
    return bucket, key


def aws_enabled(cfg: Any) -> bool:
    aws = getattr(cfg, "aws", None)
    return bool(aws and getattr(aws, "use_s3", False) and getattr(aws, "s3_bucket", ""))


def get_s3_client(region: Optional[str] = None):
    key = region or "__default__"
    if key not in _S3_CLIENTS:
        import boto3
        _S3_CLIENTS[key] = (boto3.client("s3", region_name=region)
                            if region else boto3.client("s3"))
    return _S3_CLIENTS[key]


def s3_upload(local_path: PathLike, uri: str, region: Optional[str] = None) -> str:
    bucket, key = parse_s3_uri(uri)
    get_s3_client(region).upload_file(str(local_path), bucket, key)
    log.debug("s3_upload: %s -> %s", local_path, uri)
    return uri


def s3_download(uri: str, local_path: PathLike, region: Optional[str] = None) -> str:
    bucket, key = parse_s3_uri(uri)
    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    get_s3_client(region).download_file(bucket, key, str(dest))
    log.debug("s3_download: %s -> %s", uri, local_path)
    return str(dest)


def _is_not_found(exc: Exception) -> bool:
    """True only for genuine 'object does not exist' errors."""
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return True
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", ""))
        return code in ("404", "NoSuchKey", "NotFound")
    return False


def s3_exists(uri: str, region: Optional[str] = None) -> bool:
    """Existence check; auth/network/region errors propagate (not 'missing')."""
    bucket, key = parse_s3_uri(uri)
    try:
        get_s3_client(region).head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        if _is_not_found(exc):
            return False
        raise


def s3_size(uri: str, region: Optional[str] = None) -> Optional[int]:
    bucket, key = parse_s3_uri(uri)
    try:
        return int(get_s3_client(region).head_object(
            Bucket=bucket, Key=key).get("ContentLength", 0))
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise


def s3_list(prefix_uri: str, region: Optional[str] = None) -> List[str]:
    bucket, prefix = parse_s3_uri(prefix_uri)
    client = get_s3_client(region)
    out: List[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append(f"s3://{bucket}/{obj['Key']}")
    return sorted(out)


def cached_download(uri: str, cache_dir: PathLike, region: Optional[str] = None) -> str:
    """Download uri into cache_dir once; return the cached local path."""
    cache = Path(cache_dir)
    key = hashlib.sha1(str(uri).encode("utf-8")).hexdigest()[:16]
    dest = cache / key / Path(str(uri)).name
    if not dest.exists():
        log.info("cached_download: %s -> %s", uri, dest)
        s3_download(uri, dest, region=region)
    return str(dest)


def s3_uri(cfg: Any, rel: str) -> Optional[str]:
    if not aws_enabled(cfg):
        return None
    bucket = str(cfg.aws.s3_bucket).strip().strip("/")
    prefix = str(cfg.aws.s3_prefix).strip().strip("/")
    base = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    return f"{base}/{rel}"


def local_artifact_path(cfg: Any, rel: str) -> Path:
    p = Path(cfg.paths.artifact_dir).expanduser() / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def s3_artifact_uri(cfg: Any, rel: str) -> Optional[str]:
    return s3_uri(cfg, f"artifacts/{rel}")


def publish_artifact(cfg: Any, rel: str, local_path: Optional[PathLike] = None) -> Optional[str]:
    """Upload a local artifact to its canonical S3 location; no-op if S3 disabled."""
    if not aws_enabled(cfg):
        log.debug("publish_artifact: S3 disabled; skipping %s", rel)
        return None
    local = Path(local_path) if local_path is not None else local_artifact_path(cfg, rel)
    if not local.exists():
        raise FileNotFoundError(f"artifact to publish not found: {local}")
    uri = s3_artifact_uri(cfg, rel)
    s3_upload(local, uri, region=getattr(cfg.aws, "region", None))
    log.info("published %s -> %s", rel, uri)
    return uri


def publish_dir(cfg: Any, rel_dir: str, local_dir: PathLike) -> List[str]:
    uris: List[str] = []
    root = Path(local_dir)
    for f in sorted(root.rglob("*")):
        if f.is_file():
            rel = f"{rel_dir}/{f.relative_to(root).as_posix()}"
            uri = publish_artifact(cfg, rel, local_path=f)
            if uri:
                uris.append(uri)
    return uris


def fetch_artifact(cfg: Any, rel: str) -> Path:
    """Return a local path for an artifact, downloading from S3 if needed."""
    local = local_artifact_path(cfg, rel)
    if local.exists():
        return local
    uri = s3_artifact_uri(cfg, rel)
    if uri and s3_exists(uri, region=getattr(cfg.aws, "region", None)):
        s3_download(uri, local, region=getattr(cfg.aws, "region", None))
        return local
    raise FileNotFoundError(
        f"artifact '{rel}' not found locally ({local}) or in S3 ({uri})")


def write_json(path: PathLike, obj: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return str(p)


def read_json(path: PathLike) -> Any:
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_manifest(path: PathLike, manifest: Dict[str, Any]) -> str:
    return write_json(path, manifest)


def read_manifest(path: PathLike) -> Dict[str, Any]:
    return read_json(path)


def sha256_file(path: PathLike, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, timeout=5, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _manifest_rel(stage: str) -> str:
    return f"manifests/{stage}/manifest.json"


def _marker_rel(stage: str) -> str:
    return f"manifests/{stage}/_SUCCESS"


def mark_stage(cfg: Any, stage: str, details: Optional[Dict[str, Any]] = None,
               artifacts: Optional[List[str]] = None,
               config_hash: Optional[str] = None,
               duration_s: Optional[float] = None,
               peak_rss_mb: Optional[float] = None) -> Dict[str, Any]:
    """Write the stage manifest + _SUCCESS marker locally and publish to S3."""
    manifest: Dict[str, Any] = {
        "stage": stage,
        "config_hash": config_hash,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "peak_rss_mb": peak_rss_mb,
        "git_sha": git_sha(),
        "details": details or {},
        "artifacts": list(artifacts or []),
    }
    write_manifest(local_artifact_path(cfg, _manifest_rel(stage)), manifest)
    marker = local_artifact_path(cfg, _marker_rel(stage))
    marker.write_text(
        json.dumps({"stage": stage, "finished_at": manifest["finished_at"]}) + "\n",
        encoding="utf-8")
    for rel in manifest["artifacts"]:
        try:
            publish_artifact(cfg, rel)
        except FileNotFoundError:
            log.warning("mark_stage: listed artifact missing locally: %s", rel)
    publish_artifact(cfg, _manifest_rel(stage))
    publish_artifact(cfg, _marker_rel(stage))
    log.info("stage '%s' marked complete (hash=%s, artifacts=%d)",
             stage, config_hash, len(manifest["artifacts"]))
    return manifest


def stage_complete(cfg: Any, stage: str,
                   expected_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the stage manifest if the stage completed with a matching hash; else None."""
    region = getattr(cfg.aws, "region", None)
    mpath = local_artifact_path(cfg, _manifest_rel(stage))
    if not mpath.exists():
        uri = s3_artifact_uri(cfg, _manifest_rel(stage))
        if uri and s3_exists(uri, region=region):
            s3_download(uri, mpath, region=region)
    if not mpath.exists():
        return None
    marker = local_artifact_path(cfg, _marker_rel(stage))
    if not marker.exists():
        uri = s3_artifact_uri(cfg, _marker_rel(stage))
        if uri and s3_exists(uri, region=region):
            s3_download(uri, marker, region=region)
    if not marker.exists():
        return None
    manifest = read_manifest(mpath)
    if expected_hash is not None and manifest.get("config_hash") != expected_hash:
        log.info("stage '%s' complete but config/inputs changed (%s != %s); will rerun",
                 stage, manifest.get("config_hash"), expected_hash)
        return None
    return manifest


def fetch_stage_outputs(cfg: Any, stage: str) -> List[Path]:
    manifest = stage_complete(cfg, stage)
    if manifest is None:
        raise RuntimeError(f"stage '{stage}' is not complete; run it first")
    return [fetch_artifact(cfg, rel) for rel in manifest.get("artifacts", [])]
