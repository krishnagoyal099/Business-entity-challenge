"""Centralized configuration: YAML + environment overrides + path resolution.

- Precedence: real environment variables > .env file > YAML > dataclass defaults.
- resolve_path() turns a local path or s3:// URI into a local filesystem path.
- config_hash(cfg, stage) produces the cache key used by stage manifests.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

log = logging.getLogger(__name__)

ENV_TO_CONFIG = {
    "AWS_REGION": ("aws", "region"),
    "S3_BUCKET": ("aws", "s3_bucket"),
    "S3_PREFIX": ("aws", "s3_prefix"),
    "SAGEMAKER_ROLE": ("aws", "role"),
    "LOCAL_DATA_DIR": ("paths", "data_dir"),
    "LOCAL_ARTIFACT_DIR": ("paths", "artifact_dir"),
    "LOCAL_OUTPUT_DIR": ("paths", "output_dir"),
    "RANDOM_SEED": ("project", "seed"),
    "N_JOBS": ("execution", "n_jobs"),
}
_INT_FIELDS = {("project", "seed"), ("execution", "n_jobs")}

STAGE_CONFIG_KEYS: Dict[str, tuple] = {
    "audit": ("audit", "data", "schema"),
    "normalize": ("normalization", "schema"),
    "norm_bench": ("normalization", "schema"),
    "candidates": ("retrieval", "normalization", "schema"),
}


@dataclass
class ProjectCfg:
    name: str = "business-entity-resolution"
    seed: int = 42


@dataclass
class AwsCfg:
    use_s3: bool = False
    region: str = "us-east-1"
    s3_bucket: str = ""
    s3_prefix: str = "business-entity-resolution"
    role: str = ""


@dataclass
class ExecutionCfg:
    mode: str = "local"
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 40
    n_jobs: int = 4


@dataclass
class PathsCfg:
    data_dir: str = "dataset"
    artifact_dir: str = "artifacts"
    output_dir: str = "output"


@dataclass
class SplitFiles:
    source1: str = ""
    source2: str = ""
    source3: str = ""
    ground_truth: str = ""


@dataclass
class DataCfg:
    train: SplitFiles = field(default_factory=lambda: SplitFiles(
        source1="train_source1.tsv", source2="train_source2.tsv",
        source3="train_source3.tsv", ground_truth="train_ground_truth.tsv"))
    test: SplitFiles = field(default_factory=lambda: SplitFiles(
        source1="test_source1.tsv", source2="test_source2.tsv",
        source3="test_source3.tsv", ground_truth=""))


@dataclass
class LoggingCfg:
    level: str = "INFO"
    dir: str = ""


@dataclass
class AuditCfg:
    pattern_sample_rows: int = 100_000
    top_k: int = 25
    unique_cap: int = 100_000
    save_head_rows: int = 50


@dataclass
class ValidationCfg:
    """Leakage-safe OOF fold settings (Phase 3)."""
    n_folds: int = 5
    stratify: bool = True
    # S1 ids absent from the GT are treated as zero-match. Working default;
    # verify against the official evaluator after the full audit.
    absent_is_zero: bool = True


@dataclass
class NormalizationCfg:
    """Multi-view normalization settings (Phase 4)."""
    char_ngram: int = 3
    build_phonetic: bool = True
    build_urls: bool = True


@dataclass
class RetrievalCfg:
    """Phase 5 candidate generation settings."""
    s1_chunk_rows: int = 25_000
    max_spread: int = 150_000_000      # transient nnz budget per matmul chunk
    normalize_part_rows: int = 500_000
    channels: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "exact": {"enabled": True, "k": 100, "max_postings": 300},
        "char_name": {"enabled": True, "k": 50, "max_df": 0.005},
        "word_name": {"enabled": True, "k": 30, "max_df": 0.02},
        "rare": {"enabled": True, "k": 30, "max_df_abs": 50},
        "char_addr": {"enabled": True, "k": 30, "max_df": 0.01},
    })


@dataclass
class SchemaCfg:
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    aws: AwsCfg = field(default_factory=AwsCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    audit: AuditCfg = field(default_factory=AuditCfg)
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    schema: SchemaCfg = field(default_factory=SchemaCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        specs = {
            "project": ProjectCfg, "aws": AwsCfg, "execution": ExecutionCfg,
            "paths": PathsCfg, "logging": LoggingCfg, "audit": AuditCfg,
            "validation": ValidationCfg, "normalization": NormalizationCfg,
            "retrieval": RetrievalCfg, "schema": SchemaCfg,
        }
        for key, dcls in specs.items():
            section = d.get(key)
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ValueError(
                    f"config section '{key}' must be a mapping, "
                    f"got {type(section).__name__}")
            valid = {f.name for f in fields(dcls)}
            unknown = set(section) - valid
            if unknown:
                log.warning("config: ignoring unknown keys in '%s': %s",
                            key, sorted(unknown))
            setattr(cfg, key, dcls(**{k: v for k, v in section.items()
                                      if k in valid}))
        data = d.get("data")
        if isinstance(data, dict):
            for split in ("train", "test"):
                sub = data.get(split)
                if sub is None:
                    continue
                if not isinstance(sub, dict):
                    raise ValueError(f"config section data.{split} must be a mapping")
                current = getattr(cfg.data, split)
                values = {k: v for k, v in sub.items()
                          if k in {"source1", "source2", "source3", "ground_truth"}}
                unknown = set(sub) - set(values)
                if unknown:
                    log.warning("config: ignoring unknown keys in data.%s: %s",
                                split, sorted(unknown))
                setattr(cfg.data, split,
                        SplitFiles(**{**asdict(current), **values}))
        elif data is not None:
            raise ValueError("config section 'data' must be a mapping")
        return cfg


def _apply_env(cfg: Config) -> None:
    if load_dotenv is not None:
        load_dotenv()  # does not override variables already in the environment
    for env_name, (section, attr) in ENV_TO_CONFIG.items():
        raw = os.environ.get(env_name)
        if raw is None or raw.strip() == "":
            continue
        raw = raw.strip()
        value: Any = int(raw) if (section, attr) in _INT_FIELDS else raw
        setattr(getattr(cfg, section), attr, value)
    if os.environ.get("ENVIRONMENT", "").strip().lower() == "aws":
        cfg.aws.use_s3 = True


def _validate(cfg: Config) -> None:
    if cfg.execution.mode not in ("local", "sagemaker"):
        raise ValueError(f"execution.mode must be 'local' or 'sagemaker', "
                         f"got {cfg.execution.mode!r}")
    if cfg.execution.instance_count < 1:
        raise ValueError("execution.instance_count must be >= 1")
    if cfg.validation.n_folds < 2:
        raise ValueError("validation.n_folds must be >= 2")
    if cfg.normalization.char_ngram < 2:
        raise ValueError("normalization.char_ngram must be >= 2")
    for name, p in cfg.retrieval.channels.items():
        if int(p.get("k", 30)) > 200:
            raise ValueError(f"retrieval.channels.{name}.k must be <= 200 "
                             "(ranks are stored as uint8)")
    if cfg.aws.use_s3 and not cfg.aws.s3_bucket:
        log.warning("aws.use_s3 is true but S3_BUCKET is empty; "
                    "S3 publishing will be disabled")


def load_config(path: str) -> Config:
    """Load a YAML config (local path or s3:// URI), apply env overrides, validate."""
    p = str(path)
    if p.startswith("s3://"):
        from .aws_utils import cached_download
        p = cached_download(p, Path.cwd() / "artifacts" / "cache" / "downloads")
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a YAML mapping: {p}")
    cfg = Config.from_dict(raw)
    _apply_env(cfg)
    _validate(cfg)
    return cfg


def config_hash(cfg: Config, stage: Optional[str] = None) -> str:
    """Stable 12-hex-char hash of the config subset relevant to a stage."""
    d = cfg.to_dict()
    keys = STAGE_CONFIG_KEYS.get(stage or "")
    sub = {k: d[k] for k in keys} if keys else d
    blob = json.dumps(sub, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def resolve_path(uri: str, cfg: Optional[Config] = None) -> str:
    """Return a local filesystem path for a local path or an s3:// URI."""
    if str(uri).startswith("s3://"):
        from .aws_utils import cached_download
        if cfg is not None:
            cache = Path(cfg.paths.artifact_dir).expanduser() / "cache" / "downloads"
        else:
            cache = Path.cwd() / "artifacts" / "cache" / "downloads"
        return cached_download(str(uri), cache)
    return str(Path(uri).expanduser())
