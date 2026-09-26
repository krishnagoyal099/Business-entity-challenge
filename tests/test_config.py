import yaml

from src import aws_utils
from src.config import Config, config_hash, load_config, resolve_path


def _write_yaml(tmp_path, data):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


def test_load_minimal_and_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate from any repo .env
    for v in ("LOCAL_DATA_DIR", "RANDOM_SEED"):
        monkeypatch.delenv(v, raising=False)
    cfg = load_config(_write_yaml(tmp_path, {"project": {"seed": 7}}))
    assert cfg.project.seed == 7
    assert cfg.paths.data_dir == "dataset"
    assert cfg.data.train.source1 == "train_source1.tsv"
    assert cfg.data.test.ground_truth == ""


def test_env_overrides_beat_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write_yaml(tmp_path, {"aws": {"s3_bucket": "from-yaml", "use_s3": True}})
    monkeypatch.setenv("S3_BUCKET", "from-env")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    cfg = load_config(path)
    assert cfg.aws.s3_bucket == "from-env"
    assert cfg.aws.region == "eu-west-1"
    assert cfg.aws.use_s3 is True


def test_unknown_keys_warn_not_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_write_yaml(tmp_path, {"audit": {"bogus_key": 1, "top_k": 3}}))
    assert cfg.audit.top_k == 3


def test_config_hash_stable_and_stage_scoped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_write_yaml(tmp_path, {}))
    h1 = config_hash(cfg, "audit")
    assert h1 == config_hash(cfg, "audit")
    cfg.execution.n_jobs = 99
    assert config_hash(cfg, "audit") == h1
    cfg.audit.top_k = 11
    assert config_hash(cfg, "audit") != h1


def test_resolve_path_local_and_s3(tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths.artifact_dir = str(tmp_path / "artifacts")
    local = str(tmp_path / "x.tsv")
    assert resolve_path(local) == local
    monkeypatch.setattr(aws_utils, "cached_download",
                        lambda uri, cache, region=None: "/cached/x.tsv")
    assert resolve_path("s3://bucket/key/x.tsv", cfg) == "/cached/x.tsv"
