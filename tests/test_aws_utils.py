from pathlib import Path

import pytest

from src import aws_utils
from src.config import Config


class _FakePaginator:
    def __init__(self, client):
        self._client = client

    def paginate(self, **kwargs):
        return iter([self._client.list_objects_v2(**kwargs)])


class FakeS3Client:
    def __init__(self):
        self.store = {}

    def upload_file(self, Filename, Bucket, Key):
        self.store[(Bucket, Key)] = Path(Filename).read_bytes()

    def download_file(self, Bucket, Key, Filename):
        if (Bucket, Key) not in self.store:
            raise FileNotFoundError(f"missing s3://{Bucket}/{Key}")
        p = Path(Filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.store[(Bucket, Key)])

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise KeyError(f"missing s3://{Bucket}/{Key}")
        return {"ContentLength": len(self.store[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix=""):
        contents = [{"Key": k} for (b, k) in sorted(self.store)
                    if b == Bucket and k.startswith(Prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self)


@pytest.fixture
def fake_s3(monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(aws_utils, "get_s3_client", lambda region=None: client)
    return client


def _aws_cfg(tmp_path):
    cfg = Config()
    cfg.aws.use_s3 = True
    cfg.aws.s3_bucket = "bkt"
    cfg.aws.s3_prefix = "proj"
    cfg.paths.artifact_dir = str(tmp_path / "artifacts")
    return cfg


def test_parse_s3_uri():
    assert aws_utils.parse_s3_uri("s3://bkt/a/b.txt") == ("bkt", "a/b.txt")
    assert aws_utils.is_s3_uri("s3://bkt/x")
    assert not aws_utils.is_s3_uri("/local/x")
    with pytest.raises(ValueError):
        aws_utils.parse_s3_uri("not-a-uri")


def test_json_roundtrip(tmp_path):
    p = tmp_path / "a.json"
    aws_utils.write_json(p, {"x": 1})
    assert aws_utils.read_json(p)["x"] == 1


def test_mark_stage_and_complete_local(tmp_path):
    cfg = Config()
    cfg.paths.artifact_dir = str(tmp_path / "artifacts")
    aws_utils.mark_stage(cfg, "audit", details={"n": 1}, artifacts=[],
                         config_hash="abc123")
    m = aws_utils.stage_complete(cfg, "audit", expected_hash="abc123")
    assert m is not None and m["details"]["n"] == 1
    assert aws_utils.stage_complete(cfg, "audit", expected_hash="zzz") is None
    assert aws_utils.stage_complete(cfg, "other_stage") is None


def test_publish_disabled_noop(tmp_path):
    cfg = Config()
    cfg.paths.artifact_dir = str(tmp_path / "artifacts")
    assert aws_utils.publish_artifact(cfg, "data_audit/profile.json") is None


def test_s3_roundtrip_and_list(fake_s3, tmp_path):
    cfg = _aws_cfg(tmp_path)
    local = aws_utils.local_artifact_path(cfg, "x.txt")
    local.write_text("hello", encoding="utf-8")
    uri = aws_utils.publish_artifact(cfg, "x.txt")
    assert uri == "s3://bkt/proj/artifacts/x.txt"
    assert aws_utils.s3_exists(uri) is True
    assert aws_utils.s3_exists("s3://bkt/proj/artifacts/nope") is False
    dest = tmp_path / "dest.txt"
    aws_utils.s3_download(uri, dest)
    assert dest.read_text(encoding="utf-8") == "hello"
    assert uri in aws_utils.s3_list("s3://bkt/proj/")


def test_s3_exists_propagates_auth_errors(monkeypatch):
    class Denied(Exception):
        response = {"Error": {"Code": "403"}}

    class Client:
        def head_object(self, **kw):
            raise Denied()

    monkeypatch.setattr(aws_utils, "get_s3_client", lambda region=None: Client())
    with pytest.raises(Denied):
        aws_utils.s3_exists("s3://bkt/key")


def test_cached_download(fake_s3, tmp_path):
    cfg = _aws_cfg(tmp_path)
    local = aws_utils.local_artifact_path(cfg, "y.tsv")
    local.write_text("data", encoding="utf-8")
    aws_utils.publish_artifact(cfg, "y.tsv")
    cached = aws_utils.cached_download("s3://bkt/proj/artifacts/y.tsv", tmp_path / "cache")
    assert Path(cached).read_text(encoding="utf-8") == "data"
    assert aws_utils.cached_download("s3://bkt/proj/artifacts/y.tsv",
                                     tmp_path / "cache") == cached


def test_fetch_artifact_restores_from_s3(fake_s3, tmp_path):
    cfg = _aws_cfg(tmp_path)
    local = aws_utils.local_artifact_path(cfg, "z.bin")
    local.write_bytes(b"123")
    aws_utils.publish_artifact(cfg, "z.bin")
    local.unlink()
    assert aws_utils.fetch_artifact(cfg, "z.bin").read_bytes() == b"123"
