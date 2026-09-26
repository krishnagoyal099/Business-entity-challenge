import pytest

from conftest import _write_tsv
from src.data_loader import IdMaps, TsvReader, ensure_source, load_source


def test_reader_basic_and_blank(tmp_path):
    p = tmp_path / "a.tsv"
    _write_tsv(p, ["id", "name"], [["1", "Alpha"], [], ["2", " Beta "]])
    r = TsvReader(p)
    assert r.read_header() == ["id", "name"]
    assert r.read_all() == [["1", "Alpha"], ["2", " Beta "]]
    assert r.blank_rows == 1
    assert r.stats()["rows"] == 2


def test_reader_ragged_and_short(tmp_path):
    p = tmp_path / "a.tsv"
    _write_tsv(p, ["id", "n", "a"], [["1", "x", "y", "EXTRA"], ["2", "only"]])
    r = TsvReader(p)
    assert r.read_all() == [["1", "x", "y"], ["2", "only", ""]]
    assert r.ragged_rows == 1
    assert r.short_rows == 1


def test_reader_utf8_replacement(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_bytes(b"id\tname\nA\t\xff\xfe\n")
    assert TsvReader(p).read_all()[0][1].count("�") == 2


def test_reader_dedupes_header(tmp_path):
    p = tmp_path / "a.tsv"
    _write_tsv(p, ["id", "id"], [["1", "2"]])
    assert TsvReader(p).read_header() == ["id", "id#2"]


def test_reader_handles_huge_field(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("id\ttext\n1\t" + "x" * 300_000 + "\n", encoding="utf-8")
    rows = TsvReader(p).read_all()
    assert len(rows[0][1]) == 300_000


def test_load_source_attrs(synth):
    cfg, _ = synth
    df = load_source(cfg, "train", 1)
    assert df.shape == (4, 4)
    assert df.attrs["source"] == "source1"
    assert df.attrs["tsv_stats"]["rows"] == 4


def test_ensure_source_missing(synth, tmp_path):
    cfg, _ = synth
    cfg.paths.data_dir = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        ensure_source(cfg, "train", 1)


def test_idmaps_roundtrip(tmp_path):
    m = IdMaps.build({"source2": ["A", "B", "A", ""]})
    assert m.size("source2") == 2
    assert m.duplicates["source2"] == 1
    assert m.encode("source2", "B") == 1
    assert m.decode("source2", 1) == "B"
    with pytest.raises(KeyError):
        m.encode("source2", "Z")
    p = tmp_path / "ids.json"
    m.save(p)
    assert IdMaps.load(p).encode("source2", "B") == 1
