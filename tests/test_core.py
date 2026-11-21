import json
import sys
from types import SimpleNamespace

from app import core


def test_parse_label_handles_generation_noise():
    assert core.parse_label("  payment terms.\n", ["payment terms", "governing law"]) == "payment terms"
    assert core.parse_label("unrecognized output", ["payment terms"]) == "<unknown>"


def test_directory_size(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"12345")
    assert core.directory_size(tmp_path) == 8


def test_load_rows_reports_missing_preparation(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    try:
        core.load_rows("test")
    except FileNotFoundError as exc:
        assert "prepare" in str(exc)
    else:
        raise AssertionError("expected a clear missing dataset error")


def test_json_writer_is_machine_readable(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    core._json(destination, {"ok": True})
    assert json.loads(destination.read_text()) == {"ok": True}


def test_prepare_dataset_cleans_splits_and_caches(monkeypatch, tmp_path):
    labels = ["Payment", "Term", "Law"]
    class FakeSplit(list):
        features = {"label": SimpleNamespace(names=labels)}
    rows = [{"text": f" clause {i} ", "label": i % len(labels)} for i in range(60)]
    fake = {"train": FakeSplit(rows), "validation": FakeSplit([]), "test": FakeSplit([])}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: fake))
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    first = core.prepare_dataset(limit=30, seed=3)
    second = core.prepare_dataset(limit=30, seed=3)
    assert first == second
    assert first["examples"] == 30 and first["classes"] == 3
    assert first["splits"] == {"train": 24, "validation": 3, "test": 3}
    loaded = {name: core.load_rows(name) for name in ("train", "validation", "test")}
    text_sets = [{r["text"] for r in values} for values in loaded.values()]
    assert not (text_sets[0] & text_sets[1] or text_sets[0] & text_sets[2] or text_sets[1] & text_sets[2])
    assert all(r["text"] == r["text"].strip() for values in loaded.values() for r in values)
