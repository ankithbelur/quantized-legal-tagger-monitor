import json
import sys
from types import SimpleNamespace

import pytest

from app import core


def test_parse_label_is_exact_and_ambiguity_safe():
    labels = ["payment terms", "governing law", "agreements", "binding effects"]
    assert core.parse_label("  PAYMENT TERMS.\n", labels) == "payment terms"
    assert core.parse_label("Category: governing law — selected from clause", labels) == "governing law"
    assert core.parse_label("Binding Agreements", labels) == "<unknown>"
    assert core.parse_label("a label that does not exist", labels) == "<unknown>"


def test_directory_size(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"12345")
    assert core.directory_size(tmp_path) == 8


def test_load_rows_reports_missing_preparation(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare"):
        core.load_rows("test")


def test_json_writer_is_machine_readable(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    core._json(destination, {"ok": True})
    assert json.loads(destination.read_text()) == {"ok": True}


def test_prepare_dataset_preserves_official_splits_and_stratifies(monkeypatch, tmp_path):
    labels = ["Payment", "Term", "Law"]
    class FakeSplit(list):
        features = {"label": SimpleNamespace(names=labels)}
    def make_split(prefix, size):
        return FakeSplit([{"text": f"{prefix} clause {i}", "label": i % 3} for i in range(size)])
    fake = {"train": make_split("train", 60), "validation": make_split("validation", 30),
            "test": make_split("test", 30)}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: fake))
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    manifest = core.prepare_dataset(limit=30, seed=3)
    assert manifest["splits"] == {"train": 24, "validation": 3, "test": 3}
    assert manifest["classes"] == 3
    assert set(manifest["split_label_distribution"]["train"].values()) == {8}
    assert core.prepare_dataset(limit=30, seed=3) == manifest
    loaded = {name: core.load_rows(name) for name in ("train", "validation", "test")}
    assert all(row["source_split"] == split for split, values in loaded.items() for row in values)
    texts = [{row["text"] for row in values} for values in loaded.values()]
    assert not any(texts[i] & texts[j] for i in range(3) for j in range(i + 1, 3))


def test_stratified_cap_is_repeatable_and_proportional():
    rows = [{"label": label, "i": i} for label, count in ((0, 90), (1, 10))
            for i in range(count)]
    first = core._stratified_cap(rows, 20, 11)
    second = core._stratified_cap(rows, 20, 11)
    assert first == second
    assert {label: sum(row["label"] == label for row in first) for label in (0, 1)} == {0: 18, 1: 2}


def test_classification_metrics_include_macro_weighted_and_confusion():
    result = core.classification_metrics(["A", "A", "B"], ["A", "B", "B"], ["A", "B", "C"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert result["weighted_f1"] == pytest.approx(2 / 3)
    assert result["per_class"]["C"]["support"] == 0
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="empty"):
        core.classification_metrics([], [], ["A"])


def test_benchmark_statistics_and_baseline_comparison():
    stats = core.summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert stats["mean_latency_seconds"] == pytest.approx(2.5)
    assert stats["median_latency_seconds"] == pytest.approx(2.5)
    assert stats["p95_latency_seconds"] == 4.0
    assert stats["throughput_per_second"] == pytest.approx(0.4)
    rows = [{"name":"original","size_bytes":100,"mean_latency_seconds":2.0,"accuracy":0.8,"macro_f1":0.7,"weighted_f1":0.8},
            {"name":"dynamic","size_bytes":40,"mean_latency_seconds":1.0,"accuracy":0.75,"macro_f1":0.65,"weighted_f1":0.75}]
    compared = core.compare_efficiency(rows)[1]
    assert compared["compression_ratio"] == pytest.approx(2.5)
    assert compared["size_reduction_percent"] == pytest.approx(60)
    assert compared["speedup"] == pytest.approx(2)
    assert compared["accuracy_change_percentage_points"] == pytest.approx(-5)
    assert compared["macro_f1_change_percentage_points"] == pytest.approx(-5)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        return {"Legal clause: ": [3], "\nCategory:": [4], "example clause": [5, 6],
                " Class A": [7, 8], " Class B": [7, 9], " Class C": [10]}[text]
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids[:max_length] if truncation and max_length is not None else ids}


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_ctx=32,
                                     n_embd=16, n_layer=1, n_head=2)).eval()


def test_closed_set_prediction_scores_all_label_sequences():
    import torch
    model = _tiny_gpt2()
    predicted, scores = core.score_labels(model, TinyTokenizer(), torch.device("cpu"),
                                          "example clause", ["Class A", "Class B", "Class C"],
                                          max_length=16, candidate_batch_size=2)
    assert predicted in {"Class A", "Class B", "Class C"}
    assert set(scores) == {"Class A", "Class B", "Class C"}
    assert all(torch.isfinite(torch.tensor(score)) for score in scores.values())


def test_gpt2_conv1d_conversion_preserves_outputs_and_dynamic_reload(tmp_path):
    import torch
    model = _tiny_gpt2()
    inputs = torch.tensor([[1, 5, 6, 4]])
    with torch.inference_mode():
        original = model(input_ids=inputs).logits
    converted = core.convert_gpt2_conv1d(model)
    core._protect_tied_lm_head(model)
    with torch.inference_mode():
        converted_logits = model(input_ids=inputs).logits
    assert len(converted) == 4
    torch.testing.assert_close(original, converted_logits, atol=1e-5, rtol=1e-5)
    quantized = core._quantized_gpt2(model)
    dynamic_names = [name for name, module in quantized.named_modules()
                     if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
    assert len(dynamic_names) == len(converted)
    state_file = tmp_path / "state.pt"
    torch.save(quantized.state_dict(), state_file)
    reloaded = core._new_dynamic_model(model.config)
    reloaded.load_state_dict(torch.load(state_file, map_location="cpu", weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        actual = reloaded(input_ids=inputs).logits
    torch.testing.assert_close(actual, quantized(input_ids=inputs).logits)


def test_static_qnnpack_projection_calibration_and_reload():
    import torch
    model = _tiny_gpt2()
    tokenizer = TinyTokenizer()
    rows = [{"text": "example clause"}]
    static = core._new_static_model(model.config, tokenizer, rows)
    with torch.inference_mode():
        logits = static(input_ids=torch.tensor([[1, 3, 5, 6, 4]]),
                        attention_mask=torch.ones(1, 5, dtype=torch.long), use_cache=False).logits
    assert torch.isfinite(logits).all()
    static_layers = [module for module in static.modules() if isinstance(module, torch.ao.nn.quantized.Linear)]
    assert len(static_layers) == 4
    reloaded = core._new_static_model(model.config, tokenizer, rows)
    reloaded.load_state_dict(static.state_dict())
    with torch.inference_mode():
        loaded_logits = reloaded(input_ids=torch.tensor([[1, 3, 5, 6, 4]]),
                                 attention_mask=torch.ones(1, 5, dtype=torch.long), use_cache=False).logits
    torch.testing.assert_close(loaded_logits, logits)
