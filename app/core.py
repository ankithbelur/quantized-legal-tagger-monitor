"""Reproducible LEDGAR classification-as-generation workflow."""
from __future__ import annotations

import csv
import json
import os
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SEED = 42
DATA_DIR = Path(os.environ.get("LEGAL_TAGGER_DATA_DIR", "data"))
MODEL_DIR = Path(os.environ.get("LEGAL_TAGGER_MODEL_DIR", "artifacts"))
MODEL_NAME = "openai-community/gpt2"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install project dependencies with `pip install -r requirements.txt`.") from exc


def _stratified_cap(rows: list[dict], cap: int, seed: int) -> list[dict]:
    """Sample a capped source split in proportion to its class distribution."""
    if cap <= 0 or cap >= len(rows):
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        return selected
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row["label"]), []).append(row)
    quotas = {label: (cap * len(group) / len(rows)) for label, group in groups.items()}
    allocation = {label: min(len(groups[label]), int(quota)) for label, quota in quotas.items()}
    remaining = cap - sum(allocation.values())
    order = sorted(groups, key=lambda label: (-(quotas[label] % 1), label))
    while remaining:
        eligible = [label for label in order if allocation[label] < len(groups[label])]
        if not eligible:
            break
        for label in eligible:
            if remaining == 0:
                break
            allocation[label] += 1
            remaining -= 1
    rng = random.Random(seed)
    selected = []
    for label, group in groups.items():
        rng.shuffle(group)
        selected.extend(group[:allocation[label]])
    rng.shuffle(selected)
    return selected


def prepare_dataset(limit: int = 3000, seed: int = SEED, force: bool = False) -> dict:
    """Cache LEDGAR's official, independent splits with reproducible stratified caps.

    ``limit`` is a total cap allocated 80/10/10 across the official train,
    validation, and test splits. No records are moved between source splits.
    """
    from datasets import load_dataset
    out = DATA_DIR / "ledgar"
    split_names = ["train", "validation", "test"]
    manifest = out / "manifest.json"
    if manifest.exists() and not force:
        cached = json.loads(manifest.read_text())
        if (cached.get("limit") == limit and cached.get("seed") == seed
                and all((out / f"{split}.jsonl").exists() for split in split_names)):
            return cached
    # Try the Dataset API first. If the machine's cached Hub OAuth token is stale,
    # fetch the same public Parquet files directly and read their embedded schema.
    source = "coastalcph/lex_glue"
    try:
        cached_sources = [DATA_DIR / "raw" / "ledgar", DATA_DIR.parent / "raw" / "ledgar"]
        if any(all((candidate / f"{split}-00000-of-00001.parquet").exists() for split in split_names)
               for candidate in cached_sources):
            raise ConnectionError("Public source Parquet files are already cached locally")
        raw = load_dataset(source, "ledgar")
        train_raw = raw["train"]
        names = list(train_raw.features["label"].names)
        split_rows = {split: raw[split] for split in split_names}
    except Exception as hub_error:
        import pyarrow.parquet as pq
        import requests
        source_dir = DATA_DIR / "raw" / "ledgar"
        shared_cache = DATA_DIR.parent / "raw" / "ledgar"
        expected = [f"{split}-00000-of-00001.parquet" for split in split_names]
        if all((shared_cache / filename).exists() for filename in expected):
            source_dir = shared_cache
        source_dir.mkdir(parents=True, exist_ok=True)
        split_rows = {}
        names = None
        for split in split_names:
            filename = f"{split}-00000-of-00001.parquet"
            cached_file = source_dir / filename
            if not cached_file.exists():
                url = f"https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/ledgar/{filename}"
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                cached_file.write_bytes(response.content)
            parquet = pq.ParquetFile(cached_file)
            if names is None:
                import json as _json_module
                metadata = _json_module.loads(parquet.metadata.metadata[b"huggingface"].decode())
                names = metadata["info"]["features"]["label"]["names"]
            split_rows[split] = parquet.read().to_pylist()
        if not names:
            raise RuntimeError("LEDGAR did not provide label names in its Parquet schema") from hub_error
    if not names:
        raise ValueError("LEDGAR dataset did not expose named class labels.")
    by_split: dict[str, list[dict]] = {split: [] for split in split_names}
    seen_text: set[str] = set()
    for split in split_names:
        for row in split_rows[split]:
            text = " ".join(str(row.get("text", "")).split())
            label = int(row.get("label", -1))
            if not text or label < 0 or label >= len(names) or text in seen_text:
                continue
            seen_text.add(text)
            by_split[split].append({"text": text, "label": label, "label_name": names[label], "source_split": split})
    if limit > 0:
        if limit < 3:
            raise ValueError("A capped dataset requires at least 3 examples for train/validation/test")
        train_cap = max(1, int(limit * 0.8))
        validation_cap = max(1, int(limit * 0.1))
        test_cap = max(1, limit - train_cap - validation_cap)
        while train_cap + validation_cap + test_cap > limit:
            if train_cap > 1:
                train_cap -= 1
            elif validation_cap > 1:
                validation_cap -= 1
            else:
                test_cap -= 1
        caps = {"train": train_cap, "validation": validation_cap, "test": test_cap}
    else:
        caps = {split: 0 for split in split_names}
    buckets = {split: _stratified_cap(by_split[split], caps[split], seed + i)
               for i, split in enumerate(split_names)}
    rows = [row for split in split_names for row in buckets[split]]
    # Verify the official-split and text-level leakage invariants before writing.
    split_texts = [{row["text"] for row in buckets[split]} for split in split_names]
    if any(split_texts[i] & split_texts[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Duplicate clause text leaked across LEDGAR splits")
    out.mkdir(parents=True, exist_ok=True)
    for name, examples in buckets.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples))
    counts = {split: {} for split in split_names}
    for split, examples in buckets.items():
        for row in examples:
            counts[split][row["label_name"]] = counts[split].get(row["label_name"], 0) + 1
    result = {"source": "coastalcph/lex_glue/ledgar", "source_splits": split_names, "limit": limit,
              "seed": seed, "examples": len(rows), "classes": len({r["label"] for r in rows}),
              "label_names": names, "splits": {k: len(v) for k, v in buckets.items()},
              "split_label_distribution": counts, "duplicate_text_policy": "first occurrence wins in source order; no cross-split duplicates",
              "sampling": "per-official-split proportional stratified cap; total cap allocated 80/10/10"}
    _json(manifest, result)
    return result


def load_rows(split: str) -> list[dict]:
    path = DATA_DIR / "ledgar" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `python -m app.cli prepare` first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_token_ids(tokenizer, text: str, max_length: int) -> list[int]:
    """Encode the clause while always retaining the Category prompt suffix."""
    prefix = tokenizer.encode("Legal clause: ", add_special_tokens=False)
    suffix = tokenizer.encode("\nCategory:", add_special_tokens=False)
    budget = max_length - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError("max_length is too small for the classification prompt")
    body = tokenizer(text, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
    return prefix + body + suffix


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def classification_metrics(actual: list[str], predicted: list[str], labels: list[str]) -> dict:
    """Calculate closed-set classification metrics with a fixed label universe."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    if not actual:
        raise ValueError("cannot calculate metrics for an empty evaluation set")
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="weighted", zero_division=0)
    per_p, per_r, per_f, per_n = precision_recall_fscore_support(
        actual, predicted, labels=labels, average=None, zero_division=0)
    per_class = {label: {"precision": float(per_p[i]), "recall": float(per_r[i]),
                         "f1": float(per_f[i]), "support": int(per_n[i])}
                 for i, label in enumerate(labels)}
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f),
            "weighted_precision": float(weighted_p), "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f), "per_class": per_class,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist()}


def parse_label(generated: str, labels: list[str]) -> str:
    """Parse a free generation without accepting an incidental label substring.

    Exact labels and a documented label prefix followed by explanatory punctuation
    are accepted. Ambiguous or unrelated text remains ``<unknown>``.
    """
    import re
    candidate = generated.strip().splitlines()[0].strip() if generated.strip() else ""
    normalize = lambda value: " ".join(re.sub(r"[.,:;\"'`]+", " ", value).casefold().split())
    exact = [label for label in labels if normalize(candidate) == normalize(label)]
    if len(exact) == 1:
        return exact[0]
    candidate = re.sub(r"^(?:(?:the )?(?:predicted )?(?:class|label|category)(?: is)?\s*[:=\-]\s*)", "", candidate, flags=re.IGNORECASE)
    matches = []
    for label in labels:
        pattern = r"^" + r"\s+".join(re.escape(part) for part in label.split())
        match = re.match(pattern + r"(?=$|[\s.,;:!?—\-])", candidate, flags=re.IGNORECASE)
        if match:
            tail = candidate[match.end():].strip(" \t.,;:!?—-")
            if not tail or re.match(r"^(?:because|since|as|based on|is the best fit|selected from|selected because|chosen because|model predicts)\b", tail, flags=re.IGNORECASE):
                matches.append(label)
    if len(matches) == 1:
        return matches[0]
    # A valid label preceded by a short explicit class wrapper is also safe.
    matches = []
    for label in labels:
        pattern = r"(?:^|\b(?:class|label|category)\s*(?:is|:|=)\s*)" + r"\s+".join(re.escape(part) for part in label.split()) + r"$"
        if re.search(pattern, candidate, flags=re.IGNORECASE):
            matches.append(label)
    return matches[0] if len(matches) == 1 else "<unknown>"


def _device(torch, requested: str | None = None):
    if requested and requested != "auto":
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(limit: int = 0, epochs: int = 1, batch_size: int = 2, max_length: int = 256,
          model_name: str = MODEL_NAME, device: str = "auto", seed: int = SEED, resume: bool = True) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if epochs < 1 or batch_size < 1 or max_length < 32:
        raise ValueError("epochs and batch_size must be positive and max_length must be at least 32")
    rows = load_rows("train")
    val = load_rows("validation")
    if limit:
        rows = _stratified_cap(rows, limit, seed)
        val = _stratified_cap(val, max(1, limit // 5), seed + 1)
    if not rows or not val:
        raise ValueError("Training and validation splits must both contain examples")
    data_manifest = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())
    labels = data_manifest["label_names"]
    random.seed(seed)
    torch.manual_seed(seed)
    dev = _device(torch, device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "checkpoint"
    training_path = MODEL_DIR / "training.json"
    run_config = {"model_name": model_name, "train_examples": len(rows), "validation_examples": len(val),
                  "epochs": epochs, "batch_size": batch_size, "max_length": max_length,
                  "seed": seed, "learning_rate": 5e-5, "sampler": "inverse-square-root class frequency",
                  "dataset_source": data_manifest.get("source"), "dataset_seed": data_manifest.get("seed"),
                  "dataset_limit": data_manifest.get("limit"), "dataset_splits": data_manifest.get("splits")}
    if resume and training_path.exists() and (MODEL_DIR / "original" / "config.json").exists():
        previous = json.loads(training_path.read_text())
        if previous.get("status") == "completed" and previous.get("run_config") == run_config:
            return previous
    if not resume and checkpoint.exists():
        shutil.rmtree(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint) if resume and checkpoint.exists() else model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(dev)
    class Clauses(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            r = self.data[i]
            target = tokenizer.encode(" " + r["label_name"], add_special_tokens=False) + [tokenizer.eos_token_id]
            prompt = prompt_token_ids(tokenizer, r["text"], max_length - len(target))
            ids = prompt + target
            return ids, [-100] * len(prompt) + target
    def collate(batch):
        width = min(max(len(x[0]) for x in batch), max_length)
        ids, labs, masks = [], [], []
        for tokens, targets in batch:
            tokens, targets = tokens[:width], targets[:width]
            pad = width - len(tokens)
            ids.append(tokens + [tokenizer.pad_token_id] * pad)
            labs.append(targets + [-100] * pad)
            masks.append([1] * len(tokens) + [0] * pad)
        return {"input_ids": torch.tensor(ids, device=dev), "attention_mask": torch.tensor(masks, device=dev), "labels": torch.tensor(labs, device=dev)}
    counts = {label: sum(r["label_name"] == label for r in rows) for label in labels}
    example_weights = [1.0 / math.sqrt(max(1, counts[r["label_name"]])) for r in rows]
    dataset = Clauses(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text()) if resume and state_path.exists() else {"epoch": 0, "next_batch": 0, "step": 0}
    step = int(state.get("step", 0))
    if resume and (checkpoint / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=dev, weights_only=True))
        rng_file = checkpoint / "torch_rng.pt"
        if rng_file.exists():
            torch.set_rng_state(torch.load(rng_file, map_location="cpu", weights_only=True))
    def save_checkpoint(epoch: int, next_batch: int) -> None:
        checkpoint.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        for filename, value in (("optimizer.pt", optimizer.state_dict()), ("torch_rng.pt", torch.get_rng_state())):
            temporary = checkpoint / (filename + ".tmp")
            torch.save(value, temporary)
            temporary.replace(checkpoint / filename)
        _json(state_path, {"epoch": epoch, "next_batch": next_batch, "step": step, "run_config": run_config})
    epoch_losses = []
    start_epoch = int(state.get("epoch", 0))
    for epoch in range(start_epoch, epochs):
        sampler = torch.utils.data.WeightedRandomSampler(
            example_weights, num_samples=len(dataset), replacement=True,
            generator=torch.Generator().manual_seed(seed + epoch))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate)
        skip_batches = int(state.get("next_batch", 0)) if epoch == start_epoch else 0
        model.train(); total = 0.0; batch_count = 0
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_batches:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()); batch_count += 1; step += 1
            if step % 100 == 0:
                save_checkpoint(epoch, batch_index + 1)
                print(f"epoch={epoch+1} step={step} loss={float(loss.detach()):.4f}", flush=True)
        if batch_count == 0 and epoch < epochs:
            raise RuntimeError("No training batches ran; checkpoint resume cursor exceeds this epoch")
        mean_loss = total / batch_count
        epoch_losses.append(mean_loss)
        save_checkpoint(epoch + 1, 0)
        print(f"epoch={epoch+1}/{epochs} mean_train_loss={mean_loss:.4f}", flush=True)
    model.save_pretrained(MODEL_DIR / "original"); tokenizer.save_pretrained(MODEL_DIR / "original")
    model.eval(); losses=[]
    with torch.no_grad():
        for batch in DataLoader(Clauses(val), batch_size=batch_size, collate_fn=collate):
            losses.append(float(model(**batch).loss))
    metadata = {"status": "completed", "run_config": run_config, "device": str(dev),
                "labels": labels, "training_label_counts": counts, "train_loss_by_epoch": epoch_losses,
                "validation_loss": statistics.mean(losses) if losses else None, "optimizer_steps": step,
                "checkpoint": str(checkpoint)}
    _json(training_path, metadata)
    return metadata


def convert_gpt2_conv1d(model) -> list[str]:
    """Replace GPT-2's transposed-weight Conv1D projections with equivalent Linear layers."""
    import torch
    from transformers.pytorch_utils import Conv1D
    converted = []
    for name, module in list(model.named_modules()):
        if type(module) is not Conv1D:
            continue
        input_features, output_features = module.weight.shape
        linear = torch.nn.Linear(input_features, output_features, bias=module.bias is not None)
        linear = linear.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(module.weight.T)
            if module.bias is not None:
                linear.bias.copy_(module.bias)
        linear.weight.requires_grad_(module.weight.requires_grad)
        if module.bias is not None:
            linear.bias.requires_grad_(module.bias.requires_grad)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)
        converted.append(name)
    return converted


def _protect_tied_lm_head(model):
    """Keep GPT-2's output projection in float because its weights alias the token embedding."""
    import torch
    class _FloatTiedLinear(torch.nn.Linear):
        pass
    old = model.lm_head
    head = _FloatTiedLinear(old.in_features, old.out_features, bias=old.bias is not None)
    head = head.to(device=old.weight.device, dtype=old.weight.dtype)
    head.weight = model.transformer.wte.weight
    if old.bias is not None:
        head.bias = old.bias
    model.lm_head = head
    return model


def _quantized_gpt2(model):
    import torch
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "x86" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "x86"
    elif "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    else:
        raise RuntimeError(f"No supported CPU dynamic quantization engine: {torch.backends.quantized.supported_engines}")
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _prepare_static_gpt2(model):
    """Wrap only converted projections; leave embeddings, norms, attention, and tied head float."""
    import torch
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f"Static QNNPACK is unavailable; supported engines: {torch.backends.quantized.supported_engines}")
    torch.backends.quantized.engine = "qnnpack"
    qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    wrapped = []
    for name, module in list(model.named_modules()):
        if type(module) is not torch.nn.Linear:
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        wrapper = torch.ao.quantization.QuantWrapper(module)
        wrapper.qconfig = qconfig
        setattr(parent, parts[-1], wrapper)
        wrapped.append(name)
    model.qconfig = None
    return torch.ao.quantization.prepare(model, inplace=False), wrapped


def _new_static_model(config, tokenizer, calibration_rows: list[dict]):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config).eval()
    convert_gpt2_conv1d(model)
    _protect_tied_lm_head(model)
    prepared, _ = _prepare_static_gpt2(model)
    prepared.eval()
    with torch.inference_mode():
        for row in calibration_rows:
            ids = torch.tensor([prompt_token_ids(tokenizer, row["text"], 160)], dtype=torch.long)
            prepared(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    converted = torch.ao.quantization.convert(prepared, inplace=False).eval()
    return converted


def _new_dynamic_model(config):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config)
    convert_gpt2_conv1d(model)
    _protect_tied_lm_head(model)
    return _quantized_gpt2(model)


def _load_model(path: Path, device: str = "auto"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = Path(path)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    marker = path / "quantization.json"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata.get("type") == "static_int8":
            calibration_rows = load_rows("train")[:1]
            model = _new_static_model(AutoConfig.from_pretrained(path), tok, calibration_rows)
        else:
            model = _new_dynamic_model(AutoConfig.from_pretrained(path))
        model.load_state_dict(torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True))
        dev = torch.device("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
        dev = _device(torch, device)
    model.to(dev)
    model.eval()
    return model, tok, dev


def score_labels(model, tokenizer, device, text: str, labels: list[str], max_length: int = 256,
                 candidate_batch_size: int = 8) -> tuple[str, dict[str, float]]:
    """Score closed-set labels by length-normalized conditional token likelihood.

    The clause prefix is computed once and its KV cache is reused across small
    candidate batches, avoiding 100 full-context model passes per prediction.
    """
    import torch
    if not labels:
        raise ValueError("At least one candidate label is required")
    prompt = prompt_token_ids(tokenizer, text, max_length)
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    sequences = [tokenizer.encode(" " + label, add_special_tokens=False) for label in labels]
    if any(not sequence for sequence in sequences):
        raise ValueError("Every candidate label must tokenize to at least one token")
    scores: dict[str, float] = {}
    with torch.inference_mode():
        context = model(input_ids=ids, attention_mask=attention, use_cache=True)
        first_log_probs = torch.log_softmax(context.logits[0, -1].float(), dim=-1)
        cache = context.past_key_values
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        for start in range(0, len(labels), candidate_batch_size):
            end = min(len(labels), start + candidate_batch_size)
            batch_sequences = sequences[start:end]
            batch_size = len(batch_sequences)
            current_scores = [float(first_log_probs[seq[0]]) for seq in batch_sequences]
            max_suffix = max(len(seq) - 1 for seq in batch_sequences)
            if max_suffix:
                suffix_ids = [seq[:-1] + [tokenizer.pad_token_id] * (max_suffix - len(seq) + 1)
                              for seq in batch_sequences]
                suffix_mask = [[1] * (len(seq) - 1) + [0] * (max_suffix - len(seq) + 1)
                               for seq in batch_sequences]
                suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
                mask_tensor = torch.tensor(suffix_mask, dtype=torch.long, device=device)
                full_attention = torch.cat([attention.expand(batch_size, -1), mask_tensor], dim=1)
                from transformers.cache_utils import DynamicCache
                repeated_cache = DynamicCache.from_legacy_cache(cache)
                repeated_cache.batch_repeat_interleave(batch_size)
                output = model(input_ids=suffix_tensor, attention_mask=full_attention,
                               past_key_values=repeated_cache, use_cache=False)
                log_probs = torch.log_softmax(output.logits.float(), dim=-1)
                for i, seq in enumerate(batch_sequences):
                    if len(seq) > 1:
                        positions = torch.arange(len(seq) - 1, device=device)
                        targets = torch.tensor(seq[1:], dtype=torch.long, device=device)
                        current_scores[i] += float(log_probs[i, positions, targets].sum())
            for offset, score in enumerate(current_scores):
                scores[labels[start + offset]] = score / len(batch_sequences[offset])
    prediction = max(labels, key=lambda label: (scores[label], -labels.index(label)))
    return prediction, scores


def _context_length() -> int:
    metadata = None
    try:
        metadata = json.loads((MODEL_DIR / "training.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return int(metadata.get("run_config", {}).get("max_length", metadata.get("max_length", 160))) if metadata else 160


def raw_generation(model, tokenizer, device, text: str, max_length: int | None = None) -> str:
    import torch
    max_length = max_length or _context_length()
    ids = torch.tensor([prompt_token_ids(tokenizer, text, max_length)], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=24, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()


def predict(model, tokenizer, device, text: str, max_length: int | None = None, labels: list[str] | None = None) -> str:
    max_length = max_length or _context_length()
    if labels is None:
        labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    return score_labels(model, tokenizer, device, text, labels, max_length)[0]


def evaluate(model_path: Path | str | None = None, split: str = "test", device: str = "auto", limit: int = 0) -> dict:
    path = Path(model_path) if model_path else MODEL_DIR / "original"
    model, tok, dev = _load_model(path, device)
    rows = load_rows(split)
    if limit: rows = rows[:limit]
    if not rows:
        raise ValueError(f"The {split} evaluation split contains no examples")
    labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    actual, pred, diagnostics = [], [], []
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        raw = raw_generation(model, tok, dev, row["text"])
        parsed = parse_label(raw, labels)
        prediction = predict(model, tok, dev, row["text"], labels=labels)
        valid = prediction in labels
        actual.append(row["label_name"])
        pred.append(prediction)
        diagnostics.append({"text": row["text"], "ground_truth": row["label_name"], "raw_generation": raw,
                            "raw_parsed_label": parsed, "parse_success": parsed != "<unknown>",
                            "prediction": prediction, "valid_prediction": valid,
                            "correct": prediction == row["label_name"],
                            "error_reason": "correct" if prediction == row["label_name"] else
                            ("invalid_closed_set_prediction" if not valid else "closed_set_label_mismatch")})
        if i % 50 == 0:
            print(f"evaluated {i}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(actual, pred, labels)
    key = "original" if path.name == "original" else path.name
    csv_path = MODEL_DIR / f"predictions_{key}_{split}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(diagnostics[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    result = {"model": str(path), "split": split, "examples": len(rows),
              "valid_predictions": sum(d["valid_prediction"] for d in diagnostics),
              "invalid_predictions": sum(not d["valid_prediction"] for d in diagnostics),
              "parse_failures": sum(not d["parse_success"] for d in diagnostics),
              "prediction_strategy": "closed-set mean conditional token log probability",
              "evaluation_seconds": elapsed, **metrics,
              "precision": metrics["weighted_precision"], "recall": metrics["weighted_recall"],
              "f1": metrics["weighted_f1"],
              "label_distribution": {label: actual.count(label) for label in labels if label in actual},
              "prediction_distribution": {label: pred.count(label) for label in labels if label in pred},
              "labels": labels,
              "predictions_csv": str(csv_path)}
    out = MODEL_DIR / f"evaluation_{key}_{split}.json"
    _json(out, result)
    return result


def quantize_dynamic() -> dict:
    import torch
    src = MODEL_DIR / "original"
    if not src.exists(): raise FileNotFoundError("Train a model first.")
    model, tok, _ = _load_model(src, "cpu")
    model.eval()
    rows = load_rows("train")
    probe = torch.tensor([prompt_token_ids(tok, rows[0]["text"], 128)], dtype=torch.long)
    with torch.inference_mode():
        reference_logits = model(input_ids=probe, attention_mask=torch.ones_like(probe)).logits
    converted_names = convert_gpt2_conv1d(model)
    _protect_tied_lm_head(model)
    with torch.inference_mode():
        converted_logits = model(input_ids=probe, attention_mask=torch.ones_like(probe)).logits
    conversion_error = float((reference_logits - converted_logits).abs().max())
    if conversion_error > 1e-4:
        raise RuntimeError(f"Conv1D-to-Linear conversion changed outputs (max abs error {conversion_error:.6g})")
    quant = _quantized_gpt2(model)
    with torch.inference_mode():
        quantized_logits = quant(input_ids=probe, attention_mask=torch.ones_like(probe)).logits
    max_error = float((converted_logits - quantized_logits).abs().max())
    mean_error = float((converted_logits - quantized_logits).abs().mean())
    quantized_names = [name for name, module in quant.named_modules()
                       if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
    dest = MODEL_DIR / "dynamic"
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tok.save_pretrained(dest)
    quant.config.save_pretrained(dest)
    torch.save(quant.state_dict(), dest / "pytorch_model.bin")
    metadata = {"type": "dynamic_int8", "backend": torch.backends.quantized.engine,
                "converted_conv1d_count": len(converted_names), "converted_conv1d_modules": converted_names,
                "quantized_linear_count": len(quantized_names), "quantized_linear_modules": quantized_names,
                "excluded_modules": {"lm_head": "shares token embedding weights; kept FP32 to avoid duplicate tied storage"},
                "conversion_max_abs_logit_error": conversion_error,
                "quantization_max_abs_logit_error": max_error,
                "quantization_mean_abs_logit_error": mean_error,
                "quantization": "PyTorch eager dynamic qint8 weights-only Linear conversion"}
    _json(dest / "quantization.json", metadata)
    # Verify the persisted graph and packed state before reporting a successful artifact.
    reloaded, reloaded_tok, reloaded_device = _load_model(dest, "cpu")
    labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    smoke_prediction = predict(reloaded, reloaded_tok, reloaded_device, rows[0]["text"], labels=labels)
    result = {**metadata, "path": str(dest), "size_bytes": directory_size(dest),
              "reloaded_inference_prediction": smoke_prediction, "reload_inference_verified": smoke_prediction in labels}
    _json(MODEL_DIR / "quantization_dynamic.json", result)
    if not result["reload_inference_verified"]:
        raise RuntimeError("Quantized model reload did not return a valid closed-set label")
    return result


def quantize_static(calibration_limit: int = 128, max_length: int = 160) -> dict:
    """Static qint8 PTQ for GPT-2 projection layers, calibrated on train rows only."""
    import torch
    from transformers import AutoConfig
    src = MODEL_DIR / "original"
    if not src.exists():
        raise FileNotFoundError("Train a model first")
    if calibration_limit < 1:
        raise ValueError("calibration_limit must be positive")
    model, tokenizer, _ = _load_model(src, "cpu")
    train_rows = load_rows("train")
    calibration_rows = _stratified_cap(train_rows, min(calibration_limit, len(train_rows)), SEED + 99)
    model.eval()
    probe = torch.tensor([prompt_token_ids(tokenizer, train_rows[0]["text"], max_length)], dtype=torch.long)
    with torch.inference_mode():
        reference_logits = model(input_ids=probe, attention_mask=torch.ones_like(probe), use_cache=False).logits
    converted_names = convert_gpt2_conv1d(model)
    _protect_tied_lm_head(model)
    with torch.inference_mode():
        linear_logits = model(input_ids=probe, attention_mask=torch.ones_like(probe), use_cache=False).logits
    conversion_error = float((reference_logits - linear_logits).abs().max())
    if conversion_error > 1e-4:
        raise RuntimeError(f"Conv1D-to-Linear conversion changed outputs (max abs error {conversion_error:.6g})")
    prepared, wrapped_names = _prepare_static_gpt2(model)
    prepared.eval()
    calibration_started = time.perf_counter()
    with torch.inference_mode():
        for row in calibration_rows:
            ids = torch.tensor([prompt_token_ids(tokenizer, row["text"], max_length)], dtype=torch.long)
            prepared(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    calibration_seconds = time.perf_counter() - calibration_started
    quant = torch.ao.quantization.convert(prepared, inplace=False).eval()
    with torch.inference_mode():
        static_logits = quant(input_ids=probe, attention_mask=torch.ones_like(probe), use_cache=False).logits
    max_error = float((linear_logits - static_logits).abs().max())
    mean_error = float((linear_logits - static_logits).abs().mean())
    quantized_names = [name for name, module in quant.named_modules()
                       if isinstance(module, torch.ao.nn.quantized.Linear)]
    dest = MODEL_DIR / "static"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tokenizer.save_pretrained(dest)
    quant.config.save_pretrained(dest)
    torch.save(quant.state_dict(), dest / "pytorch_model.bin")
    metadata = {"type": "static_int8", "backend": torch.backends.quantized.engine,
                "calibration_split": "train", "calibration_examples": len(calibration_rows),
                "calibration_seconds": calibration_seconds, "converted_conv1d_count": len(converted_names),
                "converted_conv1d_modules": converted_names, "wrapped_projection_modules": wrapped_names,
                "quantized_linear_count": len(quantized_names), "quantized_linear_modules": quantized_names,
                "excluded_modules": {"embeddings": "not a supported static quantized module in this wrapper path",
                                     "layer_norm_and_attention": "left in float32",
                                     "lm_head": "tied to token embeddings; kept in float32"},
                "conversion_max_abs_logit_error": conversion_error,
                "quantization_max_abs_logit_error": max_error,
                "quantization_mean_abs_logit_error": mean_error,
                "quantization": "PyTorch eager static qint8 Linear projections with QNNPACK activation observers"}
    _json(dest / "quantization.json", metadata)
    # Rebuild, convert and reload, using a train-only record to initialize the graph.
    reloaded, reloaded_tok, reloaded_device = _load_model(dest, "cpu")
    labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    smoke_prediction = predict(reloaded, reloaded_tok, reloaded_device, train_rows[0]["text"], max_length=max_length, labels=labels)
    result = {**metadata, "path": str(dest), "size_bytes": directory_size(dest),
              "reloaded_inference_prediction": smoke_prediction, "reload_inference_verified": smoke_prediction in labels}
    _json(MODEL_DIR / "quantization_static.json", result)
    if not result["reload_inference_verified"]:
        raise RuntimeError("Static quantized model reload failed inference verification")
    return result


def directory_size(path: Path | str) -> int:
    p=Path(path)
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) if p.is_dir() else p.stat().st_size


def model_parameter_count(model) -> int:
    """Count float parameters plus packed INT8 Linear logical parameter counts."""
    import torch
    count = sum(parameter.numel() for parameter in model.parameters())
    quantized_types = (torch.ao.nn.quantized.dynamic.Linear, torch.ao.nn.quantized.Linear)
    for module in model.modules():
        if isinstance(module, quantized_types):
            count += int(module.in_features) * int(module.out_features)
            bias = module.bias() if callable(getattr(module, "bias", None)) else getattr(module, "bias", None)
            if bias is not None:
                count += int(module.out_features)
    return count


def summarize_latencies(timings: list[float]) -> dict:
    if not timings or any(value < 0 for value in timings):
        raise ValueError("latency samples must be nonempty and nonnegative")
    ordered = sorted(timings)
    return {"mean_latency_seconds": statistics.mean(timings),
            "median_latency_seconds": statistics.median(timings),
            "p95_latency_seconds": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "latency_stddev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "throughput_per_second": len(timings) / sum(timings) if sum(timings) else None}


def compare_efficiency(results: list[dict]) -> list[dict]:
    """Add baseline-relative size, speed, and quality changes."""
    baseline = next((item for item in results if item.get("name") == "original"), None)
    for item in results:
        item["compression_ratio"] = baseline["size_bytes"] / item["size_bytes"] if baseline and item["size_bytes"] else None
        item["size_reduction_percent"] = 100 * (1 - item["size_bytes"] / baseline["size_bytes"]) if baseline and baseline["size_bytes"] else None
        item["latency_change_percent"] = 100 * (item["mean_latency_seconds"] - baseline["mean_latency_seconds"]) / baseline["mean_latency_seconds"] if baseline and baseline["mean_latency_seconds"] else None
        item["speedup"] = baseline["mean_latency_seconds"] / item["mean_latency_seconds"] if baseline and item["mean_latency_seconds"] else None
        item["accuracy_change_percentage_points"] = 100 * (item["accuracy"] - baseline["accuracy"]) if baseline and item.get("accuracy") is not None else None
        item["macro_f1_change_percentage_points"] = 100 * (item["macro_f1"] - baseline["macro_f1"]) if baseline and item.get("macro_f1") is not None else None
        item["weighted_f1_change_percentage_points"] = 100 * (item["weighted_f1"] - baseline["weighted_f1"]) if baseline and item.get("weighted_f1") is not None else None
    return results


def benchmark(model_paths: dict[str, Path] | None = None, split: str = "test", limit: int = 50,
              warmup: int = 20, iterations: int = 100, sessions: int = 3) -> dict:
    import platform
    import subprocess
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    if limit < 1 or warmup < 0 or iterations < 1 or sessions < 1:
        raise ValueError("limit/iterations/sessions must be positive and warmup must be nonnegative")
    rows = load_rows(split)
    samples = rows[:limit]
    if not samples:
        raise ValueError(f"The {split} split contains no benchmark inputs")
    paths = model_paths or {"original": MODEL_DIR / "original", "dynamic": MODEL_DIR / "dynamic", "static": MODEL_DIR / "static"}
    results = []
    labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    for name, path in paths.items():
        path = Path(path)
        if not path.exists():
            continue
        model, tok, dev = _load_model(path, "cpu")
        for i in range(warmup):
            predict(model, tok, dev, samples[i % len(samples)]["text"], labels=labels)
        timings, predictions, expected, session_means = [], [], [], []
        for session in range(sessions):
            session_times = []
            for iteration in range(iterations):
                row = samples[(session * iterations + iteration) % len(samples)]
                started = time.perf_counter()
                prediction = predict(model, tok, dev, row["text"], labels=labels)
                elapsed = time.perf_counter() - started
                timings.append(elapsed); session_times.append(elapsed)
                predictions.append(prediction); expected.append(row["label_name"])
            session_means.append(statistics.mean(session_times))
        weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
            expected, predictions, labels=labels, average="weighted", zero_division=0)
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
            expected, predictions, labels=labels, average="macro", zero_division=0)
        latency_stats = summarize_latencies(timings)
        results.append({"name": name, "path": str(path), "size_bytes": directory_size(path),
                        "parameter_count": model_parameter_count(model),
                        "device": str(dev), "batch_size": 1, "distinct_inputs": len(samples),
                        "warmup_iterations": warmup, "timed_iterations_per_session": iterations,
                        "sessions": sessions, "timed_iterations": len(timings),
                        **latency_stats,
                        "session_mean_latency_seconds": session_means,
                        "accuracy": sum(p == y for p, y in zip(predictions, expected)) / len(expected),
                        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f,
                        "weighted_precision": weighted_p, "weighted_recall": weighted_r,
                        "weighted_f1": weighted_f, "memory_bytes": None,
                        "memory_note": "Per-model RSS was not measured; no comparable process-isolated memory sampler is configured."})
        del model, tok
    compare_efficiency(results)
    processor = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            processor = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip() or processor
        except (OSError, subprocess.CalledProcessError):
            pass
    output = {"hardware": {"platform": platform.platform(), "processor": processor,
                           "python": platform.python_version(), "torch": torch.__version__,
                           "cpu_count": os.cpu_count(), "device": "cpu"},
              "split": split, "benchmark_config": {"distinct_inputs": len(samples), "warmup_iterations": warmup,
              "timed_iterations_per_session": iterations, "sessions": sessions, "batch_size": 1,
              "prediction_method": "closed-set mean conditional token log probability"}, "results": results}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _json(MODEL_DIR / "benchmark.json", output)
    if results:
        with (MODEL_DIR / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
    return output


def milestone_1(): return prepare_dataset()
def milestone_2(): return train()
def milestone_3(): return evaluate()
def milestone_4(): return quantize_dynamic()
def milestone_5(): return benchmark()
def milestone_6(): return {"dashboard":"Run `streamlit run app/dashboard.py`.","results":str(MODEL_DIR/"benchmark.json")}
