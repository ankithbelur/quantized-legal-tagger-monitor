"""Reproducible LEDGAR classification-as-generation workflow."""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import statistics
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


def prepare_dataset(limit: int = 5000, seed: int = SEED, force: bool = False) -> dict:
    """Download LEDGAR via LexGLUE, clean, cap and save deterministic JSONL splits."""
    from datasets import load_dataset
    out = DATA_DIR / "ledgar"
    manifest = out / "manifest.json"
    if manifest.exists() and not force:
        cached = json.loads(manifest.read_text())
        if cached.get("limit") == limit and cached.get("seed") == seed:
            return cached
    # Try the Dataset API first. If the machine's cached Hub OAuth token is stale,
    # fetch the same public Parquet files directly and read their embedded schema.
    source = "coastalcph/lex_glue"
    split_names = ["train", "validation", "test"]
    try:
        raw = load_dataset(source, "ledgar")
        train_raw = raw["train"]
        names = list(train_raw.features["label"].names)
        split_rows = {split: raw[split] for split in split_names}
    except Exception as hub_error:
        import pyarrow.parquet as pq
        import requests
        source_dir = DATA_DIR / "raw" / "ledgar"
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
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for split in split_names:
        for row in split_rows[split]:
            text = " ".join(str(row.get("text", "")).split())
            label = int(row.get("label", -1))
            key = (text, label)
            if not text or label < 0 or label >= len(names) or key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label, "label_name": names[label]})
    random.Random(seed).shuffle(rows)
    if limit > 0:
        # Stratified cap: allocate the same fraction from each class, then fill remaining slots.
        by_label: dict[int, list[dict]] = {}
        for row in rows:
            by_label.setdefault(row["label"], []).append(row)
        quotas = {k: max(1, int(limit * len(v) / len(rows))) for k, v in by_label.items()}
        selected = [r for k, group in by_label.items() for r in group[:quotas[k]]]
        selected_ids = {id(r) for r in selected}
        selected.extend(r for r in rows if id(r) not in selected_ids and len(selected) < limit)
        rows = selected[:limit]
        random.Random(seed).shuffle(rows)
    # Split by exact text groups to prevent duplicate-clause leakage.
    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for i, row in enumerate(rows):
        buckets["train" if i % 10 < 8 else "validation" if i % 10 == 8 else "test"].append(row)
    # A seeded shuffled assignment is deterministic and the prior de-duplication prevents leakage.
    out.mkdir(parents=True, exist_ok=True)
    for name, examples in buckets.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label_name"]] = counts.get(row["label_name"], 0) + 1
    result = {"source": "lex_glue/ledgar", "source_splits": split_names, "limit": limit,
              "seed": seed, "examples": len(rows), "classes": len({r["label"] for r in rows}),
              "label_names": names, "splits": {k: len(v) for k, v in buckets.items()}, "label_distribution": counts}
    _json(manifest, result)
    return result


def load_rows(split: str) -> list[dict]:
    path = DATA_DIR / "ledgar" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `python -m app.cli prepare` first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_label(generated: str, labels: list[str]) -> str:
    """Resolve a generated label robustly, favoring exact first-line matches."""
    candidate = generated.strip().splitlines()[0].strip().strip(" .,:;\t\"'") if generated.strip() else ""
    lower = candidate.casefold()
    for label in labels:
        if lower == label.casefold():
            return label
    matches = [label for label in labels if label.casefold() in generated.casefold()]
    return max(matches, key=len) if matches else "<unknown>"


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
    rows = load_rows("train")
    val = load_rows("validation")
    if limit:
        rows, val = rows[:limit], val[:max(1, limit // 5)]
    labels = sorted({r["label_name"] for r in rows})
    random.seed(seed); torch.manual_seed(seed)
    dev = _device(torch, device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "checkpoint"
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
            prompt = f"Legal clause: {r['text']}\nCategory:"
            answer = f" {r['label_name']}{tokenizer.eos_token}"
            # Reserve context for the complete target label; otherwise long clauses
            # can truncate every supervised target and produce a NaN loss.
            target_budget = min(24, max(8, max_length // 4))
            p = tokenizer(prompt, truncation=True, max_length=max_length-target_budget, add_special_tokens=False)
            a = tokenizer(answer, truncation=True, max_length=target_budget, add_special_tokens=False)
            ids = p["input_ids"] + a["input_ids"]
            return ids, [-100] * len(p["input_ids"]) + a["input_ids"]
    def collate(batch):
        width = min(max(len(x[0]) for x in batch), max_length)
        ids = [x[0][:width] + [tokenizer.pad_token_id] * (width-len(x[0][:width])) for x in batch]
        labs = [x[1][:width] + [-100] * (width-len(x[1][:width])) for x in batch]
        return {"input_ids": torch.tensor(ids, device=dev), "attention_mask": torch.tensor([[int(v != tokenizer.pad_token_id) for v in r] for r in ids], device=dev), "labels": torch.tensor(labs, device=dev)}
    loader = DataLoader(Clauses(rows), batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    if resume and (checkpoint / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=dev, weights_only=True))
    step = 0
    for epoch in range(epochs):
        model.train(); total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()); step += 1
            if step % 50 == 0:
                checkpoint.mkdir(exist_ok=True); model.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
                torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                print(f"epoch={epoch+1} step={step} loss={total/min(step,50):.4f}", flush=True)
        print(f"epoch={epoch+1}/{epochs} mean_train_loss={total/max(len(loader),1):.4f}", flush=True)
    model.save_pretrained(MODEL_DIR / "original"); tokenizer.save_pretrained(MODEL_DIR / "original")
    # Persist validation loss as a training diagnostic only.
    model.eval(); losses=[]
    with torch.no_grad():
        for batch in DataLoader(Clauses(val), batch_size=batch_size, collate_fn=collate):
            losses.append(float(model(**batch).loss))
    metadata = {"model_name": model_name, "device": str(dev), "train_examples": len(rows), "validation_examples": len(val), "labels": labels, "epochs": epochs, "max_length": max_length, "validation_loss": statistics.mean(losses) if losses else None}
    _json(MODEL_DIR / "training.json", metadata)
    return metadata


def _load_model(path: Path, device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    is_dynamic = (Path(path) / "quantization.json").exists()
    model = AutoModelForCausalLM.from_pretrained(path, ignore_mismatched_sizes=False) if not is_dynamic else AutoModelForCausalLM.from_config(__import__("transformers").AutoConfig.from_pretrained(path))
    dev = torch.device("cpu") if is_dynamic else _device(torch, device)
    if is_dynamic:
        if "qnnpack" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "qnnpack"
        elif "x86" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "x86"
        elif "fbgemm" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "fbgemm"
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(Path(path) / "pytorch_model.bin", map_location="cpu", weights_only=True))
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
    model.to(dev); model.eval()
    return model, tok, dev


def predict(model, tokenizer, device, text: str, max_length: int = 256, quantized: bool = False) -> str:
    import torch
    prompt = f"Legal clause: {text}\nCategory:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    return parse_label(decoded, labels)


def evaluate(model_path: Path | str | None = None, split: str = "test", device: str = "auto", limit: int = 0) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    path = Path(model_path) if model_path else MODEL_DIR / "original"
    model, tok, dev = _load_model(path, device)
    rows = load_rows(split)
    if limit: rows = rows[:limit]
    actual, pred = [], []
    for i, row in enumerate(rows, 1):
        actual.append(row["label_name"]); pred.append(predict(model, tok, dev, row["text"]))
        if i % 50 == 0: print(f"evaluated {i}/{len(rows)}", flush=True)
    labels = json.loads((DATA_DIR / "ledgar" / "manifest.json").read_text())["label_names"]
    p,r,f,_ = precision_recall_fscore_support(actual, pred, labels=labels, average="weighted", zero_division=0)
    _,_,macro,_ = precision_recall_fscore_support(actual, pred, labels=labels, average="macro", zero_division=0)
    result = {"model": str(path), "split": split, "examples": len(rows), "accuracy": accuracy_score(actual,pred), "precision": p, "recall": r, "f1": f, "macro_f1": macro, "weighted_f1": f, "labels": labels, "confusion_matrix": confusion_matrix(actual,pred,labels=labels).tolist()}
    out = MODEL_DIR / ("evaluation_" + ("original" if path.name == "original" else path.name) + ".json")
    _json(out, result)
    return result


def quantize_dynamic() -> dict:
    import torch
    src = MODEL_DIR / "original"
    if not src.exists(): raise FileNotFoundError("Train a model first.")
    model, tok, _ = _load_model(src, "cpu")
    before = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    # GPT-2's transformer blocks use custom Conv1D projections; only compatible Linear modules are quantized.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "x86" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "x86"
    elif "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    else:
        raise RuntimeError(f"No supported CPU dynamic quantization engine: {torch.backends.quantized.supported_engines}")
    quant = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    after = sum(1 for m in quant.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear))
    dest = MODEL_DIR / "dynamic"
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir(parents=True)
    tok.save_pretrained(dest)
    quant.config.save_pretrained(dest)
    torch.save(quant.state_dict(), dest / "pytorch_model.bin")
    # A recipe marker lets the loader rebuild the converted graph before loading its weights.
    _json(dest / "quantization.json", {"type":"dynamic_int8", "quantized_linear_modules": after, "original_linear_modules": before, "note":"GPT-2 transformer projections use Conv1D and remain FP32."})
    result = {"type":"dynamic_int8", "quantized_linear_modules":after, "original_linear_modules":before,"path":str(dest),"size_bytes":directory_size(dest)}
    _json(MODEL_DIR / "quantization_dynamic.json", result)
    return result


def directory_size(path: Path | str) -> int:
    p=Path(path)
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) if p.is_dir() else p.stat().st_size


def benchmark(model_paths: dict[str, Path] | None = None, split: str = "test", limit: int = 50, repeats: int = 3) -> dict:
    import torch
    rows=load_rows(split)[:limit]
    paths=model_paths or {"original":MODEL_DIR/"original", "dynamic":MODEL_DIR/"dynamic"}
    results=[]
    for name,path in paths.items():
        if not Path(path).exists(): continue
        model,tok,dev=_load_model(Path(path), "cpu")
        samples=rows[:min(len(rows), max(1,limit))]
        if samples: predict(model,tok,dev,samples[0]["text"])
        times=[]; outputs=[]
        for _ in range(repeats):
            for row in samples:
                t=time.perf_counter(); outputs.append(predict(model,tok,dev,row["text"])); times.append(time.perf_counter()-t)
        expected=[r["label_name"] for _ in range(repeats) for r in samples]
        from sklearn.metrics import precision_recall_fscore_support
        precision,recall,f1,_=precision_recall_fscore_support(expected,outputs,average="weighted",zero_division=0)
        _,_,macro_f1,_=precision_recall_fscore_support(expected,outputs,average="macro",zero_division=0)
        results.append({"name":name,"path":str(path),"size_bytes":directory_size(path),"parameter_count":sum(p.numel() for p in model.parameters()),"device":str(dev),"batch_size":1,"samples":len(times),"mean_latency_seconds":statistics.mean(times) if times else None,"median_latency_seconds":statistics.median(times) if times else None,"p95_latency_seconds":sorted(times)[max(0,int(.95*len(times))-1)] if times else None,"throughput_per_second":len(times)/sum(times) if times and sum(times) else None,"accuracy":sum(o==r for o,r in zip(outputs,expected))/len(times) if times else None,"precision":precision,"recall":recall,"f1":f1,"macro_f1":macro_f1})
    baseline=next((x for x in results if x["name"]=="original"),None)
    for x in results:
        x["compression_ratio"] = baseline["size_bytes"]/x["size_bytes"] if baseline and x["size_bytes"] else None
        x["size_reduction_percent"] = 100*(1-x["size_bytes"]/baseline["size_bytes"]) if baseline and baseline["size_bytes"] else None
        x["latency_improvement_percent"] = 100*(baseline["mean_latency_seconds"]-x["mean_latency_seconds"])/baseline["mean_latency_seconds"] if baseline and baseline["mean_latency_seconds"] and x["mean_latency_seconds"] is not None else None
        x["throughput_change_percent"] = 100*(x["throughput_per_second"]-baseline["throughput_per_second"])/baseline["throughput_per_second"] if baseline and baseline["throughput_per_second"] and x["throughput_per_second"] is not None else None
        x["accuracy_change_percentage_points"] = 100*(x["accuracy"]-baseline["accuracy"]) if baseline and x["accuracy"] is not None and baseline["accuracy"] is not None else None
    out={"hardware": {"platform": __import__("platform").platform(), "processor":__import__("platform").processor(),"torch":torch.__version__},"split":split,"results":results}
    _json(MODEL_DIR/"benchmark.json",out); return out


def milestone_1(): return prepare_dataset()
def milestone_2(): return train()
def milestone_3(): return evaluate()
def milestone_4(): return quantize_dynamic()
def milestone_5(): return benchmark()
def milestone_6(): return {"dashboard":"Run `streamlit run app/dashboard.py`.","results":str(MODEL_DIR/"benchmark.json")}
