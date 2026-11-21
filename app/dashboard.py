"""Interactive view of stored, measured experiment results."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Legal Tagger Monitor", page_icon="⚖️", layout="wide")
st.title("⚖️ Quantized Legal Tagger Monitor")
st.caption("Measured held-out quality and CPU efficiency. All values come from local run artifacts.")


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def discover_runs():
    found = {}
    if ARTIFACTS.exists():
        for path in ARTIFACTS.rglob("benchmark.json"):
            run_id = str(path.parent.relative_to(ARTIFACTS)) or "phase1"
            report = read_json(path)
            if report and report.get("results"):
                found[run_id] = (path.parent, report)
    return found


runs = discover_runs()
if not runs:
    trained = list(ARTIFACTS.rglob("config.json")) if ARTIFACTS.exists() else []
    if trained:
        st.info("A model exists, but no completed benchmark report is available yet. Run `python -m app.cli benchmark`.")
    else:
        st.info("No trained model or benchmark report is available. Prepare data, train, evaluate, and benchmark to populate this dashboard.")
    st.stop()

run_names = sorted(runs)
run_id = st.selectbox("Experiment", run_names, index=run_names.index("phase2") if "phase2" in runs else 0)
run_dir, benchmark = runs[run_id]
results = benchmark["results"]
models = {item["name"]: item for item in results}
base = models.get("original")
compressed = [item for item in results if item["name"] != "original"]
best = min(compressed, key=lambda item: item["size_bytes"]) if compressed else None

st.subheader("Efficiency summary")
cards = st.columns(4)
cards[0].metric("Original size", f"{base['size_bytes'] / 1024**2:.1f} MiB" if base else "Not measured")
cards[1].metric("Smallest compressed size", f"{best['size_bytes'] / 1024**2:.1f} MiB" if best else "Not available")
cards[2].metric("Size reduction", f"{best['size_reduction_percent']:.1f}%" if best and best.get("size_reduction_percent") is not None else "Not available")
cards[3].metric("Compression ratio", f"{best['compression_ratio']:.2f}×" if best and best.get("compression_ratio") else "Not available")

cards = st.columns(4)
cards[0].metric("Original accuracy", f"{base['accuracy']:.3f}" if base and base.get("accuracy") is not None else "Not measured")
cards[1].metric("Compressed accuracy", f"{best['accuracy']:.3f}" if best and best.get("accuracy") is not None else "Not available")
cards[2].metric("Original mean latency", f"{base['mean_latency_seconds'] * 1000:.1f} ms" if base else "Not measured")
cards[3].metric("Compressed latency", f"{best['mean_latency_seconds'] * 1000:.1f} ms" if best and best.get("mean_latency_seconds") is not None else "Not available")
if best and best.get("speedup"):
    label = f"{best['speedup']:.2f}× faster" if best["speedup"] > 1 else f"{1 / best['speedup']:.2f}× slower"
    st.caption(f"Measured speed change for {best['name']}: {label}. Accuracy change: {best.get('accuracy_change_percentage_points', 0):+.2f} percentage points.")

st.subheader("Model comparison")
table = []
for item in results:
    table.append({"Model": item["name"], "Quantization": item.get("quantization", "FP32 baseline" if item["name"] == "original" else item["name"]),
                  "Size (MiB)": round(item["size_bytes"] / 1024**2, 2), "Accuracy": item.get("accuracy"),
                  "Macro F1": item.get("macro_f1"), "Weighted F1": item.get("weighted_f1"),
                  "Mean latency (ms)": round(item["mean_latency_seconds"] * 1000, 2),
                  "Median (ms)": round(item["median_latency_seconds"] * 1000, 2),
                  "P95 (ms)": round(item["p95_latency_seconds"] * 1000, 2),
                  "Latency σ (ms)": round(item["latency_stddev_seconds"] * 1000, 2) if item.get("latency_stddev_seconds") is not None else None,
                  "Throughput/s": round(item["throughput_per_second"], 2),
                  "Size reduction %": item.get("size_reduction_percent"), "Speedup": item.get("speedup")})
st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

chart_col, metric_col = st.columns([2, 1])
metric_key = metric_col.selectbox("Compare metric", ["size_bytes", "mean_latency_seconds", "accuracy", "macro_f1", "weighted_f1", "throughput_per_second"])
metric_names = {"size_bytes": "Size (MiB)", "mean_latency_seconds": "Mean latency (ms)", "accuracy": "Accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "throughput_per_second": "Throughput / second"}
chart_values = []
for item in results:
    value = item.get(metric_key)
    if value is None:
        continue
    if metric_key == "size_bytes": value /= 1024**2
    if metric_key == "mean_latency_seconds": value *= 1000
    chart_values.append({"Model": item["name"], metric_names[metric_key]: value})
chart_col.subheader(metric_names[metric_key])
chart_col.bar_chart(chart_values, x="Model", y=metric_names[metric_key], width="stretch")

st.subheader("Accuracy and efficiency trade-off")
tradeoff = [{"Model": item["name"], "Mean latency (ms)": item.get("mean_latency_seconds", 0) * 1000,
             "Accuracy": item.get("accuracy"), "Size (MiB)": item["size_bytes"] / 1024**2}
            for item in results if item.get("accuracy") is not None]
if tradeoff:
    st.scatter_chart(tradeoff, x="Mean latency (ms)", y="Accuracy", size="Size (MiB)", color="Model", width="stretch")

st.subheader("Benchmark configuration and hardware")
st.json({"hardware": benchmark.get("hardware", {}), "benchmark_config": benchmark.get("benchmark_config", {}), "split": benchmark.get("split")})

quant = read_json(run_dir / "quantization_dynamic.json") or read_json(run_dir / "dynamic" / "quantization.json")
static = read_json(run_dir / "quantization_static.json") or read_json(run_dir / "static" / "quantization.json")
with st.expander("Quantization details and status"):
    if quant:
        st.json(quant)
    else:
        st.info("Dynamic quantization results are not available for this experiment.")
    if static:
        st.warning(static.get("reason", static.get("status", "Static quantization status recorded.")))
    else:
        st.caption("No static quantization report is present for this experiment.")

eval_files = sorted(run_dir.glob("evaluation_*_test.json"))
if not eval_files and run_dir == ARTIFACTS:
    eval_files = sorted(ARTIFACTS.glob("evaluation_*.json"))
if eval_files:
    eval_labels = [p.stem.removeprefix("evaluation_").removesuffix("_test") for p in eval_files]
    selected_model = st.selectbox("Explore evaluation", eval_labels)
    eval_path = eval_files[eval_labels.index(selected_model)]
    evaluation = read_json(eval_path)
    if evaluation:
        st.subheader(f"{selected_model}: held-out evaluation")
        st.caption(f"{evaluation.get('examples', 0)} examples · {evaluation.get('valid_predictions', 'not recorded')} valid closed-set predictions · {evaluation.get('parse_failures', 'not recorded')} raw-generation parse failures · {evaluation.get('evaluation_seconds', 0):.1f}s")
        per_class = evaluation.get("per_class", {})
        if per_class:
            class_name = st.selectbox("Inspect class", sorted(per_class))
            st.json({"class": class_name, **per_class[class_name]})
            st.dataframe(pd.DataFrame([{"Label": label, **values} for label, values in per_class.items()]), width="stretch", hide_index=True)
        if evaluation.get("confusion_matrix"):
            labels = evaluation["labels"]
            confusion = pd.DataFrame(evaluation["confusion_matrix"], index=labels, columns=labels)
            with st.expander("Confusion matrix"):
                st.dataframe(confusion, width="stretch")
        pred_file = run_dir / f"predictions_{selected_model}_test.csv"
        if not pred_file.exists() and run_dir == ARTIFACTS:
            pred_file = ARTIFACTS / f"predictions_{selected_model}_test.csv"
        predictions = read_csv(pred_file)
        if not predictions.empty:
            st.subheader("Prediction diagnostics")
            st.dataframe(predictions, width="stretch", hide_index=True)
            st.download_button("Download prediction CSV", pred_file.read_bytes(), file_name=pred_file.name, mime="text/csv")
        st.download_button("Download evaluation JSON", eval_path.read_bytes(), file_name=eval_path.name, mime="application/json")

st.download_button("Download benchmark JSON", (run_dir / "benchmark.json").read_bytes(), file_name=f"{run_id}_benchmark.json", mime="application/json")
bench_csv = run_dir / "benchmark.csv"
if bench_csv.exists():
    st.download_button("Download benchmark CSV", bench_csv.read_bytes(), file_name=f"{run_id}_benchmark.csv", mime="text/csv")
