"""Streamlit view over measured artifacts; absent runs are shown as not yet measured."""
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Legal Tagger Monitor", page_icon="⚖️", layout="wide")
st.title("⚖️ Quantized Legal Tagger Monitor")
st.caption("Measured local model size, held-out quality, and CPU inference performance")

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

bench = read_json(ARTIFACTS / "benchmark.json")
eval_base = read_json(ARTIFACTS / "evaluation_original.json")
if not bench or not bench.get("results"):
    st.info("No benchmark results yet. Prepare data, train, quantize, and benchmark with `python -m app.cli pipeline`.")
    st.stop()
results = bench["results"]
names = [x["name"] for x in results]
selected = st.selectbox("Inspect model", names)
row = next(x for x in results if x["name"] == selected)
cols = st.columns(4)
cols[0].metric("Model size", f"{row['size_bytes']/1024**2:.1f} MiB")
cols[1].metric("Compression", f"{row['compression_ratio']:.2f}×" if row.get("compression_ratio") else "—")
cols[2].metric("Accuracy", f"{row['accuracy']:.3f}" if row.get("accuracy") is not None else "—")
cols[3].metric("Mean latency", f"{row['mean_latency_seconds']*1000:.1f} ms" if row.get("mean_latency_seconds") is not None else "—")

st.subheader("Model comparison")
st.dataframe([{**x, "size_mib":round(x["size_bytes"]/1024**2,2), "mean_latency_ms":round(x["mean_latency_seconds"]*1000,2) if x.get("mean_latency_seconds") else None} for x in results], width="stretch", hide_index=True)
chart_rows = [{"Model":x["name"],"Size (MiB)":x["size_bytes"]/1024**2,"Mean latency (ms)":x["mean_latency_seconds"]*1000 if x.get("mean_latency_seconds") is not None else None,"Accuracy":x.get("accuracy")} for x in results]
left,right=st.columns(2)
left.subheader("Size and latency"); left.bar_chart(chart_rows,x="Model",y=["Size (MiB)","Mean latency (ms)"],width="stretch")
right.subheader("Accuracy comparison"); right.bar_chart(chart_rows,x="Model",y="Accuracy",width="stretch")
if eval_base and eval_base.get("confusion_matrix"):
    labels=eval_base["labels"]
    st.subheader("Original model confusion matrix")
    st.caption("Rows are true labels; columns are predicted labels. Full LEDGAR contains many classes; use the CSV/JSON artifact for search and filtering.")
    import pandas as pd
    st.dataframe(pd.DataFrame(eval_base["confusion_matrix"],index=labels,columns=labels),width="stretch")
st.caption(f"Runtime: {bench.get('hardware',{}).get('platform','unknown')} · Split: {bench.get('split','unknown')} · timings are local measured wall-clock values.")
