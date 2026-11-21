"""Generate report figures directly from the recorded Phase 2 benchmark."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "phase2" / "benchmark.json"
OUT = ROOT / "reports" / "figures"
COLORS = {"original": "#355070", "dynamic": "#2a9d8f", "static": "#e9a23b"}


def load_benchmark(path: Path = BENCHMARK) -> dict:
    """Load and minimally validate the measured benchmark artifact."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load benchmark report at {path}: {exc}") from exc
    if not isinstance(data.get("results"), list) or not data["results"]:
        raise ValueError(f"Benchmark report at {path} has no results")
    required = {"name", "size_bytes", "accuracy", "mean_latency_seconds", "throughput_per_second"}
    for row in data["results"]:
        if not required.issubset(row):
            raise ValueError(f"Benchmark result is missing fields: {sorted(required - row.keys())}")
    return data


def _bar_svg(title: str, ylabel: str, rows: list[tuple[str, float]], suffix: str, filename: str) -> Path:
    width, height = 1200, 720
    left, right, top, bottom = 130, 80, 110, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max((v for _, v in rows), default=1) or 1
    maximum *= 1.18
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{left}" y="58" font-family="Arial,sans-serif" font-size="32" font-weight="700" fill="#16324f">{title}</text>',
             f'<text x="28" y="{top + plot_h/2}" transform="rotate(-90 28 {top + plot_h/2})" font-family="Arial,sans-serif" font-size="19" fill="#52677d">{ylabel}</text>']
    for i in range(5):
        y = top + plot_h * i / 4
        val = maximum * (1 - i / 4)
        parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6edf3"/>',
                  f'<text x="{left-15}" y="{y+6:.1f}" text-anchor="end" font-family="Arial" font-size="16" fill="#52677d">{val:.2f}</text>']
    bw = plot_w / max(1, len(rows)) * .48
    for i, (name, value) in enumerate(rows):
        cx = left + plot_w * (i + .5) / len(rows)
        bh = value / maximum * plot_h
        x, y = cx-bw/2, top+plot_h-bh
        color = COLORS.get(name, "#64748b")
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="8" fill="{color}"/>',
                  f'<text x="{cx:.1f}" y="{y-12:.1f}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="#16324f">{value:.2f}{suffix}</text>',
                  f'<text x="{cx:.1f}" y="{top+plot_h+38}" text-anchor="middle" font-family="Arial" font-size="18" fill="#334e68">{name.title()}</text>']
    parts.append("</svg>")
    path = OUT / filename
    path.write_text("\n".join(parts))
    return path


def _scatter_svg(title: str, xlabel: str, ylabel: str, points: list[tuple[str, float, float]], filename: str) -> Path:
    width, height, left, right, top, bottom = 1200, 720, 130, 80, 110, 120
    pw, ph = width-left-right, height-top-bottom
    xs = [p[1] for p in points] or [0, 1]
    ys = [p[2] for p in points] or [0, 1]
    xmin, xmax = min(0, min(xs)), max(xs) or 1
    ymin, ymax = min(0, min(ys)), max(ys) or 1
    xmax, ymax = xmax * 1.18, ymax * 1.18
    xspan, yspan = xmax-xmin or 1, ymax-ymin or 1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="58" font-family="Arial" font-size="32" font-weight="700" fill="#16324f">{title}</text>',
           f'<text x="{width/2}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="19" fill="#52677d">{xlabel}</text>',
           f'<text x="28" y="{top+ph/2}" transform="rotate(-90 28 {top+ph/2})" font-family="Arial" font-size="19" fill="#52677d">{ylabel}</text>',
           f'<line x1="{left}" y1="{top+ph}" x2="{width-right}" y2="{top+ph}" stroke="#8da2b5"/>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#8da2b5"/>']
    for name, xv, yv in points:
        x = left+(xv-xmin)/xspan*pw
        y = top+ph-(yv-ymin)/yspan*ph
        out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="{COLORS.get(name, "#64748b")}"/>',
                f'<text x="{x+20:.1f}" y="{y+6:.1f}" font-family="Arial" font-size="18" fill="#334e68">{name.title()}</text>']
    path = OUT / filename
    path.write_text("\n".join(out+["</svg>"]))
    return path


def generate_figures(benchmark: dict | None = None, out: Path = OUT) -> list[Path]:
    """Create seven vector charts; values are read from the run report."""
    global OUT
    old_out = OUT
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    data = benchmark or load_benchmark()
    rows = data["results"]
    vals = lambda key, scale=1: [(r["name"], float(r[key]) * scale) for r in rows if r.get(key) is not None]
    reduction = [(r["name"], float(r.get("size_reduction_percent", 0))) for r in rows]
    accuracy_loss = [(r["name"], float(r.get("size_reduction_percent", 0)), max(0, -float(r.get("accuracy_change_percentage_points", 0)))) for r in rows]
    created = [
        _bar_svg("Accuracy on benchmark subset", "Accuracy (%)", vals("accuracy", 100), "%", "accuracy_comparison.svg"),
        _bar_svg("Serialized model size", "Size (MiB)", vals("size_bytes", 1 / 1024**2), " MiB", "model_size.svg"),
        _bar_svg("Model size reduction", "Reduction (%)", reduction, "%", "compression_reduction.svg"),
        _bar_svg("Mean inference latency", "Latency (ms)", vals("mean_latency_seconds", 1000), " ms", "latency_comparison.svg"),
        _bar_svg("Inference throughput", "Predictions / second", vals("throughput_per_second"), " /s", "throughput_comparison.svg"),
        _scatter_svg("Accuracy loss vs. compression", "Model size reduction (%)", "Accuracy loss (percentage points)", accuracy_loss, "accuracy_loss_compression.svg"),
        _scatter_svg("Accuracy and storage trade-off", "Model size (MiB)", "Accuracy (%)", [(r["name"], r["size_bytes"]/1024**2, 100*r["accuracy"]) for r in rows], "efficiency_tradeoff.svg"),
    ]
    OUT = old_out
    return created


if __name__ == "__main__":
    print("\n".join(str(p.relative_to(ROOT)) for p in generate_figures()))
