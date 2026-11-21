import json
from pathlib import Path

import pytest

from app import reporting


def test_benchmark_loader_rejects_missing_and_corrupt_files(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Cannot load"):
        reporting.load_benchmark(missing)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{")
    with pytest.raises(ValueError, match="Cannot load"):
        reporting.load_benchmark(corrupt)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"results": []}))
    with pytest.raises(ValueError, match="no results"):
        reporting.load_benchmark(empty)


def test_figures_are_generated_from_supplied_measurements(tmp_path):
    report = {"results": [
        {"name": "original", "size_bytes": 100, "accuracy": 0.8,
         "mean_latency_seconds": 1, "throughput_per_second": 1,
         "macro_f1": 0.7, "size_reduction_percent": 0,
         "accuracy_change_percentage_points": 0},
        {"name": "dynamic", "size_bytes": 50, "accuracy": 0.75,
         "mean_latency_seconds": 0.9, "throughput_per_second": 1.1,
         "macro_f1": 0.65, "size_reduction_percent": 50,
         "accuracy_change_percentage_points": -5},
    ]}
    paths = reporting.generate_figures(report, tmp_path)
    assert len(paths) == 7
    assert all(path.is_file() and "<svg" in path.read_text() for path in paths)
    assert "75.00%" in (tmp_path / "accuracy_comparison.svg").read_text()


def test_dashboard_displays_precomputed_phase2_and_downloads():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "dashboard.py").run()
    assert not app.exception
    assert any("Phase 2 reports loaded" in str(element.value) for element in app.success)
    labels = [button.label for button in app.get("download_button")]
    assert any("Download benchmark" in label for label in labels)
    assert any("model comparison" in label for label in labels)


def test_demo_command_launches_streamlit_for_existing_artifacts(monkeypatch):
    from app import cli

    calls = []
    monkeypatch.setattr("subprocess.run", lambda command, check: calls.append((command, check)))
    cli.main(["demo"])
    assert calls and calls[0][1] is True
    assert calls[0][0][-2:] == ["run", "app/dashboard.py"]
