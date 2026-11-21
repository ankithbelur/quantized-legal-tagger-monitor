# Reproducibility and Demo Guide

## Fast demo (precomputed results)

Use Python 3.12 and install dependencies once:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.cli health
python -m app.cli demo
```

`demo` checks that benchmark and all three evaluation JSONs exist, then launches Streamlit. It prefers the complete local `artifacts/phase2` run; compact copies of the actual reports and prediction CSVs are included under `reports/demo/` so a clean checkout can demonstrate the measured results without downloading data, training, or running inference. Model weights and data remain excluded from Git. Missing reports produce a CLI error naming them.

## Reproduce from data and pretrained checkpoint

The following commands download LEDGAR and the pretrained GPT-2 checkpoint as needed, so they require network access. Outputs are isolated from Phase 1:

```bash
export LEGAL_TAGGER_DATA_DIR=data/phase2
export LEGAL_TAGGER_MODEL_DIR=artifacts/phase2
python -m app.cli prepare --limit 3000
python -m app.cli train --limit 3000 --epochs 2 --batch-size 4 --max-length 160 --device cpu
python -m app.cli evaluate --split test
python -m app.cli quantize
python -m app.cli evaluate --model-path artifacts/phase2/dynamic --split test
python -m app.cli static-quantize --calibration-limit 128
python -m app.cli evaluate --model-path artifacts/phase2/static --split test
python -m app.cli benchmark --limit 50 --warmup 20 --iterations 100 --sessions 3
python -m app.reporting
```

Seed 42 is used by the dataset/training pipeline. The 3,000-row cap allocates 2,400/300/300 across official train/validation/test splits. Training has two epochs, batch size four, max prompt length 160, and learning rate 5e-5. Static calibration uses up to 128 training rows. Benchmarking uses 50 distinct held-out test inputs, batch size one, 20 warmups, and three sessions of 100 timed calls. CPU measurements depend on hardware and load.

`train` reuses a matching completed run and can resume interrupted checkpoints. Do not pass `--no-resume` unless intentionally starting a new run. A full reproduction requires substantial disk space and may take hours on CPU. To launch the dashboard after a run use `python -m app.cli dashboard`.

## Outputs and portability

`LEGAL_TAGGER_DATA_DIR` and `LEGAL_TAGGER_MODEL_DIR` can point to any writable relative or absolute directory. Defaults are repository-local `data/` and `artifacts/`. The repository does not require a user-specific absolute path. Reports and charts are under `reports/`; metrics, models, and data are written under the configured artifact/data roots. `.gitignore` excludes datasets, model weights, checkpoints, local environments, and caches. Keep `training.json`, evaluation JSON/CSV, quantization JSON, and benchmark JSON/CSV with the model bundle to preserve measured evidence.

Run checks with `pytest -q`. Tests use synthetic fixtures and small models; they do not retrain GPT-2 or download LEDGAR.
