# Phase 3 Project Audit

Audit completed against commit `b9c7560` before Phase 3 changes. Existing Phase 1 and Phase 2 artifacts were left in place.

## Existing and verified functionality

- Python CLI supports health, dataset preparation, training/checkpoint resume, evaluation, dynamic and static quantization, benchmark, dashboard, and full pipeline commands.
- Dataset preparation uses official LEDGAR train/validation/test splits, deterministic capped sampling, and split-separated outputs.
- The model uses pretrained GPT-2 with closed-set conditional label scoring; evaluations save aggregate/per-class metrics, confusion matrix, and prediction CSVs.
- Dynamic and static QNNPACK paths quantize GPT-2 projection layers after checking Conv1D-to-Linear conversion. Saved quantized models are reloadable.
- Benchmark report records three models, 50 test inputs, 20 warmups, 3 sessions × 100 iterations, CPU environment, latency summaries, size, quality on the benchmark subset, and throughput.
- Existing dashboard loads recorded experiments, compares models, explores class/evaluation data, and offers existing report downloads.
- Phase 2 unit tests previously passed (14 tests). Dashboard AppTest previously rendered with no exceptions and the live Streamlit health endpoint returned `200 ok`.
- `artifacts/phase2` contains all three model folders, training metadata, three 300-example held-out evaluation reports and prediction files, quantization reports, and benchmark JSON/CSV. Large model/checkpoint outputs are ignored by Git.

## Measured Phase 2 baseline verified from artifacts

Full test-set accuracy: original 69.33%, dynamic 67.67%, static 68.67% (300 rows each). The separate benchmark subset reports model sizes 502,588,018 / 247,863,111 / 247,905,858 bytes; dynamic size reduction 50.68%, mean latency 395.20 ms vs 406.63 ms baseline, and static mean latency 407.10 ms. Thus the measured dynamic run was 2.81% lower latency; static was 0.12% higher latency. These are results on the recorded Mac CPU only.

## Issues addressed in Phase 3

- Dashboard previously lacked a clear overview, explicit precomputed/demo status, explanatory quantization context, and a comparison JSON export. It also did not surface precision/recall consistently in the comparison table.
- No one-command precomputed-results demo or compact portable report bundle existed. Phase 3 adds both.
- No final academic report, portable reproducibility guide, or programmatic report-figure generator existed.
- README had project details but no direct submission/report artifacts.

## Limitations

- Benchmark is small (50 distinct examples) and CPU-specific; no process-isolated RSS measurement exists. Benchmark-subset quality differs from the separate 300-example evaluation and must be labeled accordingly.
- Exact Apple SoC model was not captured (processor metadata is `arm`).
- Eager `torch.ao.quantization` emits deprecation warnings in the installed PyTorch. Static quantization covers only selected projection modules; most of GPT-2 remains floating point.
- The experiment fine-tunes a small 3,000-example subset over two epochs; it is a reproducible engineering demonstration, not a production legal advice system.
- Repository has no configured Git remote; saved GitHub CLI credentials were invalid at the last check. Local commit is possible, push is not currently configured.

## Submission assessment

After Phase 3 report, charts, demo command, dashboard improvements, and verification are completed, the repository is suitable for a classroom submission and local demonstration. It should be presented as an experimental ML systems project with the limitations above, not as a production legal classifier.
