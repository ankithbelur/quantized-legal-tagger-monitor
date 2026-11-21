# Quantized Legal Clause Tagging: Quality and CPU Efficiency Study

## Abstract

This project evaluates a fine-tuned GPT-2 classifier for 100-way legal contract provision tagging and compares its original floating-point model with partial dynamic and static INT8 quantization. The recorded Phase 2 experiment uses 3,000 LEDGAR examples, preserves the official train/validation/test separation, and evaluates three models on 300 held-out test clauses. Original, dynamic, and static accuracies were 69.33%, 67.67%, and 68.67%. In a separate CPU benchmark of 50 held-out clauses, quantized model files were about 50.7% smaller. Dynamic INT8 was 2.81% lower in mean latency; static INT8 was 0.12% higher. The small subset and one-machine results support an engineering demonstration, not production claims.

## 1. Introduction

Legal contracts contain provisions with distinct legal functions. Automatic clause categorization can help organize documents, but a classifier's output requires expert review and is not legal advice. This project studies whether quantizing selected transformer projections can reduce local inference storage while retaining useful classification quality.

## 2. Problem statement and objectives

The engineering problem is to run a 100-label legal clause tagger locally and measure model size, quality, and CPU latency. Objectives are to (1) use split-separated data, (2) fine-tune a pretrained language model, (3) evaluate with class-aware metrics, (4) produce loadable dynamic/static INT8 variants, and (5) preserve reproducible benchmark evidence.

## 3. Background

INT8 represents values using 8-bit integers. Quantization can reduce stored parameter size and may change execution speed, but affects numerical precision. Dynamic quantization stores selected weights in integer form and handles activations during inference. Static quantization estimates activation ranges from calibration data and quantizes both weights and selected activations. This implementation only quantizes selected projection layers; it is not a fully INT8 GPT-2.

## 4. System architecture

The pipeline prepares official LEDGAR splits, tokenizes prompt/label pairs, fine-tunes GPT-2, scores all supported labels using mean conditional token log probability, writes metrics and prediction diagnostics, converts GPT-2 Conv1D projections to equivalent Linear layers, creates dynamic/static variants, benchmarks CPU inference, and serves saved reports via Streamlit. See the architecture diagram in the project README.

## 5. Dataset and preprocessing

The dataset is LexGLUE LEDGAR, a contract provision classification task. The verified run used a deterministic 3,000-example cap across official splits: 2,400 train, 300 validation, and 300 test; the prepared label space contains 100 classes. Exact duplicate clause text is removed in source order. Capped sampling is proportional to source split class distributions with seed 42. Training uses inverse-square-root class-frequency sampling. Test data is reserved for final evaluation and benchmark inputs; static calibration uses training data.

## 6. Model architecture and training

The base model is `openai-community/gpt2` (GPT-2 small configuration, 124,439,808 parameters). It is fine-tuned as prompt-to-label text generation. Training configuration: two epochs, batch size four, maximum sequence length 160, learning rate 5e-5, seed 42, CPU. This run completed 1,200 optimizer steps. Recorded mean training loss was 0.8012 for epoch one and 0.3418 for epoch two; validation loss was 0.3381. These values come from `artifacts/phase2/training.json`.

At evaluation, the classifier scores each known label by mean conditional token log probability and chooses the highest-scoring label. This closed-set decision always maps to a supported class. Greedy raw text is retained as a diagnostic, not used as the scored prediction.

## 7. Evaluation methodology

All three models were evaluated on the same 300-row test split. Metrics include accuracy, macro and weighted precision/recall/F1, per-class scores, confusion matrix, and prediction records. Raw-generation parse failures are separately reported and do not invalidate the closed-set prediction. The headline results below are read directly from the three evaluation JSON artifacts.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Original GPT-2 | 69.33% | 0.5568 | 0.5789 | 0.5496 | 0.6773 |
| Dynamic INT8 | 67.67% | 0.5469 | 0.5624 | 0.5360 | 0.6643 |
| Static INT8 | 68.67% | 0.5513 | 0.5715 | 0.5362 | 0.6711 |

## 8. Quantization methodology

GPT-2 uses Conv1D projections with transposed weight layout. The project transposes each projection into `nn.Linear` and verifies equivalence before quantization. Forty-eight transformer projections are quantized. Embeddings, layer normalization, attention/KV cache operations, and tied language-model head remain floating point. Dynamic quantization uses QNNPACK qint8 weights. Static quantization wraps the projection modules with activation observers and calibrates on up to 128 train examples. Both variants were saved, reloaded, and exercised with inference.

## 9. Benchmark methodology

The benchmark report records macOS 27.2.1 arm64, Python 3.12.8, PyTorch 2.15.0, CPU execution, and 18 logical CPUs. The CPU model identifier was not available beyond `arm`. Three models used the same first 50 distinct test inputs, batch size one, 20 warmup iterations, and three sessions of 100 timed predictions each (300 timed calls per model). Latency mean, median, p95, standard deviation, throughput, serialized model size, and quality on that subset are recorded. No comparable process-isolated RSS measurement was configured.

## 10. Experimental efficiency results

These values are from `artifacts/phase2/benchmark.json` and `benchmark.csv`; quality here uses the 50-example benchmark subset, not the 300-example table above.

| Model | Serialized size | Reduction | Mean latency | Throughput | Subset accuracy | Subset macro F1 | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original GPT-2 | 479.31 MiB | — | 406.63 ms | 2.459/s | 70% | 0.2449 | 1.000× |
| Dynamic INT8 | 236.35 MiB | 50.68% | 395.20 ms | 2.530/s | 70% | 0.2449 | 1.029× |
| Static INT8 | 236.39 MiB | 50.67% | 407.10 ms | 2.456/s | 66% | 0.2263 | 0.999× |

## 11. Results analysis and trade-offs

Both quantized variants reduced serialized model size by approximately half. Dynamic INT8 produced a 2.81% lower measured mean latency and no accuracy change on the small benchmark subset, while its full test accuracy was 1.67 percentage points below original. Static INT8 was 0.12% slower by mean latency; its subset accuracy was four points lower and full-test accuracy 0.67 points lower. These small timing differences should be interpreted cautiously because a single local benchmark is sensitive to system load and cache effects. The 50-example subset is intentionally distinguished from the 300-example evaluation. Quantization also introduced substantial logit differences in the recorded probe, even though aggregate held-out quality changed modestly.

## 12. Limitations

- Only 3,000 examples and two training epochs were used; class coverage is limited relative to the full corpus.
- Benchmark quality uses only 50 examples, and per-model RSS was not measured.
- Results are specific to one Mac CPU environment; exact SoC metadata was unavailable.
- Partial quantization leaves much of the model in floating point, so the size/speed results do not represent a fully integer model.
- PyTorch eager quantization APIs emit deprecation warnings and may need migration in a future runtime.
- This is a research/engineering classifier demonstration, not a legal decision system. Human review is required.

## 13. Future work

Evaluate on a larger untouched test set, compare multiple random seeds, add repeated process-isolated memory measurements, capture exact hardware identifiers, investigate TorchAO-supported quantization, and compare with smaller encoder models. Any quality improvement should be measured on a held-out split and reviewed for class-level regressions.

## 14. Conclusion

The measured run shows that partial INT8 quantization cut serialized model size by about 50.7%. Dynamic quantization was modestly faster in this run; static quantization was approximately latency-neutral. Full test accuracy remained within 1.7 percentage points of baseline for both variants. These findings support a local demonstration of the accuracy/storage/latency trade-off, with the subset and hardware limitations stated above.

## 15. Reproducibility

The precomputed-result demo is launched with `python -m app.cli demo`. For a full rerun, see [`reproducibility_guide.md`](reproducibility_guide.md) and the step-by-step commands in the README. Data and model weights are generated locally and excluded from Git; preserve the `artifacts/phase2` experiment bundle to reproduce the demo without retraining.

## References

1. Chalkidis et al., “LexGLUE: A Benchmark Dataset for Legal Language Understanding in English,” ACL-IJCNLP 2021. Dataset card: [LexGLUE LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue).
2. Hugging Face Transformers, GPT-2 model documentation: [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2).
3. PyTorch, dynamic quantization API documentation: [quantize_dynamic](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html).
