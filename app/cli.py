"""Command-line entry point for the complete local pipeline."""
import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-legal-tagger-monitor", description="Legal aid workers use this to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("health", help="check dependencies and local hardware")
    p = sub.add_parser("prepare", help="download and prepare LEDGAR")
    p.add_argument("--limit", type=int, default=5000); p.add_argument("--force", action="store_true")
    p = sub.add_parser("train", help="fine-tune GPT-2")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--epochs", type=int, default=1); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--max-length", type=int, default=256); p.add_argument("--model", default="openai-community/gpt2"); p.add_argument("--device", default="auto"); p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("evaluate", help="evaluate held-out examples")
    p.add_argument("--model-path", default="artifacts/original"); p.add_argument("--split", default="test", choices=["train","validation","test"]); p.add_argument("--limit", type=int, default=0); p.add_argument("--device", default="auto")
    sub.add_parser("quantize", help="apply CPU dynamic INT8 quantization")
    p = sub.add_parser("benchmark", help="measure local CPU latency and size")
    p.add_argument("--limit", type=int, default=50); p.add_argument("--repeats", type=int, default=3)
    sub.add_parser("static-quantize", help="report whether static quantization is supported")
    sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p = sub.add_parser("pipeline", help="prepare, train, evaluate, quantize and benchmark")
    p.add_argument("--limit", type=int, default=5000, help="prepared dataset cap; also training cap")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Quantized Legal Tagger Monitor is ready. Open README.md for your milestones.")
    elif args.command == "health":
        import platform
        from app.core import _device
        import torch
        print(json.dumps({"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"device":str(_device(torch))},indent=2))
    elif args.command == "prepare":
        from app.core import prepare_dataset
        print(json.dumps(prepare_dataset(args.limit, force=args.force),indent=2))
    elif args.command == "train":
        from app.core import train
        print(json.dumps(train(args.limit,args.epochs,args.batch_size,args.max_length,args.model,args.device,resume=not args.no_resume),indent=2))
    elif args.command == "evaluate":
        from app.core import evaluate
        print(json.dumps(evaluate(args.model_path,args.split,args.device,args.limit),indent=2))
    elif args.command == "quantize":
        from app.core import quantize_dynamic
        print(json.dumps(quantize_dynamic(),indent=2))
    elif args.command == "benchmark":
        from app.core import benchmark
        print(json.dumps(benchmark(limit=args.limit,repeats=args.repeats),indent=2))
    elif args.command == "static-quantize":
        result={"supported":False,"status":"unsupported","reason":"GPT-2 uses custom Conv1D projections and generation with KV-cache; the available PyTorch eager static quantization workflow cannot convert these modules into a complete executable static INT8 generation model. No static model or metrics were produced."}
        Path("artifacts").mkdir(exist_ok=True); Path("artifacts/static_quantization.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    elif args.command == "dashboard":
        import subprocess, sys
        subprocess.run([sys.executable,"-m","streamlit","run","app/dashboard.py"],check=True)
    elif args.command == "pipeline":
        from app.core import prepare_dataset, train, evaluate, quantize_dynamic, benchmark
        print("[1/5] Preparing dataset"); print(json.dumps(prepare_dataset(args.limit),indent=2))
        print("[2/5] Fine-tuning model"); print(json.dumps(train(limit=args.limit,epochs=args.epochs,device=args.device),indent=2))
        print("[3/5] Evaluating held-out test split"); print(json.dumps(evaluate(),indent=2))
        print("[4/5] Applying dynamic quantization"); print(json.dumps(quantize_dynamic(),indent=2))
        print("[5/5] Benchmarking"); print(json.dumps(benchmark(),indent=2))


if __name__ == "__main__":
    main()
