"""Command-line entry point for the complete local pipeline."""
import argparse
import json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-legal-tagger-monitor", description="Legal aid workers use this to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("health", help="check dependencies and local hardware")
    p = sub.add_parser("prepare", help="download and prepare official LEDGAR splits")
    p.add_argument("--limit", type=int, default=3000, help="total cap; allocated 80/10/10 across official splits")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("train", help="fine-tune GPT-2")
    p.add_argument("--limit", type=int, default=0, help="stratified train cap; 0 uses prepared train split")
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=160); p.add_argument("--model", default="openai-community/gpt2")
    p.add_argument("--device", default="auto"); p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("evaluate", help="evaluate held-out examples")
    p.add_argument("--model-path", default=None); p.add_argument("--split", default="test", choices=["train","validation","test"]); p.add_argument("--limit", type=int, default=0); p.add_argument("--device", default="auto")
    sub.add_parser("quantize", help="apply CPU dynamic INT8 quantization")
    p = sub.add_parser("benchmark", help="measure local CPU latency and size")
    p.add_argument("--limit", type=int, default=50); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--sessions", "--repeats", dest="sessions", type=int, default=3)
    p = sub.add_parser("static-quantize", help="calibrate and apply static CPU INT8 projection quantization")
    p.add_argument("--calibration-limit", type=int, default=128)
    sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    sub.add_parser("demo", help="launch dashboard using the precomputed Phase 2 results")
    p = sub.add_parser("pipeline", help="prepare, train, evaluate, quantize and benchmark")
    p.add_argument("--limit", type=int, default=3000, help="prepared total cap; allocated 80/10/10 across official splits")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--benchmark-inputs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--sessions", type=int, default=3)
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
        print(json.dumps(benchmark(limit=args.limit,warmup=args.warmup,iterations=args.iterations,sessions=args.sessions),indent=2))
    elif args.command == "static-quantize":
        from app.core import quantize_static
        print(json.dumps(quantize_static(args.calibration_limit),indent=2))
    elif args.command == "dashboard":
        import subprocess, sys
        subprocess.run([sys.executable,"-m","streamlit","run","app/dashboard.py"],check=True)
    elif args.command == "demo":
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        phase2 = root / "artifacts" / "phase2"
        bundled = root / "reports" / "demo"
        candidates = [phase2, bundled]
        complete = lambda folder: all((folder / name).is_file() for name in ["benchmark.json", *[f"evaluation_{model}_test.json" for model in ("original", "dynamic", "static")]])
        if not any(complete(folder) for folder in candidates):
            missing = [str((bundled / name).relative_to(root)) for name in ["benchmark.json", *[f"evaluation_{model}_test.json" for model in ("original", "dynamic", "static")]] if not (bundled / name).is_file()]
            parser.error("Precomputed demo artifacts are missing: " + ", ".join(missing) + ". Restore reports/demo or run the documented pipeline.")
        import subprocess, sys
        subprocess.run([sys.executable,"-m","streamlit","run","app/dashboard.py"],check=True)
    elif args.command == "pipeline":
        from app import core
        prepare_dataset, train, evaluate = core.prepare_dataset, core.train, core.evaluate
        quantize_dynamic, quantize_static, benchmark = core.quantize_dynamic, core.quantize_static, core.benchmark
        print("[1/8] Preparing dataset"); print(json.dumps(prepare_dataset(args.limit),indent=2))
        print("[2/8] Fine-tuning model"); print(json.dumps(train(limit=args.limit,epochs=args.epochs,batch_size=args.batch_size,max_length=args.max_length,device=args.device),indent=2))
        print("[3/8] Evaluating original on held-out test split"); print(json.dumps(evaluate(),indent=2))
        print("[4/8] Applying dynamic quantization"); print(json.dumps(quantize_dynamic(),indent=2))
        print("[5/8] Evaluating dynamic model"); print(json.dumps(evaluate(core.MODEL_DIR / "dynamic"),indent=2))
        print("[6/8] Calibrating and applying static quantization"); print(json.dumps(quantize_static(),indent=2))
        print("[7/8] Evaluating static model"); print(json.dumps(evaluate(core.MODEL_DIR / "static"),indent=2))
        print("[8/8] Benchmarking all executable models"); print(json.dumps(benchmark(limit=args.benchmark_inputs,warmup=args.warmup,iterations=args.iterations,sessions=args.sessions),indent=2))


if __name__ == "__main__":
    main()
