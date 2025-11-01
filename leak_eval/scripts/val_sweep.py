import argparse
import subprocess
import sys
from pathlib import Path


def _parse_layers(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_strengths(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation sweep for multi-layer steering")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_file", default="datasets/airgapagent-r-val.json")
    parser.add_argument("--output_dir", default="results/qwq-results/updated/val_sweep")
    parser.add_argument("--prompt_type", default="cot_explicit_unk")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--enable_gpt_eval", action="store_true")
    parser.add_argument("--gpt_eval", action="store_true")
    parser.add_argument("--gpt_eval_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steering_layers", required=True, help="Comma-separated layers steered together")
    parser.add_argument("--strengths", required=True, help="Comma-separated strengths to sweep")
    parser.add_argument("--vector_dir", required=True, help="Directory with steering_vector_layer_{L}.npy")
    parser.add_argument("--steering_method", type=str, default="add", choices=["add", "replace"]) 
    parser.add_argument("--steer_only_last_input", action="store_true")

    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    leak_eval_root = scripts_dir.parent
    steered_entry = leak_eval_root / "steered_eval_cp_resume.py"

    output_dir = (leak_eval_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = (leak_eval_root / args.vector_dir).resolve()
    input_file = (leak_eval_root / args.input_file).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"input_file not found: {input_file}")
    if not steered_entry.exists():
        raise FileNotFoundError(f"steered_eval_cp_resume.py not found at {steered_entry}")
    if not vector_dir.exists():
        raise FileNotFoundError(f"vector_dir not found: {vector_dir}")

    layers = _parse_layers(args.steering_layers)
    strengths = _parse_strengths(args.strengths)

    missing = [L for L in layers if not (vector_dir / f"steering_vector_layer_{L}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing vectors for layers: {missing} under {vector_dir}")

    print(f"Layers: {layers}")
    print(f"Strengths: {strengths}")
    print(f"Vector dir: {vector_dir}")
    print(f"Writing results to: {output_dir}")

    for s in strengths:
        strengths_list = [s] * len(layers)
        s_tag = str(s).replace('.', '_')
        out_path = output_dir / f"multi_L{layers[0]}-{layers[-1]}_s{s_tag}.json"

        cmd = [
            sys.executable,
            str(steered_entry),
            "--model", args.model,
            "--input_file", str(input_file),
            "--output_file", str(out_path),
            "--prompt_type", args.prompt_type,
            "--batch_size", str(args.batch_size),
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--steering_layers", ','.join(map(str, layers)),
            f"--steering_strengths={','.join(map(str, strengths_list))}",
            "--steering_vector_dir", str(vector_dir),
            "--steering_method", args.steering_method,
        ]
        if args.steer_only_last_input:
            cmd.append("--steer_only_last_input")
        if args.enable_gpt_eval:
            cmd.append("--enable_gpt_eval")
        if args.gpt_eval:
            cmd.append("--gpt_eval")
            cmd += ["--gpt_eval_model", args.gpt_eval_model]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]

        print(f"\n[RUN] strength={s} -> {out_path}")
        subprocess.run(cmd, check=True)

    print("\nSweep completed.")


if __name__ == "__main__":
    main()


