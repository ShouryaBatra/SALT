import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Fixed configuration matching your requested notebook cell
    MODEL = "qwq-32b"
    INPUT_FILE = "datasets/airgapagent-r-val.json"
    OUTPUT_DIR = "results/qwq-results/updated/val_sweep"
    VECTOR_DIR = "results/real_leak_layer_analysis_vectors/steering_vectors"
    LAYERS = [25, 47, 28, 63]
    STRENGTHS = [-2.0, -1.5, -1.0, -0.5]
    PROMPT_TYPE = "cot_explicit_unk"
    BATCH_SIZE = 5
    MAX_TOKENS = 500
    TEMPERATURE = 0.4
    ENABLE_GPT_EVAL = True
    GPT_EVAL = True
    LIMIT = 300

    # Resolve paths relative to leak_eval/
    scripts_dir = Path(__file__).resolve().parent
    leak_eval_root = scripts_dir.parent
    steered_entry = leak_eval_root / "steered_eval_cp_resume.py"

    output_dir = (leak_eval_root / OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_dir = (leak_eval_root / VECTOR_DIR).resolve()
    input_file = (leak_eval_root / INPUT_FILE).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"input_file not found: {input_file}")
    if not steered_entry.exists():
        raise FileNotFoundError(f"steered_eval_cp_resume.py not found at {steered_entry}")

    print(f"Sweeping layers={LAYERS}\nStrengths={STRENGTHS}")
    print(f"Vectors from: {vector_dir}")
    print(f"Writing results under: {output_dir}")

    for layer in LAYERS:
        vector_path = vector_dir / f"steering_vector_layer_{layer}.npy"
        if not vector_path.exists():
            print(f"[WARN] Missing vector for layer {layer}: {vector_path} — skipping this layer")
            continue

        for strength in STRENGTHS:
            out_path = output_dir / f"l{layer}_s{strength}.json"
            cmd = [
                sys.executable,
                str(steered_entry),
                "--model", MODEL,
                "--input_file", str(input_file),
                "--output_file", str(out_path),
                "--prompt_type", PROMPT_TYPE,
                "--batch_size", str(BATCH_SIZE),
                "--max_tokens", str(MAX_TOKENS),
                "--temperature", str(TEMPERATURE),
                "--steering_layer", str(layer),
                "--steering_vector_path", str(vector_path),
                "--steering_strength", str(strength),
                "--limit", str(LIMIT),
            ]
            if ENABLE_GPT_EVAL:
                cmd.append("--enable_gpt_eval")
            if GPT_EVAL:
                cmd.append("--gpt_eval")

            print(f"\n[RUN] layer={layer} strength={strength}\n  -> {out_path}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Run failed for layer={layer} strength={strength}: {e}")
                continue

    print("\nSweep completed.")


if __name__ == "__main__":
    main()


