import os
import json
import argparse
import pandas as pd

from dotenv import load_dotenv

# Allow running as a standalone script
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cp_eval_utils import (
    compute_gpt_extraction_for_all,
    compute_gpt_utility_score,
    compute_gpt_pii_leakage,
    calculate_openai_cost,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GPT-based PII/utility eval on an existing results JSON",
    )
    parser.add_argument("--results_file", required=True, help="Path to results JSON produced by eval_cp.py")
    parser.add_argument("--gpt_model", default="gpt-4o-mini", help="GPT model for eval (default: gpt-4o-mini)")
    parser.add_argument("--ref_answer", default="ref_answer", help="Field name for reference answer")
    parser.add_argument("--prompt_inj", action="store_true", help="Enable prompt-injection mode (answer-only eval)")
    parser.add_argument("--approp_matrix", default="approp_matrix.csv", help="Path to appropriateness matrix CSV")
    return parser.parse_args()


def main():
    load_dotenv(dotenv_path=str(ROOT / ".env"))

    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required for GPT evaluation")

    if not os.path.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}")

    with open(args.results_file, "r") as f:
        results = json.load(f)

    data = results.get("data", [])
    if not data:
        print("No data points found in results file. Nothing to evaluate.")
        return

    print(f"Loaded {len(data)} items from {args.results_file}")
    print(f"Running GPT eval using model: {args.gpt_model}")

    # 1) Extract PII with GPT for all items
    all_responses = compute_gpt_extraction_for_all(
        data,
        model=args.gpt_model,
        prompt_inj=args.prompt_inj,
    )

    total_cost = calculate_openai_cost(all_responses)
    print(f"Estimated OpenAI API cost: ${total_cost:.4f}")

    # 2) Compute GPT-based utility and PII leakage
    approp_matrix_df = pd.read_csv(args.approp_matrix, index_col=0)
    gpt_utility_score = compute_gpt_utility_score(data, args.ref_answer)
    gpt_pii_leakage = compute_gpt_pii_leakage(data, approp_matrix_df)

    # 3) Update summary and write back
    results.setdefault("summary", {})
    results["summary"].update(
        {
            "gpt_utility_score": gpt_utility_score,
            "gpt_pii_leakage": gpt_pii_leakage,
            "total_gpt_api_cost": total_cost,
        }
    )
    results["data"] = data

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Finished GPT eval and updated:", args.results_file)


if __name__ == "__main__":
    main()


