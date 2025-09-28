import os
import json
import argparse
from pathlib import Path
import sys


# Ensure repo root on sys.path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from leak_eval.cp_eval_utils import (
    compute_utility_score,
    compute_pii_leakage,
    compute_gpt_utility_score,
    compute_gpt_pii_leakage,
)
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two eval results JSONs by example_id, optionally recompute summary",
    )
    parser.add_argument("--base", required=True, help="Path to base results JSON (will be updated)")
    parser.add_argument("--update", required=True, help="Path to results JSON providing replacements")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write merged JSON (default: overwrite --base)",
    )
    parser.add_argument(
        "--recompute-summary",
        action="store_true",
        help="Recompute utility/PII summaries after merge (no extra API calls)",
    )
    parser.add_argument(
        "--fallback-positional",
        action="store_true",
        help="If base items lack example_id, replace by list index position",
    )
    parser.add_argument(
        "--ref_answer",
        default="ref_answer",
        help="Reference answer field for utility computation",
    )
    parser.add_argument(
        "--approp-matrix",
        default="leak_eval/approp_matrix.csv",
        help="Appropriateness matrix CSV for PII metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output if args.output else args.base

    if not os.path.exists(args.base):
        raise FileNotFoundError(f"Base file not found: {args.base}")
    if not os.path.exists(args.update):
        raise FileNotFoundError(f"Update file not found: {args.update}")

    with open(args.base, "r") as f:
        base_json = json.load(f)
    with open(args.update, "r") as f:
        update_json = json.load(f)

    base_data = base_json.get("data", [])
    update_data = update_json.get("data", [])

    merged_data = list(base_data)
    replaced = 0

    # Primary: replace by example_id when available
    update_by_id = {it.get("example_id"): it for it in update_data if it.get("example_id") is not None}
    if update_by_id:
        for i, item in enumerate(merged_data):
            eid = item.get("example_id")
            if eid is not None and eid in update_by_id:
                merged_data[i] = update_by_id[eid]
                replaced += 1

    # Fallback: positional replacement when requested
    if args.fallback_positional:
        limit = min(len(merged_data), len(update_data))
        for i in range(limit):
            base_has_id = merged_data[i].get("example_id") is not None
            update_has_id = update_data[i].get("example_id") is not None
            # Only replace positionally if base lacks id OR ids differ
            if not base_has_id or (update_has_id and merged_data[i].get("example_id") != update_data[i].get("example_id")):
                merged_data[i] = update_data[i]
                replaced += 1

    base_json["data"] = merged_data

    # Recompute summary if requested (no API calls)
    if args.recompute_summary:
        summary = base_json.get("summary", {})

        # Non-GPT metrics
        try:
            approp_df = pd.read_csv(args.approp_matrix, index_col=0)
            pii_leakage = compute_pii_leakage(merged_data, approp_df)
        except Exception:
            pii_leakage = None

        utility_score = compute_utility_score(merged_data, args.ref_answer)

        # GPT metrics (derived from existing gpt_extractions; no API calls)
        try:
            approp_df = pd.read_csv(args.approp_matrix, index_col=0)
            gpt_utility_score = compute_gpt_utility_score(merged_data, args.ref_answer)
            gpt_pii_leakage = compute_gpt_pii_leakage(merged_data, approp_df)
        except Exception:
            gpt_utility_score = None
            gpt_pii_leakage = None

        summary.update(
            {
                "utility_score": utility_score,
                "total_examples": len(merged_data),
                "positive_examples": sum(1 for d in merged_data if d.get("label") == 1),
                "negative_examples": sum(1 for d in merged_data if d.get("label") == 0),
            }
        )
        if pii_leakage is not None:
            summary["pii_leakage"] = pii_leakage
        if gpt_utility_score is not None:
            summary["gpt_utility_score"] = gpt_utility_score
        if gpt_pii_leakage is not None:
            summary["gpt_pii_leakage"] = gpt_pii_leakage

        base_json["summary"] = summary

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(base_json, f, indent=2)

    print(f"Merged {replaced} examples into {output_path}")


if __name__ == "__main__":
    main()


