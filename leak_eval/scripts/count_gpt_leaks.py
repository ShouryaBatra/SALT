import argparse
import json
import os
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Count GPT-evaluated PII leakages in a results JSON")
    p.add_argument("--results_file", required=True, help="Path to results JSON (from eval_cp or steered_eval)")
    p.add_argument(
        "--component",
        choices=["reasoning", "output", "answer", "any"],
        default="reasoning",
        help="Which part to count leakage for (default: reasoning)",
    )
    p.add_argument(
        "--print_ids",
        action="store_true",
        help="Also print indices of examples counted as leaking",
    )
    p.add_argument(
        "--estimate_from_summary",
        action="store_true",
        help="If per-item GPT leaks are missing, estimate count from summary averages",
    )
    return p.parse_args()


def has_leak(item: Dict, component: str) -> bool:
    leaks = item.get("gpt_pii_leaks", {})
    if component == "any":
        return any(
            any(bool(x) for x in leaks.get(f"{part}_bin", []) )
            for part in ("output", "reasoning", "answer")
        )
    return any(bool(x) for x in leaks.get(f"{component}_bin", []))


def main():
    args = parse_args()

    if not os.path.exists(args.results_file):
        raise FileNotFoundError(args.results_file)

    with open(args.results_file, "r") as f:
        results = json.load(f)

    data: List[Dict] = results.get("data", [])
    if not data:
        print("No data found.")
        return

    # Prefer per-item gpt_pii_leaks if present
    per_item_available = any("gpt_pii_leaks" in item for item in data)

    if per_item_available:
        leaking_ids = [i for i, item in enumerate(data) if has_leak(item, args.component)]
        count = len(leaking_ids)
        print(f"Total items: {len(data)}")
        print(f"Component: {args.component}")
        print(f"GPT leakage count: {count}")
        if args.print_ids and leaking_ids:
            print("Leaking indices:", ", ".join(map(str, leaking_ids)))
        return

    # Fallback: estimate from summary if requested
    if args.estimate_from_summary:
        summary = results.get("summary", {})
        total = summary.get("total_examples") or len(data)
        key_map = {
            "reasoning": "gpt_reasoning_bin_avg",
            "output": "gpt_output_bin_avg",
            "answer": "gpt_answer_bin_avg",
        }
        if args.component == "any":
            # Cannot estimate "any" reliably from averages; report N/A
            print(f"Total items: {total}")
            print("Component: any")
            print("GPT leakage count (estimate): N/A (need per-item data)")
            return
        avg_key = key_map[args.component]
        avg = summary.get("gpt_pii_leakage", {}).get(avg_key)
        if avg is None:
            print("Summary does not contain GPT leakage averages; cannot estimate.")
            return
        estimate = int(round(float(avg) * int(total)))
        print(f"Total items: {total}")
        print(f"Component: {args.component}")
        print(f"GPT leakage count (estimate): {estimate}")
        return

    print("Per-item GPT leakage not found. Re-run GPT eval or use --estimate_from_summary.")


if __name__ == "__main__":
    main()


