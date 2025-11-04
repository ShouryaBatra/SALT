import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _any_list(x) -> bool:
    if isinstance(x, list):
        return any(bool(v) for v in x)
    return bool(x)


def extract_flags_from_json(json_path: Path, component: str) -> Dict:
    with open(json_path, "r") as f:
        obj = json.load(f)

    items: List[Dict] = obj.get("data", [])
    model_name: str = (obj.get("args", {}) or {}).get("model", "unknown-model")

    flags: List[int] = []
    for it in items:
        leaks = it.get("pii_leaks") or it.get("gpt_pii_leaks") or {}
        if component == "union":
            flag = (
                _any_list(leaks.get("output_bin", []))
                or _any_list(leaks.get("reasoning_bin", []))
                or _any_list(leaks.get("answer_bin", []))
            )
        elif component in ("output_bin", "reasoning_bin", "answer_bin"):
            flag = _any_list(leaks.get(component, []))
        else:
            raise ValueError(f"Unknown component: {component}")
        flags.append(1 if flag else 0)

    flags_arr = np.asarray(flags, dtype=float)
    n = int(flags_arr.size)
    p = float(flags_arr.mean()) if n > 0 else float("nan")
    se = float(math.sqrt(p * (1.0 - p) / n)) if n > 0 else float("nan")
    return {"model": model_name, "n": n, "p": p, "se": se}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CPL (mean leakage rate) and standard error from eval_cp JSONs. "
            "Works with baseline and steered result files."
        )
    )
    parser.add_argument(
        "--baseline_json",
        type=str,
        required=True,
        help="Path to baseline eval_cp JSON",
    )
    parser.add_argument(
        "--steered_json",
        type=str,
        required=True,
        help="Path to steered eval_cp JSON",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="union",
        choices=["union", "output_bin", "reasoning_bin", "answer_bin"],
        help=(
            "Which leakage component to use. "
            "'union' = any of output/reasoning/answer (recommended for CPL)."
        ),
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default="Baseline",
        help="Label for baseline condition",
    )
    parser.add_argument(
        "--steered_label",
        type=str,
        default="Steered",
        help="Label for steered condition",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to write a summary CSV with CPL and SE values",
    )

    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    steered_path = Path(args.steered_json)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {baseline_path}")
    if not steered_path.exists():
        raise FileNotFoundError(f"Steered JSON not found: {steered_path}")

    base_res = extract_flags_from_json(baseline_path, args.component)
    steer_res = extract_flags_from_json(steered_path, args.component)

    rows = [
        {
            "condition": args.baseline_label,
            "model": base_res["model"],
            "n": base_res["n"],
            "cpl": base_res["p"],
            "std_error": base_res["se"],
            "moe95": 1.96 * base_res["se"] if not math.isnan(base_res["se"]) else float("nan"),
            "component": args.component,
        },
        {
            "condition": args.steered_label,
            "model": steer_res["model"],
            "n": steer_res["n"],
            "cpl": steer_res["p"],
            "std_error": steer_res["se"],
            "moe95": 1.96 * steer_res["se"] if not math.isnan(steer_res["se"]) else float("nan"),
            "component": args.component,
        },
    ]

    df = pd.DataFrame(rows)

    # Pretty print
    def fmt(x: Optional[float]) -> str:
        try:
            return f"{x:.4f}"
        except Exception:
            return str(x)

    print("\nCPL and Standard Errors (component=", args.component, ")", sep="")
    for _, r in df.iterrows():
        print(
            f"{r['condition']:<8} model={r['model']}  n={int(r['n'])}  "
            f"CPL={fmt(r['cpl'])}  SE={fmt(r['std_error'])}  95%MOE≈{fmt(r['moe95'])}"
        )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nWrote summary CSV: {out_path}")


if __name__ == "__main__":
    main()


