import os
import re
import glob
import json
import argparse
from typing import Dict

import numpy as np


def load_input_lengths_from_results(results_json_path: str) -> Dict[int, int]:
    with open(results_json_path, "r") as f:
        results = json.load(f)
    data = results.get("data", results)
    input_lengths: Dict[int, int] = {}
    for idx, item in enumerate(data):
        try:
            example_id = int(item.get("example_id", item.get("id", idx)))
        except Exception:
            example_id = idx
        in_len = item.get("input_token_length")
        if in_len is not None:
            try:
                input_lengths[example_id] = int(in_len)
            except Exception:
                pass
    return input_lengths


def parse_layer_and_example_from_filename(path: str):
    base = os.path.basename(path)
    m = re.match(r"layer_(\d+)_example_(\d+)\.npz$", base)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def process_file(path: str, input_lengths: Dict[int, int], overwrite: bool = False) -> bool:
    sidecar_path = path + ".last_input.npy"
    if (not overwrite) and os.path.exists(sidecar_path):
        return True

    layer_idx, example_id = parse_layer_and_example_from_filename(path)
    if example_id is None:
        return False

    in_len = input_lengths.get(example_id)
    if in_len is None:
        return False

    try:
        with np.load(path, allow_pickle=True) as npz:
            if "activation" not in npz:
                return False
            act = npz["activation"]
            if not isinstance(act, np.ndarray) or act.ndim != 2:
                return False
            seq_len = act.shape[0]
            idx = max(0, min(int(in_len) - 1, seq_len - 1))
            vec = act[idx].astype(np.float32, copy=False)
    except Exception:
        return False

    try:
        np.save(sidecar_path, vec)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description="Precompute 1D last-input vectors as sidecar .npy files for each activation .npz")
    p.add_argument("--activations_dir", required=True, help="Directory containing layer_*_example_*.npz")
    p.add_argument("--results_json", required=True, help="Results JSON containing per-example input_token_length")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecar .npy files if present")
    p.add_argument("--pattern", type=str, default="layer_*_example_*.npz", help="Glob pattern within activations_dir")
    args = p.parse_args()

    input_lengths = load_input_lengths_from_results(args.results_json)
    paths = glob.glob(os.path.join(args.activations_dir, args.pattern))
    if not paths:
        raise FileNotFoundError(f"No activation files found in {args.activations_dir}")

    total = len(paths)
    ok = 0
    skipped = 0

    print(f"[precompute_last_input_sidecars] Found {total} files. Writing sidecars next to originals...")
    for i, path in enumerate(paths, 1):
        success = process_file(path, input_lengths, overwrite=args.overwrite)
        if success:
            ok += 1
        else:
            skipped += 1
        if i % 1000 == 0 or i == total:
            print(f"  processed {i}/{total} | sidecars_ok={ok} | skipped={skipped}")

    print(f"[precompute_last_input_sidecars] Done. sidecars_ok={ok}, skipped={skipped}")


if __name__ == "__main__":
    main()


