import argparse
import json
import os
import random
from typing import List, Tuple


def load_dataset(path: str) -> Tuple[List[dict], bool]:
    """Load dataset from JSON.

    Supports two formats:
    - A plain JSON list of examples
    - An object with key 'data' containing the list

    Returns: (examples_list, wrapped)
      wrapped=True if original file had a top-level {'data': [...]}
    """
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"], True

    if isinstance(obj, list):
        return obj, False

    raise ValueError("Expected a JSON list or an object with key 'data'.")


def save_dataset(path: str, items: List[dict], wrap: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"data": items} if wrap else items, f, indent=2)


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)

    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Split a dataset JSON into train/val/test by ratios (deterministic shuffle).")
    parser.add_argument("--input_file", required=True, type=str, help="Path to full dataset JSON")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save split files")
    parser.add_argument("--seed", type=int, default=221097, help="Random seed for deterministic split")
    parser.add_argument("--train_ratio", type=float, default=0.15, help="Fraction for train split (0-1)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction for val split (0-1)")
    args = parser.parse_args()

    data, wrapped = load_dataset(args.input_file)
    n = len(data)

    train_idx, val_idx, test_idx = split_indices(n, args.train_ratio, args.val_ratio, args.seed)

    base = os.path.splitext(os.path.basename(args.input_file))[0]
    out_train = os.path.join(args.output_dir, f"{base}-train.json")
    out_val = os.path.join(args.output_dir, f"{base}-val.json")
    out_test = os.path.join(args.output_dir, f"{base}-test.json")

    save_dataset(out_train, [data[i] for i in train_idx], wrapped)
    save_dataset(out_val, [data[i] for i in val_idx], wrapped)
    save_dataset(out_test, [data[i] for i in test_idx], wrapped)

    print(f"Total: {n}")
    print(f"train: {len(train_idx)} -> {out_train}")
    print(f"val:   {len(val_idx)} -> {out_val}")
    print(f"test:  {len(test_idx)} -> {out_test}")


if __name__ == "__main__":
    main()


