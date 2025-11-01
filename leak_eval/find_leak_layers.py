import argparse
import glob
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_labels_from_results(results_json_path: str) -> Dict[int, int]:
    """Load per-example leak labels from a results JSON file.

    Returns a dict mapping example_id -> 1 (leaky) or 0 (non-leaky).

    Tries multiple conventions for compatibility:
    - item["pii_leaks"]["reasoning_bin"][0]
    - item["label"]
    """
    with open(results_json_path, "r") as f:
        results = json.load(f)

    data = results.get("data", results)  # allow raw list

    labels: Dict[int, int] = {}
    for example_id, item in enumerate(data):
        label_val = None

        # Preferred: per-example PII leak in reasoning (binary)
        try:
            reasoning_bin = item.get("pii_leaks", {}).get("reasoning_bin", [0])
            if isinstance(reasoning_bin, list) and len(reasoning_bin) > 0:
                label_val = int(reasoning_bin[0] > 0)
            else:
                label_val = int(bool(reasoning_bin))
        except Exception:
            label_val = None

        # Fallback: explicit label field
        if label_val is None and "label" in item:
            try:
                label_val = int(item["label"])  # assume 0/1
            except Exception:
                pass

        # Final fallback: non-leaky
        if label_val is None:
            label_val = 0

        labels[example_id] = label_val

    return labels


def pick_activation_vector(npz_obj: np.lib.npyio.NpzFile, segment: str, input_len: Optional[int] = None) -> np.ndarray:
    """Select a 1D activation vector from an activation .npz entry.

    Priority by segment:
      - reasoning_avg:  reasoning_avg_activation -> fallback to full_avg_activation -> activation.mean(axis=0)
      - full_avg:       full_avg_activation      -> fallback to activation.mean(axis=0)
      - answer_avg:     answer_avg_activation    -> fallback to full_avg_activation -> activation.mean(axis=0)
      - last_input:     activation[input_len-1]  -> fallback to activation.mean(axis=0) if not 2D
    """
    def ensure_1d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr
        # If 2D [seq_len, hidden_dim], average over tokens
        if arr.ndim == 2:
            return arr.mean(axis=0)
        # Otherwise attempt to flatten last dim by averaging all but last
        if arr.ndim > 2:
            # reshape to [N, hidden_dim] then mean over N
            reshaped = arr.reshape(-1, arr.shape[-1])
            return reshaped.mean(axis=0)
        return arr

    if segment == "last_input":
        # Prefer precomputed 1D last-input vector if present
        if "last_input" in npz_obj:
            arr = npz_obj["last_input"]  # type: ignore[index]
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                return arr
            return ensure_1d(arr)
        # Prefer using the raw per-token activation and index the last input token
        if "activation" in npz_obj:
            arr = npz_obj["activation"]  # type: ignore[index]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                if input_len is None:
                    raise ValueError("input_len is required for segment='last_input'")
                seq_len = arr.shape[0]
                idx = max(0, min(int(input_len) - 1, seq_len - 1))
                return arr[idx]
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                # Already a 1D vector; use as-is
                return arr
        # Fallback: try any key that is 2D
        for key in list(npz_obj.keys()):
            try:
                arr2 = npz_obj[key]  # type: ignore[index]
            except Exception:
                continue
            if isinstance(arr2, np.ndarray) and arr2.ndim == 2:
                if input_len is None:
                    raise ValueError("input_len is required for segment='last_input'")
                seq_len = arr2.shape[0]
                idx = max(0, min(int(input_len) - 1, seq_len - 1))
                return arr2[idx]
        # As a last resort, average
        if "activation" in npz_obj:
            return ensure_1d(npz_obj["activation"])  # type: ignore[index]
        raise ValueError("Could not locate a suitable per-token activation for last_input")

    if segment == "reasoning_avg":
        if "reasoning_avg_activation" in npz_obj:
            return ensure_1d(npz_obj["reasoning_avg_activation"])  # type: ignore[index]
        if "full_avg_activation" in npz_obj:
            return ensure_1d(npz_obj["full_avg_activation"])  # type: ignore[index]
    elif segment == "full_avg":
        if "full_avg_activation" in npz_obj:
            return ensure_1d(npz_obj["full_avg_activation"])  # type: ignore[index]
    elif segment == "answer_avg":
        if "answer_avg_activation" in npz_obj:
            return ensure_1d(npz_obj["answer_avg_activation"])  # type: ignore[index]
        if "full_avg_activation" in npz_obj:
            return ensure_1d(npz_obj["full_avg_activation"])  # type: ignore[index]

    # Generic fallback: try a single array key or the 'activation' key
    if "activation" in npz_obj:
        return ensure_1d(npz_obj["activation"])  # type: ignore[index]

    # If only one array present, use it
    keys = list(npz_obj.keys())
    if len(keys) == 1:
        return ensure_1d(npz_obj[keys[0]])  # type: ignore[index]

    # As a last resort, try common averages in order
    for key in ["full_avg_activation", "reasoning_avg_activation", "answer_avg_activation"]:
        if key in npz_obj:
            return ensure_1d(npz_obj[key])  # type: ignore[index]

    raise ValueError("Could not locate a suitable activation vector in .npz file")


def compute_effects(pos_matrix: np.ndarray, neg_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension raw difference (delta) and Cohen's d (standardized effect).

    pos_matrix, neg_matrix: shape [num_examples, hidden_dim]
    Returns (delta, d): both shape [hidden_dim]
    """
    mu_pos = pos_matrix.mean(axis=0)
    mu_neg = neg_matrix.mean(axis=0)
    delta = mu_pos - mu_neg

    # Pooled std per dim
    std_pos = pos_matrix.std(axis=0, ddof=1) if pos_matrix.shape[0] > 1 else pos_matrix.std(axis=0)
    std_neg = neg_matrix.std(axis=0, ddof=1) if neg_matrix.shape[0] > 1 else neg_matrix.std(axis=0)
    pooled = np.sqrt((std_pos ** 2 + std_neg ** 2) / 2.0)
    d = delta / (pooled + 1e-8)
    return delta, d


def analyze_layers(
    activations_dir: str,
    labels: Dict[int, int],
    input_lengths: Dict[int, int],
    segment: str,
    threshold: float,
    min_examples: int,
) -> Tuple[List[Dict], Dict[int, Dict[str, np.ndarray]]]:
    """Load activations per layer, split by labels, and compute per-layer metrics.

    Returns:
      - rows: list of per-layer dict metrics for CSV
      - vectors: per-layer dict with 'delta' and 'cohen_d' vectors
    """
    # Gather files
    npz_paths = glob.glob(os.path.join(activations_dir, "layer_*_example_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No activation files found in {activations_dir}")

    # Group vectors by layer
    by_layer_pos: Dict[int, List[np.ndarray]] = {}
    by_layer_neg: Dict[int, List[np.ndarray]] = {}
    hidden_dim_by_layer: Dict[int, int] = {}

    for path in npz_paths:
        try:
            npz = np.load(path, allow_pickle=True)
        except Exception:
            continue

        try:
            layer_idx = int(npz.get("layer")) if "layer" in npz else None
            example_id = int(npz.get("example_id")) if "example_id" in npz else None
        except Exception:
            layer_idx = None
            example_id = None

        if layer_idx is None or example_id is None:
            # Try to parse from filename as fallback
            base = os.path.basename(path).replace(".npz", "")
            # Expect: layer_{L}_example_{I}
            try:
                parts = base.split("_")
                # ["layer", L, "example", I]
                l_idx = int(parts[1])
                e_idx = int(parts[3])
                layer_idx = l_idx
                example_id = e_idx
            except Exception:
                continue

        vec = None
        try:
            if segment == "last_input":
                in_len = input_lengths.get(example_id)
                if in_len is None:
                    continue
                vec = pick_activation_vector(npz, segment, input_len=in_len)
            else:
                vec = pick_activation_vector(npz, segment)
        except Exception:
            continue

        if vec is None or vec.size == 0:
            continue

        hidden_dim_by_layer[layer_idx] = int(vec.shape[0])

        label = labels.get(example_id, 0)
        if label == 1:
            by_layer_pos.setdefault(layer_idx, []).append(vec)
        else:
            by_layer_neg.setdefault(layer_idx, []).append(vec)

    # Compute metrics per layer
    rows: List[Dict] = []
    vectors: Dict[int, Dict[str, np.ndarray]] = {}

    all_layers = sorted(set(list(by_layer_pos.keys()) + list(by_layer_neg.keys())))
    for layer in all_layers:
        pos_list = by_layer_pos.get(layer, [])
        neg_list = by_layer_neg.get(layer, [])

        num_pos = len(pos_list)
        num_neg = len(neg_list)
        hidden_dim = hidden_dim_by_layer.get(layer)

        if hidden_dim is None or num_pos < min_examples or num_neg < min_examples:
            continue

        pos_matrix = np.stack(pos_list, axis=0)
        neg_matrix = np.stack(neg_list, axis=0)

        delta, d = compute_effects(pos_matrix, neg_matrix)
        flagged = np.abs(d) >= threshold
        flagged_count = int(flagged.sum())
        density = float(flagged_count) / float(hidden_dim)

        row = {
            "layer": layer,
            "hidden_dim": hidden_dim,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "flagged_count": flagged_count,
            "density": density,
            "sum_abs_delta": float(np.abs(delta).sum()),
            "l2_norm_delta": float(np.linalg.norm(delta)),
            "mean_abs_d": float(np.abs(d).mean()),
        }
        rows.append(row)
        vectors[layer] = {"delta": delta, "cohen_d": d}

    # Sort by density descending, then flagged_count
    rows.sort(key=lambda r: (r["density"], r["flagged_count"]), reverse=True)
    return rows, vectors


def save_outputs(
    rows: List[Dict],
    vectors: Dict[int, Dict[str, np.ndarray]],
    output_dir: str,
    save_vectors: bool,
    vector_kind: str,
    threshold: float,
    top_k_dims: int,
):
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    import csv

    csv_path = os.path.join(output_dir, "leak_layer_ranking.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "hidden_dim",
                "num_pos",
                "num_neg",
                "flagged_count",
                "density",
                "sum_abs_delta",
                "l2_norm_delta",
                "mean_abs_d",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Save JSON summary
    summary_path = os.path.join(output_dir, "leak_layer_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"layers": rows}, f, indent=2)

    # Save bar chart: layer index vs flagged neuron count
    try:
        layers = [row["layer"] for row in rows]
        flagged_counts = [row["flagged_count"] for row in rows]
        plt.figure(figsize=(12, 4))
        plt.bar(layers, flagged_counts, color="#4C78A8")
        plt.xlabel("Layer index")
        plt.ylabel(f"Flagged neurons (|d| >= {threshold})")
        plt.title("Flagged neurons by layer")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "flagged_neurons_by_layer.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
    except Exception:
        # Plotting failures should not break analysis
        pass

    # Optionally save steering vectors (per layer)
    if save_vectors:
        vectors_dir = os.path.join(output_dir, "steering_vectors")
        os.makedirs(vectors_dir, exist_ok=True)

        for layer, vecs in vectors.items():
            base = vecs["delta"] if vector_kind == "delta" else vecs["cohen_d"]
            vec = base.astype(np.float32)

            if top_k_dims and top_k_dims > 0 and top_k_dims < vec.shape[0]:
                # Keep only top-K by absolute value, zero the rest
                idx = np.argpartition(np.abs(vec), -top_k_dims)[-top_k_dims:]
                sparse = np.zeros_like(vec)
                sparse[idx] = vec[idx]
                vec_to_save = sparse
            else:
                vec_to_save = vec

            # Normalize to unit length for steering
            norm = np.linalg.norm(vec_to_save)
            if norm > 0:
                vec_to_save = vec_to_save / norm

            out_path = os.path.join(vectors_dir, f"steering_vector_layer_{layer}.npy")
            np.save(out_path, vec_to_save)


def collect_layer_vectors(
    activations_dir: str,
    labels: Dict[int, int],
    input_lengths: Dict[int, int],
    segment: str,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]], Dict[int, int]]:
    """Load activation vectors per layer, split into leaky/non-leaky groups, and record hidden dims.

    Returns (by_layer_pos, by_layer_neg, hidden_dim_by_layer).
    """
    npz_paths = glob.glob(os.path.join(activations_dir, "layer_*_example_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No activation files found in {activations_dir}")

    by_layer_pos: Dict[int, List[np.ndarray]] = {}
    by_layer_neg: Dict[int, List[np.ndarray]] = {}
    hidden_dim_by_layer: Dict[int, int] = {}

    print(f"[find_leak_layers] Scanning {len(npz_paths)} files in '{activations_dir}' (segment='{segment}')")
    processed = 0
    used = 0
    skipped_no_meta = 0
    skipped_no_input_len = 0
    skipped_bad_vec = 0

    for path in npz_paths:
        # Fast path: if segment is last_input and a sidecar 1D vector exists, use it without opening the .npz
        if segment == "last_input":
            sidecar = path + ".last_input.npy"
            if os.path.exists(sidecar):
                # Parse layer/example from filename; fallback to npz if parsing fails
                base = os.path.basename(path).replace(".npz", "")
                try:
                    parts = base.split("_")
                    layer_idx = int(parts[1])
                    example_id = int(parts[3])
                except Exception:
                    layer_idx = None
                    example_id = None
                if layer_idx is not None and example_id is not None:
                    try:
                        vec = np.load(sidecar)
                        if isinstance(vec, np.ndarray) and vec.ndim == 1 and vec.size > 0:
                            hidden_dim_by_layer[layer_idx] = int(vec.shape[0])
                            label = labels.get(example_id, 0)
                            if label == 1:
                                by_layer_pos.setdefault(layer_idx, []).append(vec)
                            else:
                                by_layer_neg.setdefault(layer_idx, []).append(vec)
                            used += 1
                            processed += 1
                            if processed % 1000 == 0:
                                print(
                                    f"[find_leak_layers] Processed {processed}/{len(npz_paths)} | used={used} | "
                                    f"skipped(no_meta={skipped_no_meta}, no_in_len={skipped_no_input_len}, bad_vec={skipped_bad_vec})"
                                )
                            continue
                    except Exception:
                        # Fall back to normal npz loading below
                        pass
        try:
            npz = np.load(path, allow_pickle=True)
        except Exception:
            continue

        try:
            layer_idx = int(npz.get("layer")) if "layer" in npz else None
            example_id = int(npz.get("example_id")) if "example_id" in npz else None
        except Exception:
            layer_idx = None
            example_id = None

        if layer_idx is None or example_id is None:
            base = os.path.basename(path).replace(".npz", "")
            try:
                parts = base.split("_")
                layer_idx = int(parts[1])
                example_id = int(parts[3])
            except Exception:
                skipped_no_meta += 1
                continue

        try:
            if segment == "last_input":
                in_len = input_lengths.get(example_id)
                if in_len is None:
                    skipped_no_input_len += 1
                    continue
                vec = pick_activation_vector(npz, segment, input_len=in_len)
            else:
                vec = pick_activation_vector(npz, segment)
        except Exception:
            skipped_bad_vec += 1
            continue

        if vec is None or vec.size == 0:
            skipped_bad_vec += 1
            continue

        hidden_dim_by_layer[layer_idx] = int(vec.shape[0])
        label = labels.get(example_id, 0)
        if label == 1:
            by_layer_pos.setdefault(layer_idx, []).append(vec)
        else:
            by_layer_neg.setdefault(layer_idx, []).append(vec)

        used += 1
        processed += 1
        if processed % 1000 == 0:
            print(
                f"[find_leak_layers] Processed {processed}/{len(npz_paths)} | used={used} | "
                f"skipped(no_meta={skipped_no_meta}, no_in_len={skipped_no_input_len}, bad_vec={skipped_bad_vec})"
            )

    print(
        f"[find_leak_layers] Done scanning: used={used}, "
        f"skipped(no_meta={skipped_no_meta}, no_in_len={skipped_no_input_len}, bad_vec={skipped_bad_vec})"
    )
    print(f"[find_leak_layers] Layers with any vectors: {len(hidden_dim_by_layer)}")

    return by_layer_pos, by_layer_neg, hidden_dim_by_layer

def load_input_lengths_from_results(results_json_path: str) -> Dict[int, int]:
    """Load per-example input token lengths from a results JSON file.

    Returns a dict mapping example_id -> input_token_length.
    """
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


def compute_layer_effects_once(
    by_layer_pos: Dict[int, List[np.ndarray]],
    by_layer_neg: Dict[int, List[np.ndarray]],
    hidden_dim_by_layer: Dict[int, int],
    min_examples: int,
) -> Tuple[List[Dict], Dict[int, Dict[str, np.ndarray]]]:
    """Compute per-layer delta and cohen_d once, plus base row info (no thresholding)."""
    base_rows: List[Dict] = []
    vectors: Dict[int, Dict[str, np.ndarray]] = {}

    all_layers = sorted(set(list(by_layer_pos.keys()) + list(by_layer_neg.keys())))
    for layer in all_layers:
        pos_list = by_layer_pos.get(layer, [])
        neg_list = by_layer_neg.get(layer, [])
        hidden_dim = hidden_dim_by_layer.get(layer)

        num_pos = len(pos_list)
        num_neg = len(neg_list)
        if hidden_dim is None or num_pos < min_examples or num_neg < min_examples:
            continue

        pos_matrix = np.stack(pos_list, axis=0)
        neg_matrix = np.stack(neg_list, axis=0)
        delta, d = compute_effects(pos_matrix, neg_matrix)

        base_rows.append({
            "layer": layer,
            "hidden_dim": hidden_dim,
            "num_pos": num_pos,
            "num_neg": num_neg,
        })
        vectors[layer] = {"delta": delta, "cohen_d": d}

    # Sort layers by layer index for stable downstream processing
    base_rows.sort(key=lambda r: r["layer"])
    return base_rows, vectors


def rows_for_threshold(base_rows: List[Dict], vectors: Dict[int, Dict[str, np.ndarray]], threshold: float) -> List[Dict]:
    """Build rows including flagged_count and density given a threshold, using precomputed d."""
    rows: List[Dict] = []
    for r in base_rows:
        layer = r["layer"]
        hidden_dim = r["hidden_dim"]
        d = vectors[layer]["cohen_d"]
        flagged = np.abs(d) >= threshold
        flagged_count = int(flagged.sum())
        density = float(flagged_count) / float(hidden_dim)
        rows.append({
            **r,
            "flagged_count": flagged_count,
            "density": density,
            "mean_abs_d": float(np.abs(d).mean()),
            "sum_abs_delta": float(np.abs(vectors[layer]["delta"]).sum()),
            "l2_norm_delta": float(np.linalg.norm(vectors[layer]["delta"]))
        })
    # Sort by density then flagged_count, descending
    rows.sort(key=lambda rr: (rr["density"], rr["flagged_count"]), reverse=True)
    return rows


def save_csv_json_plot(rows: List[Dict], output_dir: str, threshold: float) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV with threshold suffix
    import csv
    thr_tag = str(threshold).replace(".", "_")
    csv_path = os.path.join(output_dir, f"leak_layer_ranking_thr_{thr_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "hidden_dim",
                "num_pos",
                "num_neg",
                "flagged_count",
                "density",
                "sum_abs_delta",
                "l2_norm_delta",
                "mean_abs_d",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Save JSON summary with threshold suffix
    summary_path = os.path.join(output_dir, f"leak_layer_summary_thr_{thr_tag}.json")
    with open(summary_path, "w") as f:
        json.dump({"threshold": threshold, "layers": rows}, f, indent=2)

    # Plot
    try:
        layers = [row["layer"] for row in rows]
        flagged_counts = [row["flagged_count"] for row in rows]
        plt.figure(figsize=(12, 4))
        plt.bar(layers, flagged_counts, color="#4C78A8")
        plt.xlabel("Layer index")
        plt.ylabel(f"Flagged neurons (|d| >= {threshold})")
        plt.title("Flagged neurons by layer")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"flagged_neurons_by_layer_thr_{thr_tag}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Contrastive per-neuron analysis to find layers contributing to leaky thoughts. "
            "Computes per-neuron Cohen's d between leaky and non-leaky examples, "
            "ranks layers by density of strong-effect neurons, and optionally saves steering vectors."
        )
    )
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory with layer_*_example_*.npz files")
    parser.add_argument("--results_json", type=str, required=True, help="Results JSON containing per-example leak info")
    parser.add_argument(
        "--segment",
        type=str,
        default="reasoning_avg",
        choices=["reasoning_avg", "full_avg", "answer_avg", "last_input"],
        help="Which representation to use per example (averages or last_input token)",
    )
    parser.add_argument("--threshold", type=float, default=4.0, help="Threshold on |Cohen's d| to flag neurons")
    parser.add_argument("--thresholds", nargs='+', type=float, help="Multiple thresholds for |Cohen's d|; overrides --threshold")
    parser.add_argument("--min_examples", type=int, default=1, help="Minimum examples per class per layer to compute effects")
    parser.add_argument("--output_dir", type=str, default="results/leak_layer_analysis", help="Where to save rankings and outputs")
    parser.add_argument("--save_vectors", action="store_true", help="Also save per-layer steering vectors")
    parser.add_argument(
        "--vector_kind",
        type=str,
        default="delta",
        choices=["delta", "cohen_d"],
        help="Which vector to save for steering (delta recommended)",
    )
    parser.add_argument("--top_k_dims", type=int, default=0, help="If >0, keep only top-K dims by |value| when saving vectors")

    args = parser.parse_args()

    print(
        f"[find_leak_layers] Starting with:\n"
        f"  activations_dir: {args.activations_dir}\n"
        f"  results_json:    {args.results_json}\n"
        f"  segment:         {args.segment}\n"
        f"  thresholds:      {args.thresholds if args.thresholds else [args.threshold]}\n"
        f"  min_examples:    {args.min_examples}\n"
        f"  output_dir:      {args.output_dir}\n"
        f"  save_vectors:    {args.save_vectors} ({args.vector_kind}, top_k={args.top_k_dims})"
    )

    # Create output dir early so users can see progress
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception:
        pass

    labels = load_labels_from_results(args.results_json)
    input_lengths = load_input_lengths_from_results(args.results_json)
    print(f"[find_leak_layers] Loaded {len(labels)} labels and {len(input_lengths)} input lengths")

    # Collect vectors once and compute effects once
    print("[find_leak_layers] Collecting vectors...")
    by_layer_pos, by_layer_neg, hidden_dim_by_layer = collect_layer_vectors(
        activations_dir=args.activations_dir,
        labels=labels,
        input_lengths=input_lengths,
        segment=args.segment,
    )
    total_layers = len(hidden_dim_by_layer)
    total_pos = sum(len(v) for v in by_layer_pos.values())
    total_neg = sum(len(v) for v in by_layer_neg.values())
    print(f"[find_leak_layers] Collection done: layers={total_layers}, pos={total_pos}, neg={total_neg}")

    print("[find_leak_layers] Computing effects...")
    base_rows, vectors = compute_layer_effects_once(
        by_layer_pos=by_layer_pos,
        by_layer_neg=by_layer_neg,
        hidden_dim_by_layer=hidden_dim_by_layer,
        min_examples=args.min_examples,
    )
    print(f"[find_leak_layers] Effects computed for {len(base_rows)} layers (min_examples={args.min_examples})")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine thresholds list
    thresholds = args.thresholds if args.thresholds else [args.threshold]

    # Save per-threshold CSV/JSON/plot without recomputing neurons
    for thr in thresholds:
        print(f"[find_leak_layers] Writing outputs for threshold {thr}...")
        thr_rows = rows_for_threshold(base_rows, vectors, thr)
        save_csv_json_plot(thr_rows, args.output_dir, thr)
    print(f"[find_leak_layers] Outputs saved to {args.output_dir}")

    # Optionally save steering vectors once
    if args.save_vectors:
        vectors_dir = os.path.join(args.output_dir, "steering_vectors")
        os.makedirs(vectors_dir, exist_ok=True)
        for layer, vecs in vectors.items():
            base = vecs["delta"] if args.vector_kind == "delta" else vecs["cohen_d"]
            vec = base.astype(np.float32)
            if args.top_k_dims and args.top_k_dims > 0 and args.top_k_dims < vec.shape[0]:
                idx = np.argpartition(np.abs(vec), -args.top_k_dims)[-args.top_k_dims:]
                sparse = np.zeros_like(vec)
                sparse[idx] = vec[idx]
                vec_to_save = sparse
            else:
                vec_to_save = vec
            norm = np.linalg.norm(vec_to_save)
            if norm > 0:
                vec_to_save = vec_to_save / norm
            out_path = os.path.join(vectors_dir, f"steering_vector_layer_{layer}.npy")
            np.save(out_path, vec_to_save)
        print(f"[find_leak_layers] Steering vectors saved to {vectors_dir}")

    # Print top few layers for the first threshold (if any)
    if thresholds:
        example_rows = rows_for_threshold(base_rows, vectors, thresholds[0])
        print("Top layers by density (first 10):")
        for row in example_rows[:10]:
            print(
                f"Layer {row['layer']:>3} | density={row['density']:.4f} | flagged={row['flagged_count']:>5} "
                f"| pos={row['num_pos']:>3} neg={row['num_neg']:>3} | hidden={row['hidden_dim']}"
            )


if __name__ == "__main__":
    main()


