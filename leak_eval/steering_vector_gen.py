import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import glob
import pandas as pd

def load_activations_from_npz_files(activations_dir, target_layers=None):
    """
    Load pre-computed activations from .npz files with format: layer_X_example_Y.npz
    
    Args:
        activations_dir: Directory containing .npz activation files
        target_layers: List of layer indices to load (if None, loads all available)
    
    Returns:
        activations_data: dict mapping (layer_idx, example_idx) -> activation_matrix
        metadata: dict with information about loaded activations
    """
    print(f"Loading activations from: {activations_dir}")
    
    if not os.path.exists(activations_dir):
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    npz_files = glob.glob(os.path.join(activations_dir, "layer_*_example_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found matching pattern 'layer_*_example_*.npz' in {activations_dir}")
    
    print(f"Found {len(npz_files)} .npz files")
    
    activations_data = {}
    all_available_layers = set()
    
    for npz_file in npz_files:
        filename = os.path.basename(npz_file)
        print(f"Loading {filename}...")
        
        # Parse filename: layer_15_example_0.npz -> layer=15, example=0
        try:
            parts = filename.replace('.npz', '').split('_')
            layer_idx = int(parts[1])  # parts[1] should be the layer number
            example_idx = int(parts[3])  # parts[3] should be the example number
            
            all_available_layers.add(layer_idx)
            
            if target_layers is None or layer_idx in target_layers:
                # Load the npz file
                data = np.load(npz_file, allow_pickle=True)
                
                # Look for activation data - it might be stored under different keys
                activation_data = None
                
                # Try common key patterns
                possible_keys = [
                    'activations',
                    f'layer_{layer_idx}_activations', 
                    f'layer_{layer_idx}',
                    'activation',
                    list(data.keys())[0] if len(data.keys()) == 1 else None  # If only one key, use it
                ]
                
                for key in possible_keys:
                    if key and key in data.keys():
                        activation_data = data[key]
                        break
                
                if activation_data is None:
                    print(f"  Available keys: {list(data.keys())}")
                    print(f"  Warning: Could not find activation data in {filename}")
                    continue
                
                # Handle different possible formats
                if isinstance(activation_data, np.ndarray):
                    if activation_data.dtype == object:
                        # If it's an object array, try to extract the actual array
                        try:
                            activation_data = activation_data.item()
                        except:
                            pass
                    
                    # Ensure it's a proper numpy array
                    if hasattr(activation_data, 'shape') and activation_data.size > 0:
                        activations_data[(layer_idx, example_idx)] = activation_data
                        print(f"  Layer {layer_idx}, Example {example_idx}: shape {activation_data.shape}")
                    else:
                        print(f"  Warning: Invalid activation data in {filename}")
                else:
                    print(f"  Warning: Unexpected data type in {filename}: {type(activation_data)}")
                        
        except (ValueError, IndexError) as e:
            print(f"  Error parsing filename {filename}: {e}")
            print(f"  Expected format: layer_X_example_Y.npz")
            continue
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    available_layers = sorted(list(all_available_layers))
    metadata = {
        'total_files': len(npz_files),
        'loaded_files': len(activations_data),
        'available_layers': available_layers,
        'target_layers': target_layers if target_layers else available_layers
    }
    
    print(f"Successfully loaded activations from {len(activations_data)} files")
    print(f"Available layers: {available_layers}")
    
    return activations_data, metadata

def classify_activations_by_leakiness(activations_data, classification_data):
    """
    Classify activation files as leaky or non-leaky based on the classification data
    
    Args:
        activations_data: dict mapping (layer_idx, example_idx) -> activation_matrix
        classification_data: list of examples with leakiness classifications
    
    Returns:
        positive_activations_by_layer: dict[layer_idx] -> list of leaky activation matrices
        negative_activations_by_layer: dict[layer_idx] -> list of non-leaky activation matrices
    """
    
    print(f"Classifying {len(activations_data)} activation files by leakiness...")
    
    # Get all available layers from the activations
    all_layers = set()
    for (layer_idx, example_idx) in activations_data.keys():
        all_layers.add(layer_idx)
    
    # Initialize dictionaries for each layer
    positive_activations_by_layer = {}
    negative_activations_by_layer = {}
    
    for layer_idx in all_layers:
        positive_activations_by_layer[layer_idx] = []
        negative_activations_by_layer[layer_idx] = []
    
    # Classify each example
    leaky_count = 0
    non_leaky_count = 0
    
    for (layer_idx, example_idx), activation in activations_data.items():
        # Make sure we have classification data for this example
        if example_idx < len(classification_data):
            example = classification_data[example_idx]
            reasoning_leaks = example.get('pii_leaks', {}).get('reasoning_bin', [0])[0]
            
            if reasoning_leaks > 0:
                positive_activations_by_layer[layer_idx].append(activation)
                if layer_idx == list(all_layers)[0]:  # Only count once per example
                    leaky_count += 1
                    print(f"  Example {example_idx}: LEAKY (layer {layer_idx}) - {len(example.get('pii_leaks', {}).get('leaks_reasoning', [[]])[0])} leaks in reasoning")
            else:
                negative_activations_by_layer[layer_idx].append(activation)
                if layer_idx == list(all_layers)[0]:  # Only count once per example
                    non_leaky_count += 1
                    print(f"  Example {example_idx}: NON-LEAKY (layer {layer_idx}) - no leaks in reasoning")
        else:
            print(f"  Warning: No classification data for example {example_idx}, skipping")
    
    print(f"Classification complete:")
    print(f"  {leaky_count} leaky examples")
    print(f"  {non_leaky_count} non-leaky examples")
    print(f"  Available layers: {sorted(all_layers)}")
    
    # Print per-layer counts
    for layer_idx in sorted(all_layers):
        pos_count = len(positive_activations_by_layer[layer_idx])
        neg_count = len(negative_activations_by_layer[layer_idx])
        print(f"  Layer {layer_idx}: {pos_count} leaky, {neg_count} non-leaky")
    
    return positive_activations_by_layer, negative_activations_by_layer

def load_layer_rankings(rankings_csv_path):
    """
    Load pre-computed layer rankings from CSV file
    
    Args:
        rankings_csv_path: Path to the CSV file with layer rankings
    
    Returns:
        list of layer indices ordered by rank (best first)
    """
    import pandas as pd
    
    print(f"Loading layer rankings from: {rankings_csv_path}")
    
    if not os.path.exists(rankings_csv_path):
        raise FileNotFoundError(f"Rankings file not found: {rankings_csv_path}")
    
    try:
        df = pd.read_csv(rankings_csv_path)
        print(f"Loaded CSV with columns: {list(df.columns)}")
        print(f"CSV shape: {df.shape}")
        
        # Look for layer column - try different possible names
        layer_col = None
        for col_name in ['layer', 'Layer', 'layer_idx', 'layer_index', 'Layer Index']:
            if col_name in df.columns:
                layer_col = col_name
                break
        
        if layer_col is None:
            # If no clear layer column, assume first column is layer
            layer_col = df.columns[0]
            print(f"No clear layer column found, using first column: {layer_col}")
        
        # Extract layers in order (assuming CSV is already sorted by ranking)
        layers = df[layer_col].tolist()
        
        # Convert to integers if they're not already
        layers = [int(layer) for layer in layers]
        
        print(f"Loaded rankings for {len(layers)} layers")
        print(f"Top 10 layers: {layers[:10]}")
        
        return layers
        
    except Exception as e:
        print(f"Error loading rankings CSV: {e}")
        print("Expected CSV format: first row = headers, one column should contain layer indices")
        raise

def compute_steering_vector_for_layer(positive_activations, negative_activations, method="mean_diff"):
    """
    Compute steering vector for a specific layer
    
    Args:
        positive_activations: list of activation matrices for leaky examples
        negative_activations: list of activation matrices for non-leaky examples
        method: "mean_diff" or "pca_diff"
    
    Returns:
        steering_vector: numpy array
    """
    if not positive_activations or not negative_activations:
        raise ValueError("Need both positive and negative activations")
    
    if method == "mean_diff":
        # Average across tokens and examples
        pos_avg = np.mean([act.mean(axis=0) for act in positive_activations if act.size > 0], axis=0)
        neg_avg = np.mean([act.mean(axis=0) for act in negative_activations if act.size > 0], axis=0)
        
        # Steering vector: direction from negative to positive
        steering_vector = pos_avg - neg_avg
        
        # Normalize
        steering_vector = steering_vector / np.linalg.norm(steering_vector)
        
    elif method == "pca_diff":
        # More sophisticated approach using PCA
        from sklearn.decomposition import PCA
        
        # Flatten all token activations
        pos_flat = np.vstack([act for act in positive_activations if act.size > 0])
        neg_flat = np.vstack([act for act in negative_activations if act.size > 0])
        
        # Compute mean difference
        pos_mean = pos_flat.mean(axis=0)
        neg_mean = neg_flat.mean(axis=0)
        
        # Use first PCA component of the difference as steering direction
        combined = np.vstack([pos_flat, neg_flat])
        pca = PCA(n_components=1)
        pca.fit(combined)
        
        # Project mean difference onto first PC
        mean_diff = pos_mean - neg_mean
        steering_vector = pca.components_[0]
        
        # Ensure steering vector points in the right direction
        if np.dot(steering_vector, mean_diff) < 0:
            steering_vector = -steering_vector
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return steering_vector

def analyze_steering_quality(steering_vector, positive_activations, negative_activations):
    """
    Analyze the quality of a computed steering vector by measuring separability
    """
    # Project activations onto steering vector
    pos_projections = []
    neg_projections = []
    
    for act in positive_activations:
        if act.size > 0:
            act_mean = act.mean(axis=0)
            projection = np.dot(act_mean, steering_vector)
            pos_projections.append(projection)
    
    for act in negative_activations:
        if act.size > 0:
            act_mean = act.mean(axis=0)
            projection = np.dot(act_mean, steering_vector)
            neg_projections.append(projection)
    
    if pos_projections and neg_projections:
        pos_mean_proj = np.mean(pos_projections)
        neg_mean_proj = np.mean(neg_projections)
        pos_std_proj = np.std(pos_projections)
        neg_std_proj = np.std(neg_projections)
        
        # Separation metric (difference in means relative to combined std)
        separation = abs(pos_mean_proj - neg_mean_proj) / (pos_std_proj + neg_std_proj + 1e-8)
        
        analysis = {
            'pos_mean_projection': float(pos_mean_proj),
            'neg_mean_projection': float(neg_mean_proj),
            'pos_std_projection': float(pos_std_proj),
            'neg_std_projection': float(neg_std_proj),
            'separation_score': float(separation)
        }
        
        return analysis
    
    return None

def visualize_steering_analysis(layer_rankings, computed_vectors_info, output_dir):
    """Create visualizations of steering vector results"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Layer rankings used
    plt.subplot(1, 3, 1)
    top_10_layers = layer_rankings[:10]
    ranking_positions = list(range(1, len(top_10_layers) + 1))
    
    plt.bar(ranking_positions, top_10_layers)
    plt.xlabel('Ranking Position')
    plt.ylabel('Layer Index')
    plt.title('Top 10 Layers by Pre-computed Ranking')
    plt.xticks(ranking_positions)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Successfully computed vectors
    plt.subplot(1, 3, 2)
    if computed_vectors_info:
        computed_layers = list(computed_vectors_info.keys())
        vector_magnitudes = [computed_vectors_info[layer]['magnitude'] for layer in computed_layers]
        
        plt.bar(range(len(computed_layers)), vector_magnitudes)
        plt.xlabel('Vector Index')
        plt.ylabel('Vector Magnitude')
        plt.title(f'Steering Vector Magnitudes\n({len(computed_layers)} vectors computed)')
        plt.xticks(range(len(computed_layers)), [f'L{layer}' for layer in computed_layers], rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Layer availability analysis
    plt.subplot(1, 3, 3)
    if computed_vectors_info:
        computed_layers = list(computed_vectors_info.keys())
        separation_scores = [computed_vectors_info[layer]['separation_score'] for layer in computed_layers]
        
        plt.plot(computed_layers, separation_scores, 'o-')
        plt.xlabel('Layer Index')
        plt.ylabel('Separation Score')
        plt.title('Separation Scores for Computed Vectors')
        plt.grid(True)
        
        # Highlight best layer
        if separation_scores:
            best_idx = np.argmax(separation_scores)
            best_layer = computed_layers[best_idx]
            plt.plot(best_layer, separation_scores[best_idx], 'ro', markersize=10, label=f'Best: Layer {best_layer}')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def classify_leaky_examples(data):
    """
    Classify examples as leaky or non-leaky based on PII leakage in reasoning
    
    Returns:
        leaky_examples: list of examples with PII leakage in reasoning
        non_leaky_examples: list of examples without PII leakage in reasoning
    """
    leaky_examples = []
    non_leaky_examples = []
    
    for example in data:
        # Check if there's PII leakage in reasoning (the <think> section)
        reasoning_leaks = example.get('pii_leaks', {}).get('reasoning_bin', [0])[0]
        
        # For deepseek models, we care about leakage in the reasoning section
        if reasoning_leaks > 0:
            leaky_examples.append(example)
            print(f"Leaky example found: {len(example.get('pii_leaks', {}).get('leaks_reasoning', [[]])[0])} leaks in reasoning")
        else:
            non_leaky_examples.append(example)
            print(f"Non-leaky example found: no leaks in reasoning")
    
    return leaky_examples, non_leaky_examples

def main():
    # Fixed paths for your setup
    data_file = "/workspace/psbs-research-project/leak_eval/results/test_pipeline_3sample.json"
    activations_dir = "/workspace/psbs-research-project/leak_eval/results/activations"
    rankings_csv_path = "/workspace/psbs-research-project/leak_eval/results/full_avg_rankings.csv"
    output_dir = "/workspace/psbs-research-project/leak_eval/results/steering_vectors"
    
    # Configuration
    method = "mean_diff"
    top_n_layers = 10  # Number of top layers to compute vectors for
    
    print(f"🎯 Generating steering vectors for 'leaky thoughts' phenomenon")
    print(f"Activations directory: {activations_dir}")
    print(f"Classification data file: {data_file}")
    print(f"Layer rankings file: {rankings_csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Method: {method}")
    print(f"Computing vectors for top {top_n_layers} layers")
    
    # Load layer rankings from CSV
    print(f"\n📊 Loading pre-computed layer rankings...")
    try:
        layer_rankings = load_layer_rankings(rankings_csv_path)
        target_layers = layer_rankings[:top_n_layers]
        print(f"Target layers (top {top_n_layers}): {target_layers}")
    except Exception as e:
        print(f"❌ Error loading rankings: {e}")
        print("Falling back to loading all available layers...")
        target_layers = None
    
    # Load classification data to determine which activations are leaky
    print(f"\n📊 Loading classification data...")
    with open(data_file, 'r') as f:
        result_data = json.load(f)
    
    data = result_data.get('data', [])
    leaky_examples, non_leaky_examples = classify_leaky_examples(data)
    
    print(f"Found {len(leaky_examples)} leaky examples")
    print(f"Found {len(non_leaky_examples)} non-leaky examples")
    
    if len(leaky_examples) == 0:
        print("❌ No leaky examples found! Cannot generate steering vectors.")
        print("Make sure your data has examples with reasoning_bin > 0")
        return
    
    if len(non_leaky_examples) == 0:
        print("❌ No non-leaky examples found! Cannot generate steering vectors.")
        print("Make sure your data has examples with reasoning_bin = 0")
        return
    
    # Load pre-computed activations for target layers
    print(f"\n📥 Loading pre-computed activations for target layers...")
    try:
        activations_data, metadata = load_activations_from_npz_files(activations_dir, target_layers=target_layers)
    except Exception as e:
        print(f"❌ Error loading activations: {e}")
        return
    
    if not activations_data:
        print("❌ No activation data loaded!")
        return
    
    available_layers = metadata['available_layers']
    print(f"Successfully loaded activations from {len(available_layers)} layers: {available_layers}")
    
    # If we have target layers, show which ones we actually found
    if target_layers:
        found_target_layers = [layer for layer in target_layers if layer in available_layers]
        missing_layers = [layer for layer in target_layers if layer not in available_layers]
        print(f"Found target layers: {found_target_layers}")
        if missing_layers:
            print(f"Missing target layers: {missing_layers}")
    
    # Classify activations by leakiness
    print(f"\n🔍 Classifying activations by leakiness...")
    positive_activations_by_layer, negative_activations_by_layer = classify_activations_by_leakiness(
        activations_data, data
    )
    
    # Check which layers have both positive and negative examples
    layers_with_data = set(positive_activations_by_layer.keys()) & set(negative_activations_by_layer.keys())
    layers_with_both = [l for l in layers_with_data if positive_activations_by_layer[l] and negative_activations_by_layer[l]]
    
    if not layers_with_both:
        print("❌ No layers have both leaky and non-leaky activations!")
        print("Available positive activations:", {k: len(v) for k, v in positive_activations_by_layer.items()})
        print("Available negative activations:", {k: len(v) for k, v in negative_activations_by_layer.items()})
        return
    
    print(f"Layers with both leaky and non-leaky data: {sorted(layers_with_both)}")
    
    # Compute steering vectors for all available layers (from the target set)
    print(f"\n🧮 Computing steering vectors for available layers...")
    computed_vectors = {}
    computed_vectors_info = {}
    
    # Process layers in ranking order if we have rankings, otherwise by layer index
    if target_layers:
        process_order = [layer for layer in target_layers if layer in layers_with_both]
    else:
        process_order = sorted(layers_with_both)
    
    for layer_idx in process_order:
        print(f"\nProcessing Layer {layer_idx}...")
        try:
            # Compute steering vector
            steering_vector = compute_steering_vector_for_layer(
                positive_activations_by_layer[layer_idx],
                negative_activations_by_layer[layer_idx],
                method=method
            )
            
            # Analyze quality
            analysis = analyze_steering_quality(
                steering_vector,
                positive_activations_by_layer[layer_idx],
                negative_activations_by_layer[layer_idx]
            )
            
            computed_vectors[layer_idx] = steering_vector
            computed_vectors_info[layer_idx] = {
                'magnitude': float(np.linalg.norm(steering_vector)),
                'separation_score': analysis['separation_score'] if analysis else 0.0,
                'analysis': analysis
            }
            
            print(f"  ✅ Layer {layer_idx}: magnitude={np.linalg.norm(steering_vector):.4f}")
            if analysis:
                print(f"     Separation score: {analysis['separation_score']:.4f}")
                
        except Exception as e:
            print(f"  ❌ Error computing vector for layer {layer_idx}: {e}")
            continue
    
    if not computed_vectors:
        print("❌ No steering vectors could be computed!")
        return
    
    print(f"\n✅ Successfully computed {len(computed_vectors)} steering vectors")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all computed steering vectors
    print(f"\n💾 Saving steering vectors...")
    for layer_idx, vector in computed_vectors.items():
        output_path = os.path.join(output_dir, f"steering_vector_layer_{layer_idx}.npy")
        np.save(output_path, vector)
        print(f"  Layer {layer_idx}: saved to {os.path.basename(output_path)}")
    
    # Find the best layer by separation score
    best_layer = max(computed_vectors_info.keys(), key=lambda x: computed_vectors_info[x]['separation_score'])
    best_score = computed_vectors_info[best_layer]['separation_score']
    best_vector_path = os.path.join(output_dir, f"steering_vector_layer_{best_layer}.npy")
    
    print(f"\n🏆 Best layer by separation score: Layer {best_layer} (score: {best_score:.4f})")
    
    # Rank all computed vectors by separation score
    ranked_layers = sorted(computed_vectors_info.items(), key=lambda x: x[1]['separation_score'], reverse=True)
    
    print(f"\n📊 All computed layers ranked by separation score:")
    for i, (layer, info) in enumerate(ranked_layers, 1):
        ranking_pos = "N/A"
        if target_layers and layer in target_layers:
            ranking_pos = target_layers.index(layer) + 1
        print(f"  {i:2d}. Layer {layer:2d} (CSV rank #{ranking_pos:>3}): separation={info['separation_score']:.4f}, magnitude={info['magnitude']:.4f}")
    
    # Save metadata
    metadata = {
        'activations_dir': activations_dir,
        'rankings_csv_path': rankings_csv_path,
        'target_layers': target_layers if target_layers else available_layers,
        'method': method,
        'source_data_file': data_file,
        'leaky_examples_count': len(leaky_examples),
        'non_leaky_examples_count': len(non_leaky_examples),
        'computed_layers': list(computed_vectors.keys()),
        'best_layer': int(best_layer),
        'best_separation_score': float(best_score),
        'layer_rankings_from_csv': target_layers,
        'computed_vectors_ranking': [(int(layer), float(info['separation_score'])) for layer, info in ranked_layers],
        'positive_activations_count': {int(k): len(v) for k, v in positive_activations_by_layer.items() if k in computed_vectors},
        'negative_activations_count': {int(k): len(v) for k, v in negative_activations_by_layer.items() if k in computed_vectors},
        'steering_vector_info': {int(k): v for k, v in computed_vectors_info.items()},
        'description': f'Steering vectors computed for top {len(computed_vectors)} layers from pre-computed rankings'
    }
    
    metadata_path = os.path.join(output_dir, 'steering_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n💾 Saved metadata to {os.path.basename(metadata_path)}")
    
    # Save detailed analysis for all layers
    analysis_data = {
        'source_rankings_csv': rankings_csv_path,
        'computed_layers': list(computed_vectors.keys()),
        'best_layer': int(best_layer),
        'layer_analyses': {int(k): v for k, v in computed_vectors_info.items()},
        'ranking_comparison': []
    }
    
    # Compare CSV ranking vs separation score ranking
    if target_layers:
        for layer in computed_vectors.keys():
            csv_rank = target_layers.index(layer) + 1 if layer in target_layers else None
            sep_rank = next(i for i, (l, _) in enumerate(ranked_layers, 1) if l == layer)
            analysis_data['ranking_comparison'].append({
                'layer': int(layer),
                'csv_rank': csv_rank,
                'separation_rank': sep_rank,
                'separation_score': computed_vectors_info[layer]['separation_score']
            })
    
    analysis_path = os.path.join(output_dir, 'steering_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"💾 Saved detailed analysis to {os.path.basename(analysis_path)}")
    
    # Create visualizations
    try:
        visualize_steering_analysis(target_layers or available_layers, computed_vectors_info, output_dir)
        print(f"💾 Saved visualization to steering_analysis.png")
    except Exception as e:
        print(f"⚠️  Error creating visualization: {e}")
    
    print(f"\n✅ Steering vector generation complete!")
    print(f"🎯 Best layer: {best_layer} (separation score: {best_score:.4f})")
    print(f"📁 Output directory: {output_dir}")
    print(f"🗂️ Generated {len(computed_vectors)} steering vectors")
    
    print(f"\n🚀 Recommended usage (best layer):")
    print(f"python modified_eval_cp.py \\")
    print(f"  --model deepseek-r1-distill-qwen-32b \\")
    print(f"  --steering_vector_path {best_vector_path} \\")
    print(f"  --steering_layers {best_layer} \\")
    print(f"  --steering_strength 1.0 \\")
    print(f"  # ... other arguments")
    
    print(f"\n🔄 Top 5 alternatives by separation score:")
    for i, (layer, info) in enumerate(ranked_layers[1:6], 2):  # Skip best, show next 5
        layer_path = os.path.join(output_dir, f"steering_vector_layer_{layer}.npy")
        csv_rank = target_layers.index(layer) + 1 if target_layers and layer in target_layers else "N/A"
        print(f"  {i}. Layer {layer} (CSV rank #{csv_rank}, sep score: {info['separation_score']:.4f})")
        print(f"     --steering_vector_path {layer_path} --steering_layers {layer}")

if __name__ == "__main__":
    main()
    
    # Quality analysis for the best layer