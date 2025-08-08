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

def compute_steering_vectors(positive_activations_by_layer, negative_activations_by_layer, method="mean_diff"):
    """
    Compute steering vectors from positive and negative examples across layers
    
    Args:
        positive_activations_by_layer: dict[layer_idx] -> list of activation matrices
        negative_activations_by_layer: dict[layer_idx] -> list of activation matrices  
        method: "mean_diff" or "pca_diff"
    
    Returns:
        dict[layer_idx] -> steering_vector
    """
    steering_vectors = {}
    
    for layer_idx in positive_activations_by_layer.keys():
        pos_activations = positive_activations_by_layer[layer_idx]
        neg_activations = negative_activations_by_layer[layer_idx]
        
        if not pos_activations or not neg_activations:
            print(f"Warning: No activations for layer {layer_idx}, skipping")
            continue
        
        print(f"Computing steering vector for layer {layer_idx}")
        print(f"  Positive examples: {len(pos_activations)}")
        print(f"  Negative examples: {len(neg_activations)}")
        
        if method == "mean_diff":
            # Average across tokens and examples
            pos_avg = np.mean([act.mean(axis=0) for act in pos_activations if act.size > 0], axis=0)
            neg_avg = np.mean([act.mean(axis=0) for act in neg_activations if act.size > 0], axis=0)
            
            # Steering vector: direction from negative to positive
            steering_vector = pos_avg - neg_avg
            
            # Normalize
            steering_vector = steering_vector / np.linalg.norm(steering_vector)
            
        elif method == "pca_diff":
            # More sophisticated approach using PCA
            from sklearn.decomposition import PCA
            
            # Flatten all token activations
            pos_flat = np.vstack([act for act in pos_activations if act.size > 0])
            neg_flat = np.vstack([act for act in neg_activations if act.size > 0])
            
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
        
        steering_vectors[layer_idx] = steering_vector
        print(f"  Computed steering vector with magnitude {np.linalg.norm(steering_vector):.4f}")
    
    return steering_vectors

def analyze_steering_quality(steering_vectors, positive_activations_by_layer, negative_activations_by_layer):
    """
    Analyze the quality of computed steering vectors by measuring separability
    """
    analysis = {}
    
    for layer_idx, steering_vector in steering_vectors.items():
        pos_activations = positive_activations_by_layer[layer_idx]
        neg_activations = negative_activations_by_layer[layer_idx]
        
        if not pos_activations or not neg_activations:
            continue
        
        # Project activations onto steering vector
        pos_projections = []
        neg_projections = []
        
        for act in pos_activations:
            if act.size > 0:
                act_mean = act.mean(axis=0)
                projection = np.dot(act_mean, steering_vector)
                pos_projections.append(projection)
        
        for act in neg_activations:
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
            
            analysis[layer_idx] = {
                'pos_mean_projection': float(pos_mean_proj),
                'neg_mean_projection': float(neg_mean_proj),
                'pos_std_projection': float(pos_std_proj),
                'neg_std_projection': float(neg_std_proj),
                'separation_score': float(separation)
            }
            
            print(f"Layer {layer_idx} separation analysis:")
            print(f"  Positive mean projection: {pos_mean_proj:.4f} ± {pos_std_proj:.4f}")
            print(f"  Negative mean projection: {neg_mean_proj:.4f} ± {neg_std_proj:.4f}")
            print(f"  Separation score: {separation:.4f}")
    
    return analysis

def visualize_steering_analysis(analysis, output_dir):
    """Create visualizations of steering vector quality"""
    if not analysis:
        return
    
    layers = list(analysis.keys())
    separation_scores = [analysis[layer]['separation_score'] for layer in layers]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(layers, separation_scores, 'o-')
    plt.xlabel('Layer Index')
    plt.ylabel('Separation Score')
    plt.title('Steering Vector Quality by Layer')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    pos_means = [analysis[layer]['pos_mean_projection'] for layer in layers]
    neg_means = [analysis[layer]['neg_mean_projection'] for layer in layers]
    
    plt.plot(layers, pos_means, 'o-', label='Leaky Examples', color='red')
    plt.plot(layers, neg_means, 'o-', label='Non-leaky Examples', color='green')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Projection onto Steering Vector')
    plt.title('Activation Projections by Layer')
    plt.legend()
    plt.grid(True)
    
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
    output_dir = "/workspace/psbs-research-project/leak_eval/results/steering_vectors"
    
    # Configuration
    target_layers = [15]  # Only layer 15 since that's what you have
    method = "mean_diff"
    
    print(f"🎯 Generating steering vectors for 'leaky thoughts' phenomenon")
    print(f"Activations directory: {activations_dir}")
    print(f"Classification data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Target layers: {target_layers}")
    
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
    
    # Load pre-computed activations
    print(f"\n📥 Loading pre-computed activations...")
    try:
        activations_data, metadata = load_activations_from_npz_files(activations_dir, target_layers)
    except Exception as e:
        print(f"❌ Error loading activations: {e}")
        return
    
    if not activations_data:
        print("❌ No activation data loaded!")
        return
    
    # Classify activations by leakiness
    print(f"\n🔍 Classifying activations by leakiness...")
    positive_activations_by_layer, negative_activations_by_layer = classify_activations_by_leakiness(
        activations_data, data
    )
    
    # Check if we have data for computing steering vectors
    layers_with_data = set(positive_activations_by_layer.keys()) & set(negative_activations_by_layer.keys())
    layers_with_both = [l for l in layers_with_data if positive_activations_by_layer[l] and negative_activations_by_layer[l]]
    
    if not layers_with_both:
        print("❌ No layers have both leaky and non-leaky activations!")
        print("Available positive activations:", {k: len(v) for k, v in positive_activations_by_layer.items()})
        print("Available negative activations:", {k: len(v) for k, v in negative_activations_by_layer.items()})
        return
    
    print(f"Layers with both leaky and non-leaky data: {sorted(layers_with_both)}")
    
    # Compute steering vectors
    print(f"\n🧮 Computing steering vectors using method: {method}")
    steering_vectors = compute_steering_vectors(
        positive_activations_by_layer, 
        negative_activations_by_layer,
        method=method
    )
    
    if not steering_vectors:
        print("❌ No steering vectors could be computed!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save steering vectors
    print(f"\n💾 Saving steering vectors...")
    for layer_idx, vector in steering_vectors.items():
        output_path = os.path.join(output_dir, f"steering_vector_layer_{layer_idx}.npy")
        np.save(output_path, vector)
        print(f"Saved steering vector for layer {layer_idx} to {output_path}")
        print(f"  Shape: {vector.shape}, Magnitude: {np.linalg.norm(vector):.4f}")
    
    # Save metadata
    metadata = {
        'activations_dir': activations_dir,
        'target_layers': target_layers,
        'method': method,
        'source_data_file': data_file,
        'leaky_examples_count': len(leaky_examples),
        'non_leaky_examples_count': len(non_leaky_examples),
        'positive_activations_count': {k: len(v) for k, v in positive_activations_by_layer.items()},
        'negative_activations_count': {k: len(v) for k, v in negative_activations_by_layer.items()},
        'steering_vector_shapes': {k: list(v.shape) for k, v in steering_vectors.items()},
        'layers_with_data': layers_with_both,
        'description': 'Steering vectors to increase leaky thoughts (PII leakage in reasoning section)'
    }
    
    metadata_path = os.path.join(output_dir, 'steering_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Quality analysis
    print(f"\n📊 Running quality analysis...")
    analysis = analyze_steering_quality(
        steering_vectors, 
        positive_activations_by_layer, 
        negative_activations_by_layer
    )
    
    if analysis:
        analysis_path = os.path.join(output_dir, 'steering_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to {analysis_path}")
        
        # Create visualizations
        try:
            visualize_steering_analysis(analysis, output_dir)
            print(f"Saved visualization to {os.path.join(output_dir, 'steering_analysis.png')}")
        except ImportError:
            print("Matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        # Find best layers
        best_layers = sorted(analysis.keys(), key=lambda x: analysis[x]['separation_score'], reverse=True)
        print(f"\n🏆 Best layers by separation score:")
        for i, layer in enumerate(best_layers[:3]):
            score = analysis[layer]['separation_score']
            print(f"  {i+1}. Layer {layer}: {score:.4f}")
    
    print(f"\n✅ Steering vector generation complete!")
    print(f"Generated steering vectors for layers: {list(steering_vectors.keys())}")
    print(f"Output directory: {output_dir}")
    
    if analysis and best_layers:
        best_layer = best_layers[0]
        best_score = analysis[best_layer]['separation_score']
        vector_path = os.path.join(output_dir, f"steering_vector_layer_{best_layer}.npy")
        
        print(f"\n🎯 Recommended usage (best layer: {best_layer}, score: {best_score:.4f}):")
        print(f"python modified_eval_cp.py \\")
        print(f"  --model deepseek-r1-distill-qwen-32b \\")
        print(f"  --steering_vector_path {vector_path} \\")
        print(f"  --steering_layers {best_layer} \\")
        print(f"  --steering_strength 1.0 \\")
        print(f"  # ... other arguments")
        
        print(f"\n📝 To try multiple layers:")
        for layer in best_layers[:3]:
            layer_path = os.path.join(output_dir, f"steering_vector_layer_{layer}.npy")
            print(f"  --steering_vector_path {layer_path} --steering_layers {layer}")

if __name__ == "__main__":
    main()