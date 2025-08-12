#!/usr/bin/env python3
"""
Analysis script to compare activation patterns across different layers
for steering vector selection.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_activation_files(activation_dir: str) -> Tuple[Dict, List[int]]:
    """Load activation .npz files directly from directory and return data dict and layer list"""
    activation_path = Path(activation_dir)
    
    if not activation_path.exists():
        raise ValueError(f"Activation directory not found: {activation_dir}")
    
    # Find all .npz files
    activation_files = list(activation_path.glob("*.npz"))
    
    if not activation_files:
        raise ValueError(f"No .npz files found in {activation_dir}")
    
    print(f"Found {len(activation_files)} activation files")
    
    # Group files by layer (extract layer numbers from filenames)
    layer_files = {}
    layers = set()
    
    for file_path in activation_files:
        # Try to extract layer number from filename
        filename = file_path.stem
        layer_num = None
        
        # Look for patterns like "layer_12", "layer12", "l12", etc.
        import re
        patterns = [
            r'layer_?(\d+)',
            r'l(\d+)',
            r'(\d+)_layer',
            r'(\d+)$'  # number at the end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                layer_num = int(match.group(1))
                break
        
        if layer_num is None:
            print(f"Warning: Could not extract layer number from {filename}, skipping")
            continue
            
        layers.add(layer_num)
        if layer_num not in layer_files:
            layer_files[layer_num] = []
        layer_files[layer_num].append(file_path)
    
    layers = sorted(list(layers))
    
    # Create data structure similar to original format
    data = {'data': []}
    
    # Process each file and create entries
    for layer in layers:
        for file_path in layer_files[layer]:
            try:
                # Load and inspect the .npz file
                activation_data = np.load(file_path)
                print(f"Processing {file_path.name} (Layer {layer})")
                print(f"  Keys in file: {list(activation_data.keys())}")
                
                file_entry = {
                    'file': str(file_path.name),
                    'layer': layer,
                    'activations': {
                        f'layer_{layer}': {
                            'file': str(file_path.name),
                            'activation_data': activation_data,
                            'file_path': file_path
                        }
                    }
                }
                
                data['data'].append(file_entry)
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    return data, layers, activation_path

def extract_averaged_activations(data: Dict, layers: List[int], activation_base_path: Path = None) -> Dict[str, Dict[int, np.ndarray]]:
    """Extract averaged activations for different segments across layers"""
    
    segments = ['full_avg', 'reasoning_avg', 'answer_avg']
    layer_activations = {segment: {layer: [] for layer in layers} for segment in segments}
    
    # Keep track of what keys we find across all files
    found_keys = set()
    
    for example in data['data']:
        if 'activations' not in example:
            continue
            
        layer = example.get('layer')
        if layer is None:
            continue
            
        layer_key = f'layer_{layer}'
        if layer_key in example['activations']:
            activation_info = example['activations'][layer_key]
            
            # Get the activation data
            activation_data = activation_info.get('activation_data')
            if activation_data is None:
                continue
            
            # Record what keys are available
            available_keys = list(activation_data.keys())
            found_keys.update(available_keys)
            
            # Try to extract activations for each segment
            for segment in segments:
                activation_found = False
                
                # Look for exact matches first
                segment_key = f'{segment}_activation'
                if segment_key in activation_data:
                    layer_activations[segment][layer].append(activation_data[segment_key])
                    activation_found = True
                
                # Look for alternative naming patterns
                elif segment == 'full_avg':
                    for key in ['full_avg_activation', 'activation', 'full_activation', 'avg_activation']:
                        if key in activation_data:
                            layer_activations[segment][layer].append(activation_data[key])
                            activation_found = True
                            break
                
                elif segment == 'reasoning_avg':
                    for key in ['reasoning_avg_activation', 'reasoning_activation']:
                        if key in activation_data:
                            layer_activations[segment][layer].append(activation_data[key])
                            activation_found = True
                            break
                
                elif segment == 'answer_avg':
                    for key in ['answer_avg_activation', 'answer_activation']:
                        if key in activation_data:
                            layer_activations[segment][layer].append(activation_data[key])
                            activation_found = True
                            break
                
                # If no specific segment found and there's only one array, use it for all segments
                if not activation_found and len(available_keys) == 1:
                    single_key = available_keys[0]
                    layer_activations[segment][layer].append(activation_data[single_key])
                    activation_found = True
                
                # If still no match, try the first available array for full_avg only
                elif not activation_found and segment == 'full_avg' and available_keys:
                    first_key = available_keys[0]
                    layer_activations[segment][layer].append(activation_data[first_key])
                    activation_found = True
    
    print(f"Keys found across all files: {sorted(found_keys)}")
    
    # Convert lists to numpy arrays and determine the feature dimension
    feature_dims = set()
    for segment in segments:
        for layer in layers:
            if layer_activations[segment][layer]:
                try:
                    arrays = layer_activations[segment][layer]
                    # Handle different shapes - if 1D, add batch dimension
                    processed_arrays = []
                    for arr in arrays:
                        if arr.ndim == 1:
                            processed_arrays.append(arr.reshape(1, -1))
                        elif arr.ndim == 2:
                            processed_arrays.append(arr)
                        else:
                            # For higher dimensions, flatten to 2D
                            processed_arrays.append(arr.reshape(arr.shape[0], -1))
                    
                    if processed_arrays:
                        # Stack along the batch dimension
                        stacked = np.vstack(processed_arrays)
                        layer_activations[segment][layer] = stacked
                        feature_dims.add(stacked.shape[1])
                        print(f"Layer {layer}, {segment}: shape {stacked.shape}")
                except Exception as e:
                    print(f"Warning: Could not stack arrays for layer {layer}, segment {segment}: {e}")
                    layer_activations[segment][layer] = np.empty((0, 1))
            else:
                layer_activations[segment][layer] = None
    
    # Use the most common feature dimension, or default to the largest found
    if feature_dims:
        feature_dim = max(feature_dims)
        print(f"Using feature dimension: {feature_dim}")
    else:
        feature_dim = 1024  # Conservative default
        print(f"No features found, using default dimension: {feature_dim}")
    
    # Set empty arrays for layers with no data
    for segment in segments:
        for layer in layers:
            if layer_activations[segment][layer] is None:
                layer_activations[segment][layer] = np.empty((0, feature_dim))
    
    return layer_activations

def compute_layer_statistics(layer_activations: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
    """Compute statistics for each layer to help identify the best layers"""
    
    stats_data = []
    
    for segment in layer_activations.keys():
        for layer, activations in layer_activations[segment].items():
            if len(activations) == 0:
                continue
                
            # Basic statistics
            mean_activation = np.mean(activations)
            std_activation = np.std(activations)
            max_activation = np.max(activations)
            min_activation = np.min(activations)
            
            # Activation magnitude (L2 norm)
            activation_magnitude = np.mean(np.linalg.norm(activations, axis=1))
            
            # Activation sparsity (percentage of near-zero values)
            sparsity = np.mean(np.abs(activations) < 1e-6) * 100
            
            # Variance explained by first few PCA components
            if len(activations) > 5 and activations.shape[1] > 10:
                try:
                    pca = PCA(n_components=min(10, len(activations)-1))
                    pca.fit(activations)
                    var_explained_10 = np.sum(pca.explained_variance_ratio_)
                    var_explained_3 = np.sum(pca.explained_variance_ratio_[:3])
                except:
                    var_explained_10 = 0
                    var_explained_3 = 0
            else:
                var_explained_10 = 0
                var_explained_3 = 0
            
            # Inter-example variability (how different examples vary)
            if len(activations) > 1:
                pairwise_cosine = cosine_similarity(activations)
                # Average cosine similarity between different examples
                mask = np.triu(np.ones_like(pairwise_cosine, dtype=bool), k=1)
                avg_cosine_similarity = np.mean(pairwise_cosine[mask])
            else:
                avg_cosine_similarity = 1.0
            
            stats_data.append({
                'layer': layer,
                'segment': segment,
                'n_examples': len(activations),
                'mean_activation': mean_activation,
                'std_activation': std_activation,
                'max_activation': max_activation,
                'min_activation': min_activation,
                'activation_magnitude': activation_magnitude,
                'sparsity_percent': sparsity,
                'var_explained_10_components': var_explained_10,
                'var_explained_3_components': var_explained_3,
                'avg_cosine_similarity': avg_cosine_similarity,
                'activation_range': max_activation - min_activation
            })
    
    return pd.DataFrame(stats_data)

def plot_layer_comparison(stats_df: pd.DataFrame, output_dir: Path):
    """Create visualization comparing layers across different metrics"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    segments = stats_df['segment'].unique()
    
    # Key metrics to plot
    metrics = [
        ('activation_magnitude', 'Activation Magnitude'),
        ('std_activation', 'Activation Standard Deviation'),
        ('var_explained_3_components', 'Variance Explained (Top 3 Components)'),
        ('avg_cosine_similarity', 'Average Inter-Example Cosine Similarity'),
        ('sparsity_percent', 'Activation Sparsity (%)')
    ]
    
    for metric_key, metric_name in metrics:
        fig, axes = plt.subplots(1, len(segments), figsize=(5*len(segments), 4))
        if len(segments) == 1:
            axes = [axes]
        
        for i, segment in enumerate(segments):
            segment_data = stats_df[stats_df['segment'] == segment]
            
            if len(segment_data) == 0:
                axes[i].text(0.5, 0.5, f'No data for {segment}', 
                           transform=axes[i].transAxes, ha='center')
                continue
            
            axes[i].plot(segment_data['layer'], segment_data[metric_key], 
                        marker='o', linewidth=2, markersize=6)
            axes[i].set_xlabel('Layer')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{segment.replace("_", " ").title()} - {metric_name}')
            axes[i].grid(True, alpha=0.3)
            
            # Highlight potential good layers (heuristic)
            if len(segment_data) > 0:
                if metric_key in ['activation_magnitude', 'std_activation', 'var_explained_3_components']:
                    # Higher is potentially better
                    top_layers = segment_data.nlargest(min(5, len(segment_data)), metric_key)['layer'].values
                elif metric_key == 'avg_cosine_similarity':
                    # Medium values might be better (not too similar, not too different)
                    median_val = segment_data[metric_key].median()
                    segment_data_copy = segment_data.copy()
                    segment_data_copy['distance_from_median'] = np.abs(segment_data_copy[metric_key] - median_val)
                    top_layers = segment_data_copy.nsmallest(min(5, len(segment_data)), 'distance_from_median')['layer'].values
                else:  # sparsity
                    # Lower might be better
                    top_layers = segment_data.nsmallest(min(5, len(segment_data)), metric_key)['layer'].values
                
                # Mark top layers
                for layer in top_layers[:3]:  # Top 3
                    layer_data = segment_data[segment_data['layer'] == layer]
                    if not layer_data.empty:
                        axes[i].scatter(layer, layer_data[metric_key].iloc[0], 
                                      color='red', s=100, alpha=0.7, zorder=5)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_key}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def find_optimal_layers(stats_df: pd.DataFrame, segment: str = 'reasoning_avg') -> pd.DataFrame:
    """Find optimal layers based on multiple criteria"""
    
    segment_data = stats_df[stats_df['segment'] == segment].copy()
    
    if len(segment_data) == 0:
        print(f"No data found for segment: {segment}")
        return pd.DataFrame()
    
    # Normalize metrics for scoring (0-1 range)
    metrics_to_normalize = [
        'activation_magnitude', 'std_activation', 'var_explained_3_components'
    ]
    
    for metric in metrics_to_normalize:
        min_val = segment_data[metric].min()
        max_val = segment_data[metric].max()
        if max_val > min_val:
            segment_data[f'{metric}_normalized'] = (segment_data[metric] - min_val) / (max_val - min_val)
        else:
            segment_data[f'{metric}_normalized'] = 0.5
    
    # Handle cosine similarity (we want medium values, not too high or low)
    cos_median = segment_data['avg_cosine_similarity'].median()
    segment_data['cosine_score'] = 1 - np.abs(segment_data['avg_cosine_similarity'] - cos_median)
    
    # Handle sparsity (lower is better)
    max_sparsity = segment_data['sparsity_percent'].max()
    min_sparsity = segment_data['sparsity_percent'].min()
    if max_sparsity > min_sparsity:
        segment_data['sparsity_score'] = 1 - (segment_data['sparsity_percent'] - min_sparsity) / (max_sparsity - min_sparsity)
    else:
        segment_data['sparsity_score'] = 0.5
    
    # Composite score (you can adjust weights)
    weights = {
        'activation_magnitude_normalized': 0.25,
        'std_activation_normalized': 0.20,
        'var_explained_3_components_normalized': 0.30,
        'cosine_score': 0.15,
        'sparsity_score': 0.10
    }
    
    segment_data['composite_score'] = sum(
        segment_data[metric] * weight for metric, weight in weights.items()
    )
    
    # Rank layers
    result = segment_data[['layer', 'composite_score', 'activation_magnitude', 
                         'std_activation', 'var_explained_3_components', 
                         'avg_cosine_similarity', 'sparsity_percent']].copy()
    result = result.sort_values('composite_score', ascending=False)
    result['rank'] = range(1, len(result) + 1)
    
    return result

def generate_recommendations(layer_rankings: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Generate layer recommendations based on analysis"""
    
    recommendations = {}
    
    for segment, ranking in layer_rankings.items():
        if len(ranking) == 0:
            continue
            
        # Top layers overall
        top_5 = ranking.head(5)['layer'].tolist()
        
        # Diverse layer selection (spread across depth)
        all_layers = ranking['layer'].tolist()
        n_layers = len(all_layers)
        
        if n_layers > 0:
            # Select layers from different depth ranges
            early_layers = [l for l in all_layers if l < n_layers // 3]
            middle_layers = [l for l in all_layers if n_layers // 3 <= l < 2 * n_layers // 3]
            late_layers = [l for l in all_layers if l >= 2 * n_layers // 3]
            
            diverse_selection = []
            for layer_group in [early_layers, middle_layers, late_layers]:
                if layer_group:
                    # Get best from this group
                    group_ranking = ranking[ranking['layer'].isin(layer_group)]
                    if not group_ranking.empty:
                        diverse_selection.append(int(group_ranking.iloc[0]['layer']))
            
            recommendations[segment] = {
                'top_5_overall': [int(x) for x in top_5],
                'diverse_selection': diverse_selection,
                'best_single': int(ranking.iloc[0]['layer']) if len(ranking) > 0 else None
            }
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Analyze layer activation patterns")
    parser.add_argument("activation_dir", help="Directory containing .npz activation files")
    parser.add_argument("--output_dir", type=str, default="analysis_results", 
                       help="Output directory for analysis results")
    parser.add_argument("--focus_segment", type=str, default="reasoning_avg",
                       choices=["full_avg", "reasoning_avg", "answer_avg"],
                       help="Segment to focus on for recommendations")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)
    
    print(f"🔍 Analyzing activation files in: {args.activation_dir}")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Load activation files
        data, layers, activation_path = load_activation_files(args.activation_dir)
        print(f"Found {len(layers)} layers: {layers}")
        
        # Extract activations
        layer_activations = extract_averaged_activations(data, layers, activation_path)
        
        # Compute statistics
        stats_df = compute_layer_statistics(layer_activations)
        
        if len(stats_df) == 0:
            print("❌ No valid statistics computed")
            return
        
        # Save statistics
        stats_df.to_csv(output_dir / "layer_statistics.csv", index=False)
        
        # Create plots
        plot_layer_comparison(stats_df, output_dir)
        
        # Find optimal layers for each segment
        segment_rankings = {}
        for segment in ['full_avg', 'reasoning_avg', 'answer_avg']:
            ranking = find_optimal_layers(stats_df, segment)
            if len(ranking) > 0:
                segment_rankings[segment] = ranking
                # Save detailed rankings
                ranking_file = output_dir / f"{segment}_rankings.csv"
                ranking.to_csv(ranking_file, index=False)
        
        # Generate recommendations
        recommendations = generate_recommendations(segment_rankings)
        
        # Save recommendations as JSON
        recommendations_file = output_dir / "layer_recommendations.json"
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Print recommendations
        print("\n" + "=" * 60)
        print("🎯 LAYER RECOMMENDATIONS")
        print("=" * 60)
        
        for segment, recs in recommendations.items():
            print(f"\n📊 {segment.replace('_', ' ').title()}:")
            print(f"    🥇 Best single layer: {recs['best_single']}")
            print(f"    🏆 Top 5 layers: {recs['top_5_overall']}")
            print(f"    🎯 Diverse selection: {recs['diverse_selection']}")
        
        print(f"\n💾 Complete analysis saved to: {output_dir}")
        print(f"📈 Statistics: {output_dir / 'layer_statistics.csv'}")
        print(f"🎯 Recommendations: {output_dir / 'layer_recommendations.json'}")
        print(f"📊 Plots saved to: {output_dir}")
        print("\n🚀 Use the recommended layers for your steering vector experiments!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()