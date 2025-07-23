#!/usr/bin/env python3
"""
Create a sampled dataset from airgapagent-r-small.json with better distribution.
This ensures we get diverse profiles instead of consecutive ones.
"""

import json
import random
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Create a sampled dataset with good distribution')
    parser.add_argument('--input_file', type=str, default='datasets/airgapagent-r-small.json', 
                       help='Input dataset file')
    parser.add_argument('--output_file', type=str, default='datasets/airgapagent-r-sample-100.json',
                       help='Output sample dataset file')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of samples to extract')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)}")
    
    # Group by profile name to see distribution
    profile_groups = defaultdict(list)
    for i, item in enumerate(data):
        profile_name = item.get('profile', {}).get('name', 'Unknown')
        profile_groups[profile_name].append(i)
    
    print(f"Found {len(profile_groups)} unique profiles")
    print("Profile distribution:")
    for name, indices in profile_groups.items():
        print(f"  {name}: {len(indices)} examples")
    
    # Strategy 1: Sample proportionally from each profile
    if args.sample_size >= len(profile_groups):
        # If we want more samples than profiles, sample multiple from each
        samples_per_profile = args.sample_size // len(profile_groups)
        extra_samples = args.sample_size % len(profile_groups)
        
        sampled_indices = []
        profile_names = list(profile_groups.keys())
        
        # Take samples_per_profile from each profile
        for profile_name in profile_names:
            indices = profile_groups[profile_name]
            sample_count = min(samples_per_profile, len(indices))
            sampled = random.sample(indices, sample_count)
            sampled_indices.extend(sampled)
        
        # Distribute extra samples randomly
        for _ in range(extra_samples):
            profile_name = random.choice(profile_names)
            remaining_indices = [idx for idx in profile_groups[profile_name] 
                               if idx not in sampled_indices]
            if remaining_indices:
                sampled_indices.append(random.choice(remaining_indices))
    
    else:
        # If we want fewer samples than profiles, sample one from each profile randomly
        sampled_indices = []
        selected_profiles = random.sample(list(profile_groups.keys()), args.sample_size)
        for profile_name in selected_profiles:
            indices = profile_groups[profile_name]
            sampled_indices.append(random.choice(indices))
    
    # Shuffle the final order
    random.shuffle(sampled_indices)
    
    # Create sampled dataset
    sampled_data = [data[i] for i in sampled_indices]
    
    print(f"\nSampled {len(sampled_data)} examples")
    
    # Show distribution in sample
    sample_profile_counts = defaultdict(int)
    for item in sampled_data:
        profile_name = item.get('profile', {}).get('name', 'Unknown')
        sample_profile_counts[profile_name] += 1
    
    print("Sample distribution:")
    for name, count in sample_profile_counts.items():
        print(f"  {name}: {count} examples")
    
    # Save sampled dataset
    print(f"\nSaving sampled dataset to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    
    print("Done! You can now run:")
    print(f"!python eval_cp.py \\")
    print(f"--model qwen2.5-1.5b \\")
    print(f"--input_file {args.output_file} \\")
    print(f"--output_file results/sample_100_cot.json \\")
    print(f"--gpt_eval \\")
    print(f"--prompt_type cot_explicit_unk \\")
    print(f"--max_tokens 300 \\")
    print(f"--temperature 0.1")

if __name__ == "__main__":
    main() 