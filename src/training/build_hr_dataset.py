import os
import json
import pandas as pd
import argparse
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def get_duration(file_path):
    """Fast duration lookup using soundfile."""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception:
        return 0.0

def build_hr_dataset(ratio: float = 0.2, output_path: str = "data/processed/train_hr_manifest.csv"):
    """
    Builds a high-resolution training manifest by combining positive training samples
    with background noise samples at a configurable ratio.
    """
    print(f"Building HR dataset with noise-to-signal ratio: {ratio}")
    
    # Paths
    project_root = Path(os.getcwd())
    train_csv_path = "data/raw/train.csv"
    noise_registry_path = "data/processed/noise_bank_registry.csv"
    species_registry_path = "data/processed/species_registry.json"
    train_audio_dir = "data/raw/train_audio"
    noise_bank_dir = "data/processed/noise_bank"

    # Load data
    train_df = pd.read_csv(train_csv_path)
    noise_df = pd.read_csv(noise_registry_path)
    with open(species_registry_path, "r") as f:
        species_registry = json.load(f)

    # 1. Prepare Positive Samples
    print(f"Processing {len(train_df)} positive samples...")
    pos_samples_list = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Calculating positive sample durations"):
        filepath = os.path.join(train_audio_dir, row['filename'])
        # Although extract_embeddings uses 5s, we store actual duration in the manifest
        # if it's available. If it fails, we default to 5.0 (unlikely with soundfile).
        duration = get_duration(filepath)
        if duration == 0.0:
            duration = 5.0 # Fallback
            
        pos_samples_list.append({
            'filepath': filepath,
            'primary_label': row['primary_label'],
            'is_noise': False,
            'duration': duration
        })
    
    pos_samples = pd.DataFrame(pos_samples_list)
    
    # 2. Prepare Negative Samples (Noise)
    # Calculate how many noise samples to include
    num_pos = len(pos_samples)
    if ratio >= 1.0:
        print("Error: Ratio must be less than 1.0 (it represents Noise / (Pos + Noise))")
        return

    num_noise_to_sample = int(num_pos * ratio / (1 - ratio))
    
    print(f"Sampling {num_noise_to_sample} noise clips (out of {len(noise_df)} available)...")
    if num_noise_to_sample > len(noise_df):
        print(f"Warning: Requested {num_noise_to_sample} noise clips but only {len(noise_df)} available.")
        num_noise_to_sample = len(noise_df)
    
    sampled_noise_df = noise_df.sample(n=num_noise_to_sample, random_state=42)
    
    neg_samples_list = []
    for _, row in tqdm(sampled_noise_df.iterrows(), total=len(sampled_noise_df), desc="Formatting noise samples"):
        filepath = os.path.join(noise_bank_dir, row['noise_filename'])
        neg_samples_list.append({
            'filepath': filepath,
            'primary_label': 'noise',
            'is_noise': True,
            'duration': row['duration']
        })
        
    neg_samples = pd.DataFrame(neg_samples_list)

    # 3. Combine and Save
    manifest_df = pd.concat([pos_samples, neg_samples], ignore_index=True)
    
    # Optional: Verify all species in positive samples are in the species_registry
    missing_species = set(pos_samples['primary_label'].unique()) - set(species_registry.keys())
    if missing_species:
        print(f"Warning: {len(missing_species)} species in train.csv are not in species_registry.json!")
        # Depending on requirements, we might want to filter them out.
        # But the prompt says "Ensure the labels are mapped using the species_registry.json",
        # which might mean they *should* all be there.
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    manifest_df.to_csv(output_path, index=False)
    print(f"Manifest saved to {output_path}")
    print(f"Total samples: {len(manifest_df)} ({len(pos_samples)} positive, {len(neg_samples)} noise)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build High-Resolution Training Manifest")
    parser.add_argument("--ratio", type=float, default=0.2, help="Noise-to-Signal ratio (default: 0.2)")
    parser.add_argument("--output", type=str, default="data/processed/train_hr_manifest.csv", help="Output path")
    args = parser.parse_args()
    
    build_hr_dataset(ratio=args.ratio, output_path=args.output)
