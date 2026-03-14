import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

def main():
    # Define paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    soundscapes_csv_path = raw_dir / "train_soundscapes_labels.csv"
    species_registry_path = processed_dir / "species_registry.json"
    output_csv_path = processed_dir / "soundscape_kfold_registry.csv"

    print("Loading species registry...")
    with open(species_registry_path, 'r') as f:
        species_registry = json.load(f)
    
    # Sort species by their index in the registry to keep column order consistent
    species_list = sorted(species_registry.keys(), key=lambda x: species_registry[x])
    num_species = len(species_list)
    print(f"Total species: {num_species}")

    print(f"Reading {soundscapes_csv_path}...")
    df = pd.read_csv(soundscapes_csv_path)

    # Helper function to convert HH:MM:SS to seconds
    def time_to_seconds(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s

    df['start_sec'] = df['start'].apply(time_to_seconds)
    
    # Get unique soundscapes
    unique_filenames = df['filename'].unique()
    print(f"Found {len(unique_filenames)} unique soundscapes.")

    # We will create 56 windows (0 to 55) for each soundscape
    rows = []
    
    print("Processing soundscape windows...")
    for filename in tqdm(unique_filenames):
        # Filter rows for this soundscape
        file_df = df[df['filename'] == filename]
        
        # Create a mapping from interval_start to set of species
        # We use a set and update to handle potential duplicates in the CSV
        interval_labels = {}
        for _, row in file_df.iterrows():
            labels_str = str(row['primary_label'])
            if pd.isna(row['primary_label']) or labels_str.lower() == 'nan':
                labels = []
            else:
                labels = [l.strip() for l in labels_str.split(';') if l.strip()]
            
            start_t = int(row['start_sec'])
            if start_t not in interval_labels:
                interval_labels[start_t] = set()
            interval_labels[start_t].update(labels)
            
        for window_idx in range(56):
            # Window starts at window_idx and ends at window_idx + 5
            window_start = window_idx
            window_end = window_idx + 5
            
            # Find all intervals [5k, 5k+5] that overlap with [window_start, window_end]
            # Interval k overlaps if max(window_start, 5k) < min(window_end, 5k+5)
            # Since intervals are 0-5, 5-10, ..., 55-60, we only check these.
            window_species = set()
            for interval_start in range(0, 60, 5):
                interval_end = interval_start + 5
                # Overlap check
                if max(window_start, interval_start) < min(window_end, interval_end):
                    if interval_start in interval_labels:
                        window_species.update(interval_labels[interval_start])
            
            # Create base row data
            row_data = [filename, window_idx]
            
            # Multi-label encoding (binary)
            for species in species_list:
                row_data.append(1 if species in window_species else 0)
            
            rows.append(row_data)

    # Define columns
    columns = ['filename', 'window_idx'] + species_list
    
    print("Creating DataFrame...")
    final_df = pd.DataFrame(rows, columns=columns)

    print("Applying GroupKFold (5-fold) based on filename...")
    gkf = GroupKFold(n_splits=5)
    
    # Groups are the filenames
    groups = final_df['filename']
    
    # Initialize fold column
    final_df['fold'] = -1
    
    # We use a dummy target for the split as it only depends on groups
    for fold, (train_idx, val_idx) in enumerate(gkf.split(final_df, groups=groups)):
        final_df.loc[val_idx, 'fold'] = fold

    # Reorder columns: filename, window_idx, fold, species...
    ordered_cols = ['filename', 'window_idx', 'fold'] + species_list
    final_df = final_df[ordered_cols]

    # Save to CSV
    print(f"Saving registry to {output_csv_path}...")
    final_df.to_csv(output_csv_path, index=False)
    
    print("Done!")
    print(f"Registry shape: {final_df.shape}")
    print(f"Folds distribution:\n{final_df.groupby('fold')['filename'].nunique()}")

if __name__ == "__main__":
    main()
