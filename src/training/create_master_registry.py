import pandas as pd
import json
from pathlib import Path

def main():
    # Define paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    train_csv_path = raw_dir / "train.csv"
    soundscapes_csv_path = raw_dir / "train_soundscapes_labels.csv"
    output_json_path = processed_dir / "species_registry.json"

    print(f"Reading {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    print(f"Reading {soundscapes_csv_path}...")
    soundscapes_df = pd.read_csv(soundscapes_csv_path)

    # Extract unique labels from train.csv
    train_labels = set(train_df['primary_label'].unique())
    print(f"Found {len(train_labels)} unique species in train.csv.")

    # Extract unique labels from train_soundscapes_labels.csv
    # The primary_label column can contain semicolon-separated values
    soundscape_labels = set()
    for labels_str in soundscapes_df['primary_label'].dropna():
        # Split by semicolon and strip whitespace just in case
        labels = [label.strip() for label in labels_str.split(';')]
        soundscape_labels.update(labels)
    
    # Exclude 'nocall' if present, as it is usually not a species ID, 
    # but let's see if the competition includes it in the 234. Wait, 
    # the prompt says "extract all unique species IDs... verify combined unique count is exactly 234 species".
    # I should check what is in the labels.
    print(f"Found {len(soundscape_labels)} unique labels in soundscapes.")

    # Combine
    all_labels = train_labels.union(soundscape_labels)
    
    # If 'nocall' is present and the count is 235, we might need to remove it.
    # Let's not make assumptions until we see the count.
    
    print(f"Total unique combined species: {len(all_labels)}")
    
    # Verify exactly 234
    assert len(all_labels) == 234, f"Expected exactly 234 species, but found {len(all_labels)}"
    print("Verification passed: exactly 234 species found.")

    # Sort alphabetically
    sorted_labels = sorted(list(all_labels))

    # Assign integer indices (0 to 233)
    species_registry = {species: idx for idx, species in enumerate(sorted_labels)}

    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    print(f"Saving registry to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(species_registry, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
