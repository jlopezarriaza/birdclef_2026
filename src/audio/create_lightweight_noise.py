import pandas as pd
import os

def main():
    registry_path = "data/processed/noise_bank_registry.csv"
    output_path = "data/processed/noise_bank_registry_light.csv"
    
    if not os.path.exists(registry_path):
        print("Original noise registry not found.")
        return

    df = pd.read_csv(registry_path)
    print(f"Original noise bank: {len(df)} clips.")

    # Strategy: Pick the first 2 noise clips from each unique source soundscape
    # This ensures geographic and temporal diversity across the Pantanal
    df_light = df.groupby('source_filename').head(2).reset_index(drop=True)
    
    # If it's still too big, limit to 10,000 total
    if len(df_light) > 10000:
        df_light = df_light.sample(n=10000, random_state=42)

    df_light.to_csv(output_path, index=False)
    
    # Calculate estimated size
    # 32kHz, 16-bit mono, 5s is approx 320KB per file
    est_size_mb = (len(df_light) * 320) / 1024
    
    print(f"Lightweight noise bank: {len(df_light)} clips.")
    print(f"Estimated size: {est_size_mb:.2f} MB (vs ~40,000 MB original)")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
