import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from src.audio.spectrograms import SpectrogramGenerator, save_spectrogram

def process_file(row, raw_dir, output_dir, img_size):
    filename = row['filename']
    audio_path = os.path.join(raw_dir, "train_audio", filename)
    
    # Create subdirectories matching the species labels
    rel_path = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(output_dir, rel_path)
    
    if os.path.exists(output_path):
        return True
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    gen = SpectrogramGenerator(img_size=img_size)
    img = gen.generate(audio_path)
    
    if img is not None:
        save_spectrogram(img, output_path)
        return True
    return False

def main(limit=None, workers=8):
    raw_dir = "data/raw"
    output_dir = "data/processed/spectrograms"
    img_size = 224
    
    train_csv = os.path.join(raw_dir, "train.csv")
    df = pd.read_csv(train_csv)
    
    if limit:
        df = df.head(limit)
        
    print(f"Precalculating {len(df)} spectrograms using {workers} workers...")
    
    # Use a pool to process files in parallel
    process_func = partial(process_file, raw_dir=raw_dir, output_dir=output_dir, img_size=img_size)
    
    rows = [row for _, row in df.iterrows()]
    
    with mp.Pool(workers) as pool:
        results = list(tqdm(pool.imap(process_func, rows), total=len(rows)))
        
    success_count = sum(results)
    print(f"Finished. Successfully created {success_count} spectrograms in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    main(limit=args.limit, workers=args.workers)
