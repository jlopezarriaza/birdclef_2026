import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random
import gc

def sync_from_gcs(bucket, local_dir, remote_path):
    """Simple wrapper for gsutil cp."""
    if not os.path.exists(local_dir):
        print(f"Syncing {remote_path} from gs://{bucket}...")
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        os.system(f"gsutil -m cp -r gs://{bucket}/{remote_path} {local_dir}")

def process_file_worker(row, raw_dir, noise_bank_dir, output_dir, noise_filenames):
    """
    Worker function to stitch a single file.
    """
    filename = row['filename']
    target_species = row['primary_label']
    start_time = row['start_time']
    end_time = row['end_time']
    
    input_path = os.path.join(raw_dir, "train_audio", filename)
    output_filename = filename.replace("/", "_").replace(".ogg", ".wav") # Flat structure for unified set
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        return {"filename": filename, "unified_filename": output_filename, "status": "exists"}

    try:
        # 1. Load Original Audio
        audio, sr = sf.read(input_path)
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            sr = 32000
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        duration = len(audio) / sr
        
        # 2. Stitching Logic
        if duration < 5.0:
            # --- CURATED CLIP STITCHING ---
            # Pick a random noise clip
            noise_file = random.choice(noise_filenames)
            noise_path = os.path.join(noise_bank_dir, noise_file)
            noise, _ = sf.read(noise_path) # Assumed 5s at 32kHz
            
            # Center the short audio on the noise
            # (Vocalization Centering)
            start_idx = int((5.0 - duration) / 2.0 * sr)
            end_idx = start_idx + len(audio)
            
            # Blend: We can just overwrite, but a tiny crossfade or just simple placement works for "Easy" path
            noise[start_idx:end_idx] = audio
            final_audio = noise
            method = "stitched"
        else:
            # --- LONG CLIP CROPPING ---
            start_idx = int(start_time * sr)
            end_idx = start_idx + int(5.0 * sr)
            
            # Safety crop (in case end_idx is slightly beyond len(audio))
            if end_idx > len(audio):
                final_audio = audio[len(audio) - int(5.0 * sr) : len(audio)]
            else:
                final_audio = audio[start_idx:end_idx]
            method = "cropped"

        # 3. Save
        sf.write(output_path, final_audio, 32000)
        
        return {
            "filename": filename,
            "primary_label": target_species,
            "unified_filename": output_filename,
            "method": method,
            "status": "success"
        }
    except Exception as e:
        return {"filename": filename, "error": str(e), "status": "error"}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    # Paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    peak_csv = os.path.join(processed_dir, "train_v2_peaks.csv")
    noise_bank_dir = os.path.join(processed_dir, "noise_bank")
    noise_registry = os.path.join(processed_dir, "noise_bank_registry.csv")
    output_dir = os.path.join(processed_dir, "train_v2_unified")
    master_csv = os.path.join(processed_dir, "train_v2_master.csv")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Cloud Sync (Vertex AI Ready)
    bucket = args.gcs_bucket or os.getenv("GCS_BUCKET")
    if bucket:
        print(f"Cloud mode enabled. Using bucket: {bucket}")
        # Download peak manifest and noise registry if missing
        if not os.path.exists(peak_csv):
            os.system(f"gsutil cp gs://{bucket}/processed/train_v2_peaks.csv {peak_csv}")
        if not os.path.exists(noise_registry):
            os.system(f"gsutil cp gs://{bucket}/processed/noise_bank_registry.csv {noise_registry}")
        
        # Download data (this might take time, but necessary for Vertex)
        sync_from_gcs(bucket, os.path.join(raw_dir, "train_audio"), "raw/train_audio")
        sync_from_gcs(bucket, noise_bank_dir, "processed/noise_bank")

    # 2. Load Metadata
    print("Loading manifests...")
    peaks_df = pd.read_csv(peak_csv)
    noise_df = pd.read_csv(noise_registry)
    noise_filenames = noise_df['noise_filename'].tolist()
    
    if args.limit:
        peaks_df = peaks_df.head(args.limit)
        
    rows = peaks_df.to_dict('records')
    print(f"Stitching {len(rows)} files with {args.workers} workers...")

    # 3. Parallel Processing
    with mp.Pool(processes=args.workers) as pool:
        process_func = partial(process_file_worker, 
                               raw_dir=raw_dir, 
                               noise_bank_dir=noise_bank_dir, 
                               output_dir=output_dir, 
                               noise_filenames=noise_filenames)
        
        results = []
        for res in tqdm(pool.imap_unordered(process_func, rows), total=len(rows)):
            results.append(res)
            
            # Intermediate Cloud Upload to save disk space
            if bucket and len(results) % 2000 == 0:
                print(f"Uploading batch of 2000 files to gs://{bucket}/processed/train_v2_unified/...")
                os.system(f"gsutil -m cp {output_dir}/*.wav gs://{bucket}/processed/train_v2_unified/")
                # Optional: clean local disk if needed
                # os.system(f"rm {output_dir}/*.wav")

    # 4. Final Manifest & Cleanup
    success_results = [r for r in results if r['status'] == 'success' or r['status'] == 'exists']
    final_df = pd.DataFrame(success_results)
    final_df.to_csv(master_csv, index=False)
    print(f"Master manifest saved to {master_csv}")

    if bucket:
        print("Final cloud sync...")
        os.system(f"gsutil -m cp {output_dir}/*.wav gs://{bucket}/processed/train_v2_unified/")
        os.system(f"gsutil cp {master_csv} gs://{bucket}/processed/train_v2_master.csv")
        print("Done!")

if __name__ == "__main__":
    main()
