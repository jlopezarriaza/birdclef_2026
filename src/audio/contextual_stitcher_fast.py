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
import subprocess
import json

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return False
    return True

def sync_from_gcs(bucket, local_dir, remote_path):
    print(f"Syncing {remote_path} from gs://{bucket}...")
    os.makedirs(local_dir, exist_ok=True)
    return run_command(f"gsutil -m rsync -r gs://{bucket}/{remote_path} {local_dir}")

def setup_kaggle_creds():
    """Setup kaggle.json if environment variables are present."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")
    if username and key:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        print("Kaggle credentials configured.")
        return True
    return False

def download_from_kaggle(raw_dir):
    """Download competition data if missing."""
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print("Data missing. Downloading from Kaggle...")
        setup_kaggle_creds()
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if run_command(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}"):
            if os.path.exists(zip_path):
                run_command(f"unzip -qo {zip_path} -d {raw_dir}")
                os.remove(zip_path)
                return True
    return False

def process_file_worker(row, raw_dir, noise_bank_dir, output_dir, noise_filenames):
    filename = row['filename']
    target_species = row['primary_label']
    start_time = row['start_time']
    
    input_path = os.path.join(raw_dir, "train_audio", filename)
    output_filename = filename.replace("/", "_").replace(".ogg", ".wav")
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        return {"filename": filename, "unified_filename": output_filename, "status": "exists"}

    try:
        audio, sr = sf.read(input_path)
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            sr = 32000
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        duration = len(audio) / sr
        
        if duration < 5.0:
            noise_file = random.choice(noise_filenames)
            noise_path = os.path.join(noise_bank_dir, noise_file)
            noise, _ = sf.read(noise_path)
            start_idx = int((5.0 - duration) / 2.0 * sr)
            end_idx = start_idx + len(audio)
            noise[start_idx:end_idx] = audio
            final_audio = noise
            method = "stitched"
        else:
            start_idx = int(start_time * sr)
            end_idx = start_idx + int(5.0 * sr)
            if end_idx > len(audio):
                final_audio = audio[len(audio) - int(5.0 * sr) : len(audio)]
            else:
                final_audio = audio[start_idx:end_idx]
            method = "cropped"

        sf.write(output_path, final_audio, 32000)
        return {"filename": filename, "primary_label": target_species, "unified_filename": output_filename, "method": method, "status": "success"}
    except Exception as e:
        return {"filename": filename, "error": str(e), "status": "error"}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    peak_csv = os.path.join(processed_dir, "train_v2_peaks_fast.csv")
    noise_bank_dir = os.path.join(processed_dir, "noise_bank")
    noise_registry = os.path.join(processed_dir, "noise_bank_registry_light.csv")
    output_dir = os.path.join(processed_dir, "train_v2_unified_fast")
    master_csv = os.path.join(processed_dir, "train_v2_master_fast.csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "train_audio"), exist_ok=True)

    bucket = args.gcs_bucket or os.getenv("GCS_BUCKET")
    if bucket:
        print(f"Cloud mode enabled. Using bucket: {bucket}")
        if not os.path.exists(peak_csv): run_command(f"gsutil cp gs://{bucket}/processed/train_v2_peaks_fast.csv {peak_csv}")
        if not os.path.exists(noise_registry): run_command(f"gsutil cp gs://{bucket}/processed/noise_bank_registry_light.csv {noise_registry}")
        
        # Download Audio
        if not sync_from_gcs(bucket, os.path.join(raw_dir, "train_audio"), "raw/train_audio"):
            print("GCS Audio Sync failed. Falling back to Kaggle...")
            download_from_kaggle(raw_dir)
        
        sync_from_gcs(bucket, noise_bank_dir, "processed/noise_bank")
    
    # Ensure train.csv exists
    download_from_kaggle(raw_dir)

    if not os.path.exists(peak_csv):
        print(f"CRITICAL ERROR: {peak_csv} missing.")
        return

    peaks_df = pd.read_csv(peak_csv)
    noise_filenames = pd.read_csv(noise_registry)['noise_filename'].tolist()
    if args.limit: peaks_df = peaks_df.head(args.limit)
    rows = peaks_df.to_dict('records')

    print(f"FAST Stitching {len(rows)} files with {args.workers} workers...")
    with mp.Pool(processes=args.workers) as pool:
        process_func = partial(process_file_worker, raw_dir=raw_dir, noise_bank_dir=noise_bank_dir, output_dir=output_dir, noise_filenames=noise_filenames)
        results = []
        for res in tqdm(pool.imap_unordered(process_func, rows), total=len(rows)):
            results.append(res)
            if bucket and len(results) % 2000 == 0:
                run_command(f"gsutil -m cp {output_dir}/*.wav gs://{bucket}/processed/train_v2_unified_fast/")

    final_df = pd.DataFrame([r for r in results if r['status'] in ['success', 'exists']])
    final_df.to_csv(master_csv, index=False)
    if bucket:
        run_command(f"gsutil -m cp {output_dir}/*.wav gs://{bucket}/processed/train_v2_unified_fast/")
        run_command(f"gsutil cp {master_csv} gs://{bucket}/processed/train_v2_master_fast.csv")

if __name__ == "__main__":
    main()
