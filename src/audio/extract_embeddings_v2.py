import os
import json
import warnings
import gc
import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

# Optimizations for CPU Inference
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1" 
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_text as tf_text
import kagglehub
from tqdm import tqdm

def setup_kaggle_auth():
    """Ensure kaggle.json exists for the CLI tool."""
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")
    
    if user and key:
        print(f"🛠️ Setting up Kaggle credentials for user: {user}")
        kaggle_path = Path.home() / ".kaggle"
        kaggle_path.mkdir(exist_ok=True)
        config_file = kaggle_path / "kaggle.json"
        
        with open(config_file, "w") as f:
            json.dump({"username": user, "key": key}, f)
        
        os.chmod(config_file, 0o600)
        # Also set env vars just in case
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key
        return True
    
    print("⚠️ No Kaggle credentials found in environment.")
    return False

def load_model_v2_cpu():
    model_slug = 'google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1'
    print(f"🔍 Loading Perch v2 CPU (1536-dim) from {model_slug}...")
    model_path = kagglehub.model_download(model_slug)
    return tf.saved_model.load(model_path)

def worker_init():
    global model_instance
    try:
        model_instance = load_model_v2_cpu()
    except Exception as e:
        print(f"❌ WORKER INIT ERROR: {e}")

def process_file_worker(filename, raw_dir):
    file_path = os.path.join(raw_dir, "train_audio", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"

    try:
        audio, _ = librosa.load(file_path, sr=32000, duration=5)
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        
        inputs = audio[np.newaxis, :].astype(np.float32)
        infer = model_instance.signatures['serving_default']
        outputs = infer(inputs=tf.constant(inputs))
        
        if 'embedding' in outputs:
            return outputs['embedding'].numpy()[0]
        else:
            return f"ERROR: 'embedding' missing."
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    setup_kaggle_auth()

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Download Data if missing
    train_csv = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_csv):
        print("🚀 Data missing. Downloading via Kaggle CLI...")
        # Check if we can run kaggle
        ret = os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        if ret != 0:
            print("❌ FAILED to download data from Kaggle. Check credentials and competition rules.")
            return

        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if os.path.exists(zip_path):
            print("📦 Unzipping data...")
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)
    
    if not os.path.exists(train_csv):
        print(f"❌ CRITICAL: {train_csv} not found after download attempt.")
        return
    
    train_df = pd.read_csv(train_csv)
    if args.limit: train_df = train_df.head(args.limit)
    
    print("Pre-warming model cache in main process...")
    load_model_v2_cpu()
    
    filenames = train_df['filename'].tolist()
    print(f"🚀 Starting Extraction for {len(filenames)} files using {args.workers} workers...")
    
    results = []
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=args.workers, initializer=worker_init) as pool:
        for i in range(0, len(filenames), args.batch_size):
            batch = filenames[i:i+args.batch_size]
            batch_results = list(tqdm(pool.imap(partial(process_file_worker, raw_dir=raw_dir), batch), 
                                     total=len(batch), desc=f"Batch {i//args.batch_size + 1}"))
            results.extend(batch_results)
            gc.collect()

    valid_idx = [i for i, r in enumerate(results) if not (isinstance(r, str) and r.startswith("ERROR"))]
    valid_results = [results[i] for i in valid_idx]
    
    if valid_results:
        embeddings = np.array(valid_results)
        output_path = os.path.join(processed_dir, "perch_v2_embeddings.npz")
        csv_path = os.path.join(processed_dir, "train_with_perch_v2.csv")
        
        np.savez_compressed(output_path, embeddings=embeddings, indices=valid_idx)
        train_df.iloc[valid_idx].to_csv(csv_path, index=False)
        print(f"✅ Saved {len(embeddings)} embeddings.")

        if args.gcs_bucket:
            print(f"📤 Uploading to gs://{args.gcs_bucket}...")
            os.system(f"gsutil -m cp {output_path} gs://{args.gcs_bucket}/processed/perch_v2_embeddings.npz")
            os.system(f"gsutil -m cp {csv_path} gs://{args.gcs_bucket}/processed/train_with_perch_v2.csv")
    else:
        print("❌ No valid embeddings extracted.")

if __name__ == "__main__":
    main()
