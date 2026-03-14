import os
import warnings
import gc
import argparse
import multiprocessing as mp
from functools import partial

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

def load_model_v2_cpu():
    """Load the CPU-specific variant of Perch v2."""
    model_slug = 'google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1'
    print(f"🔍 Loading Perch v2 CPU (1536-dim) from {model_slug}...")
    model_path = kagglehub.model_download(model_slug)
    return tf.saved_model.load(model_path)

def worker_init():
    """Initializer for each worker process."""
    global model_instance
    try:
        # Each worker loads its own instance
        model_instance = load_model_v2_cpu()
    except Exception as e:
        print(f"❌ WORKER INIT ERROR: {e}")

def process_file_worker(filename, raw_dir):
    """Worker function to process a single audio file."""
    file_path = os.path.join(raw_dir, "train_audio", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"

    try:
        # Load exactly 5 seconds at 32kHz
        audio, _ = librosa.load(file_path, sr=32000, duration=5)
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        
        # Inference
        inputs = audio[np.newaxis, :].astype(np.float32)
        infer = model_instance.signatures['serving_default']
        outputs = infer(inputs=tf.constant(inputs))
        
        # Extract embedding
        if 'embedding' in outputs:
            return outputs['embedding'].numpy()[0]
        else:
            return f"ERROR: 'embedding' missing."
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Full Perch v2 Embedding Extraction")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24) # Reduced from 32 to avoid OOM
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    # Check for Kaggle Credentials early
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("❌ CRITICAL: KAGGLE_USERNAME or KAGGLE_KEY not set in environment.")
        return

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Download Data if missing
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print("🚀 Data missing. Downloading via Kaggle CLI...")
        os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if os.path.exists(zip_path):
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)
    
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit: train_df = train_df.head(args.limit)
    
    # 2. Pre-warm model cache in main process (Faster worker init)
    print("Pre-warming model cache in main process...")
    load_model_v2_cpu()
    
    filenames = train_df['filename'].tolist()
    print(f"🚀 Starting Extraction for {len(filenames)} files using {args.workers} workers...")
    
    results = []
    ctx = mp.get_context('spawn')
    
    # Using 'spawn' context is more robust for TF
    with ctx.Pool(processes=args.workers, initializer=worker_init) as pool:
        for i in range(0, len(filenames), args.batch_size):
            batch = filenames[i:i+args.batch_size]
            batch_results = list(tqdm(pool.imap(partial(process_file_worker, raw_dir=raw_dir), batch), 
                                     total=len(batch), desc=f"Batch {i//args.batch_size + 1}"))
            results.extend(batch_results)
            gc.collect()

    # 3. Process and Save
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
            # Using -m for multi-threaded upload
            os.system(f"gsutil -m cp {output_path} gs://{args.gcs_bucket}/processed/perch_v2_embeddings.npz")
            os.system(f"gsutil -m cp {csv_path} gs://{args.gcs_bucket}/processed/train_with_perch_v2.csv")
    else:
        print("❌ No valid embeddings extracted.")

if __name__ == "__main__":
    main()
