import os
import warnings
import gc
import argparse
import multiprocessing as mp
from functools import partial

# Optimizations for CPU Inference
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=1" # V2 generally performs better with XLA ON
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

def check_compatibility():
    tf_version = tf.__version__
    print(f"DEBUG: TensorFlow Version: {tf_version}")
    major, minor, _ = map(int, tf_version.split('.'))
    if major == 2 and minor < 16:
        print("❌ ERROR: Perch v2 requires TensorFlow 2.16.0 or higher for StableHLO support.")
        print("          Please upgrade: uv pip install 'tensorflow>=2.16.2'")
        return False
    return True

def load_model_v2():
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
    print(f"🔍 Loading Perch v2 (1536-dim) from {model_url}...")
    return hub.load(model_url)

def worker_init():
    global model_instance
    try:
        model_instance = load_model_v2()
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
        input_key = list(infer.structured_input_signature[1].keys())[0]
        
        outputs = infer(**{input_key: tf.constant(inputs)})
        
        if 'embedding' in outputs:
            return outputs['embedding'].numpy()[0]
        else:
            return f"ERROR: 'embedding' key missing. Found: {list(outputs.keys())}"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    if not check_compatibility(): return

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    # Default to fewer workers for local stability
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count()))
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    train_csv = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"❌ Metadata not found at {train_csv}.")
        return

    # Warm up the cache in main process first
    print("Pre-warming model cache...")
    hub.resolve("https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2")
        
    train_df = pd.read_csv(train_csv)
    if args.limit: train_df = train_df.head(args.limit)
    
    filenames = train_df['filename'].tolist()
    print(f"🚀 Starting PERCH V2 extraction for {len(filenames)} files using {args.workers} workers...")
    
    results = []
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=args.workers, initializer=worker_init) as pool:
        for i in range(0, len(filenames), args.batch_size):
            batch = filenames[i:i+args.batch_size]
            batch_results = list(tqdm(pool.imap(partial(process_file_worker, raw_dir=raw_dir), batch), 
                                     total=len(batch), desc=f"Batch {i//args.batch_size + 1}"))
            results.extend(batch_results)
            gc.collect()

    # Save logic...
    valid_results = [res for res in results if not (isinstance(res, str) and res.startswith("ERROR"))]
    if valid_results:
        embeddings = np.array(valid_results)
        print(f"✅ Success! Embedding shape: {embeddings.shape}")
        output_path = os.path.join(processed_dir, "perch_v2_embeddings.npz")
        np.savez_compressed(output_path, embeddings=embeddings)
        print(f"💾 Saved to {output_path}")

if __name__ == "__main__":
    main()
