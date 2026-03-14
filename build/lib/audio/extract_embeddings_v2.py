import os
import warnings

# FORCE TF TO BE QUIET AND DISABLE XLA BEFORE ANY IMPORTS
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
import tensorflow_hub as hub
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import gc

def load_model_v2():
    """Load Perch v2 from Kaggle Hub."""
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
    return hub.load(model_url)

def worker_init():
    global model_instance
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
    try:
        model_instance = load_model_v2()
    except Exception as e:
        print(f"WORKER INIT ERROR: {e}")

def process_file_worker(filename, raw_dir):
    file_path = os.path.join(raw_dir, "train_audio", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"

    try:
        audio, _ = librosa.load(file_path, sr=32000, duration=5)
        if len(audio) < 160000: audio = np.pad(audio, (0, 160000 - len(audio)))
        
        inputs = tf.constant(audio[np.newaxis, :].astype(np.float32))
        
        # Use signature for robustness
        infer = model_instance.signatures['serving_default']
        input_key = list(infer.structured_input_signature[1].keys())[0]
        outputs = infer(**{input_key: inputs})
        
        # Check both possible keys
        if 'embedding' in outputs:
            return outputs['embedding'].numpy()[0]
        elif 'output_1' in outputs:
            return outputs['output_1'].numpy()[0]
        else:
            return f"ERROR: No embedding found in keys: {list(outputs.keys())}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. DOWNLOAD DATA IF MISSING (CRITICAL FOR VERTEX AI)
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print(f"Data missing in {raw_dir}. Downloading via Kaggle CLI...")
        # Check if zip is there
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if not os.path.exists(zip_path):
            os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        
        if os.path.exists(zip_path):
            print("Unzipping data...")
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)

    # Pre-warm TF Hub cache
    print("Pre-warming Perch v2 cache...")
    hub.resolve("https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1")
    
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit: train_df = train_df.head(args.limit)
    
    filenames = train_df['filename'].tolist()
    results = []

    print(f"Cloud Mode (V2): Using {args.workers} workers...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=worker_init) as pool:
        for i in range(0, len(filenames), args.batch_size):
            batch = filenames[i:i+args.batch_size]
            batch_results = list(tqdm(pool.imap(partial(process_file_worker, raw_dir=raw_dir), batch), 
                                     total=len(batch), desc=f"Batch {i//args.batch_size + 1}"))
            results.extend(batch_results)
            gc.collect()

    # Diagnostics
    errors = [res for res in results if isinstance(res, str) and res.startswith("ERROR")]
    valid_results = [res for res in results if not (isinstance(res, str) and res.startswith("ERROR"))]
    
    if errors:
        print(f"\nFound {len(errors)} errors. First 5 examples:")
        for err in errors[:5]:
            print(f"  - {err}")

    # Save logic
    if valid_results:
        valid_idx = [i for i, r in enumerate(results) if not (isinstance(r, str) and r.startswith("ERROR"))]
        embeddings = np.array(valid_results)
        final_output_path = os.path.join(processed_dir, "perch_v2_embeddings.npz")
        final_csv_path = os.path.join(processed_dir, "train_with_perch_v2.csv")
        
        np.savez_compressed(final_output_path, embeddings=embeddings, indices=valid_idx)
        train_df.iloc[valid_idx].to_csv(final_csv_path, index=False)
        print(f"Saved {len(embeddings)} v2 embeddings.")

        if args.gcs_bucket:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(args.gcs_bucket)
                bucket.blob("processed/perch_v2_embeddings.npz").upload_from_filename(final_output_path)
                bucket.blob("processed/train_with_perch_v2.csv").upload_from_filename(final_csv_path)
                print(f"Uploaded v2 results to gs://{args.gcs_bucket}")
            except Exception as e:
                print(f"GCS Upload Failed: {e}")
    else:
        print("No valid embeddings extracted.")

if __name__ == "__main__":
    main()
