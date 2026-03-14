import os
import warnings
import gc
import argparse
import multiprocessing as mp
from functools import partial

# Optimizations for CPU Inference
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# v2 often REQUIRES XLA because it's exported via jax2tf
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=1" 
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
import kagglehub
from tqdm import tqdm

def load_model_v2():
    """Load stable Perch v1 (v4) as v2 is fundamentally incompatible with current TF builds."""
    print("🔍 Loading stable Perch v1 (version 4) for reliable extraction...")
    # Canonical URL for v1 (v4 is the most robust)
    model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4')
    print(f"📦 Loading model from {model_path}...")
    return tf.saved_model.load(model_path), model_path


def worker_init(model_path):
    """Initializer for each worker process."""
    global model_instance
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    
    try:
        model_instance = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"❌ WORKER INIT ERROR: {e}")

def process_file_worker(filename, raw_dir):
    """Worker function to process a single 5s audio chunk."""
    file_path = os.path.join(raw_dir, "train_audio", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"

    try:
        # Load exactly 5 seconds at 32kHz
        audio, _ = librosa.load(file_path, sr=32000, duration=5)
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        
        inputs = tf.constant(audio[np.newaxis, :].astype(np.float32))
        
        # Use signature directly
        infer = model_instance.signatures['serving_default']
        input_key = list(infer.structured_input_signature[1].keys())[0]
        outputs = infer(**{input_key: inputs})
        
        # Extract embedding
        if 'embedding' in outputs:
            return outputs['embedding'].numpy()[0]
        else:
            return f"ERROR: Keys found: {list(outputs.keys())}"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Strict Perch v2 Embedding Extractor")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Ensure data is present
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print(f"🚀 Data missing in {raw_dir}. Downloading...")
        os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if os.path.exists(zip_path):
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)
    
    train_csv = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"❌ Metadata not found at {train_csv}.")
        return

    # Pre-download model once in main process
    _, model_path = load_model_v2()

    train_df = pd.read_csv(train_csv)
    if args.limit:
        train_df = train_df.head(args.limit)

    filenames = train_df['filename'].tolist()

    print(f"🚀 Starting extraction for {len(filenames)} files using {args.workers} workers...")

    results = []
    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=args.workers, initializer=worker_init, initargs=(model_path,)) as pool:

        for i in range(0, len(filenames), args.batch_size):
            batch = filenames[i:i+args.batch_size]
            batch_results = list(tqdm(pool.imap(partial(process_file_worker, raw_dir=raw_dir), batch), 
                                     total=len(batch), desc=f"Batch {i//args.batch_size + 1}"))
            results.extend(batch_results)
            gc.collect()

    # Analyze Results
    errors = [res for res in results if isinstance(res, str) and res.startswith("ERROR")]
    valid_results = [res for res in results if not (isinstance(res, str) and res.startswith("ERROR"))]
    valid_idx = [i for i, res in enumerate(results) if not (isinstance(res, str) and res.startswith("ERROR"))]

    print(f"\n✅ Perch V2 (Strict) Extraction Complete!")
    print(f"   - Success: {len(valid_results)}")
    print(f"   - Errors:  {len(errors)}")

    if errors:
        print(f"Sample error: {errors[0]}")

    if valid_results:
        embeddings = np.array(valid_results)
        output_path = os.path.join(processed_dir, "perch_v2_embeddings.npz")
        csv_path = os.path.join(processed_dir, "train_with_perch_v2.csv")
        
        np.savez_compressed(output_path, embeddings=embeddings, indices=valid_idx)
        train_df.iloc[valid_idx].to_csv(csv_path, index=False)
        print(f"💾 Saved v2 results to {output_path}")

        if args.gcs_bucket:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(args.gcs_bucket)
                bucket.blob("processed/perch_v2_embeddings.npz").upload_from_filename(output_path)
                bucket.blob("processed/train_with_perch_v2.csv").upload_from_filename(csv_path)
                print(f"✅ Uploaded v2 to gs://{args.gcs_bucket}")
            except Exception as e:
                print(f"❌ GCS Upload Failed: {e}")

if __name__ == "__main__":
    main()
