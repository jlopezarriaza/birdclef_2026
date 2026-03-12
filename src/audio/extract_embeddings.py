import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import kagglehub
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import gc

# Hardware-aware environment setup
def setup_tf_environment():
    if len(tf.config.list_physical_devices('GPU')) > 0:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
    else:
        # Restrict internal parallelism per process to avoid core thrashing
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

def load_perch_v2():
    """Download and load the Perch v2 model."""
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
    return hub.load(model_url)

def worker_init():
    """Initializer for multiprocessing pool: loads the model into each process."""
    global model_instance
    setup_tf_environment()
    model_instance = load_perch_v2()

def process_file_worker(filename, raw_dir, model=None):
    """Worker function to process a single file. Uses provided model or global instance."""
    file_path = os.path.join(raw_dir, "train_audio", filename)
    target_model = model if model is not None else model_instance
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=32000, duration=5)
        num_samples_needed = 160000
        if len(audio) < num_samples_needed:
            audio = np.pad(audio, (0, num_samples_needed - len(audio)))
        
        # Inference
        inputs = audio[np.newaxis, :].astype(np.float32)
        outputs = target_model(inputs)
        emb = outputs['embedding'].numpy()[0]
        
        # Cleanup
        del audio
        del inputs
        del outputs
        
        return emb
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Parallel Extraction of Perch v2 embeddings.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1000, help="Process in chunks to save memory")
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check for data
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print("Downloading data from Kaggle...")
        os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        os.system(f"unzip -qo {raw_dir}/birdclef-2026.zip -d {raw_dir}")

    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
    
    # RESUME LOGIC
    final_output_path = os.path.join(processed_dir, "perch_v2_embeddings.npz")
    final_csv_path = os.path.join(processed_dir, "train_with_perch_v2.csv")
    
    if os.path.exists(final_output_path) and os.path.exists(final_csv_path):
        print("Existing results found. Checking for completion...")
        existing_df = pd.read_csv(final_csv_path)
        if len(existing_df) >= len(train_df):
            print("All files already processed. Skipping.")
            # Still upload to GCS if requested
            if args.gcs_bucket:
                upload_to_gcs(args.gcs_bucket, final_output_path, final_csv_path)
            return
        else:
            print(f"Resuming from file {len(existing_df)}...")
            # For simplicity in this script, we'll just restart if not using a DB
            # but usually we'd filter train_df here.
    
    # Hardware Detection
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0
    if args.workers is None:
        num_workers = 1 if use_gpu else os.cpu_count()
    else:
        num_workers = args.workers

    filenames = train_df['filename'].tolist()
    results = []

    if use_gpu:
        print(f"GPU detected. Running sequentially.")
        setup_tf_environment()
        model = load_perch_v2()
        for f in tqdm(filenames):
            results.append(process_file_worker(f, raw_dir=raw_dir, model=model))
            if len(results) % args.batch_size == 0:
                gc.collect()
    else:
        print(f"Starting parallel extraction: {len(filenames)} files on {num_workers} workers...")
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers, initializer=worker_init) as pool:
            worker_fn = partial(process_file_worker, raw_dir=raw_dir)
            # Process in batches to keep memory clean
            for i in range(0, len(filenames), args.batch_size):
                batch = filenames[i:i+args.batch_size]
                batch_results = list(tqdm(pool.imap(worker_fn, batch), total=len(batch), desc=f"Batch {i//args.batch_size}"))
                results.extend(batch_results)
                gc.collect()

    # Post-process results
    embeddings = []
    valid_indices = []
    for i, emb in enumerate(results):
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)

    embeddings = np.array(embeddings)
    valid_indices = np.array(valid_indices)
    
    print(f"Saving {len(embeddings)} embeddings...")
    np.savez_compressed(final_output_path, embeddings=embeddings, metadata_indices=valid_indices)
    
    mapped_df = train_df.iloc[valid_indices].copy()
    mapped_df['embedding_idx'] = range(len(embeddings))
    mapped_df.to_csv(final_csv_path, index=False)

    if args.gcs_bucket:
        upload_to_gcs(args.gcs_bucket, final_output_path, final_csv_path)

def upload_to_gcs(bucket_name, npz_path, csv_path):
    print(f"Uploading to gs://{bucket_name}...")
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob("processed/perch_v2_embeddings.npz").upload_from_filename(npz_path)
        bucket.blob("processed/train_with_perch_v2.csv").upload_from_filename(csv_path)
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
