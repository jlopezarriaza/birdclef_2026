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

# Hardware-aware environment setup
def setup_tf_environment():
    if len(tf.config.list_physical_devices('GPU')) > 0:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
    else:
        # On CPU, we restrict each process to 1 thread to avoid thrashing 32 cores
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

def load_perch_v2():
    """Download and load the Perch v2 model."""
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
    # Note: hub.load is cached, so only one download occurs
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
        return outputs['embedding'].numpy()[0]
    except Exception as e:
        # Don't print thousands of errors, just return None
        return None

def main():
    parser = argparse.ArgumentParser(description="Parallel Extraction of Perch v2 embeddings.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Hardware Detection
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0
    
    # Default workers: 1 if GPU (to avoid OOM), else cpu_count
    if args.workers is None:
        num_workers = 1 if use_gpu else os.cpu_count()
    else:
        num_workers = args.workers

    # 2. Check for data
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print("Downloading data from Kaggle...")
        os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        os.system(f"unzip -qo {raw_dir}/birdclef-2026.zip -d {raw_dir}")

    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
    
    filenames = train_df['filename'].tolist()
    
    if use_gpu:
        print(f"GPU detected. Running in high-speed sequential mode.")
        setup_tf_environment()
        model = load_perch_v2()
        results = []
        for f in tqdm(filenames):
            results.append(process_file_worker(f, raw_dir=raw_dir, model=model))
    else:
        print(f"No GPU. Starting parallel extraction on {num_workers} CPU workers...")
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers, initializer=worker_init) as pool:
            worker_fn = partial(process_file_worker, raw_dir=raw_dir)
            results = list(tqdm(pool.imap(worker_fn, filenames), total=len(filenames)))

    # Post-process results
    embeddings = []
    valid_indices = []
    
    for i, emb in enumerate(results):
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i) # Original index in train_df

    embeddings = np.array(embeddings)
    valid_indices = np.array(valid_indices)
    
    output_path = os.path.join(processed_dir, "perch_v2_embeddings.npz")
    print(f"Saving {len(embeddings)} embeddings to {output_path}...")
    np.savez_compressed(output_path, embeddings=embeddings, metadata_indices=valid_indices)
    
    mapped_df = train_df.iloc[valid_indices].copy()
    mapped_df['embedding_idx'] = range(len(embeddings))
    csv_path = os.path.join(processed_dir, "train_with_perch_v2.csv")
    mapped_df.to_csv(csv_path, index=False)

    # GCS Upload
    if args.gcs_bucket:
        print(f"Uploading to gs://{args.gcs_bucket}...")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(args.gcs_bucket)
            bucket.blob("processed/perch_v2_embeddings.npz").upload_from_filename(output_path)
            bucket.blob("processed/train_with_perch_v2.csv").upload_from_filename(csv_path)
            print("Upload successful!")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
