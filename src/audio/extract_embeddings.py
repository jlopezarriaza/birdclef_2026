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
    # Force XLA OFF for CPU stability unless specifically requested
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU detected. Enabling some optimizations...")
        # Note: If we had a GPU, we might want XLA, but for now we stick to CPU.
    else:
        # Optimizations for CPU inference (Vertex n1-standard-32)
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

def load_model_optimized():
    """Load stable Perch v1 (v4). Much more reliable on CPU than v2."""
    print("Loading stable Perch v1 (version 4) for CPU inference...")
    # Perch v1 (v4) is highly compatible and robust
    model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4')
    return tf.saved_model.load(model_path)

def worker_init():
    """Initializer for multiprocessing pool: loads the model into each process."""
    global model_instance
    setup_tf_environment()
    try:
        model_instance = load_model_optimized()
    except Exception as e:
        print(f"WORKER INIT ERROR: {e}")

def process_file_worker(filename, raw_dir, model=None):
    """Worker function to process a single file."""
    file_path = os.path.join(raw_dir, "train_audio", filename)
    target_model = model if model is not None else model_instance
    
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=32000, duration=5)
        num_samples_needed = 160000
        if len(audio) < num_samples_needed:
            audio = np.pad(audio, (0, num_samples_needed - len(audio)))
        
        # Robust Inference: Try different call methods
        inputs = tf.constant(audio[np.newaxis, :].astype(np.float32))
        
        # 1. Try serving_default signature (Works for hub.load V2 and SavedModel V1)
        if hasattr(target_model, 'signatures'):
            infer = target_model.signatures['serving_default']
            # Find the input key (V1: input_1, V2: inputs)
            input_key = list(infer.structured_input_signature[1].keys())[0]
            outputs = infer(**{input_key: inputs})
        else:
            # 2. Direct call fallback
            outputs = target_model(inputs)
        
        if 'embedding' not in outputs:
            return f"ERROR: 'embedding' key missing. Keys: {list(outputs.keys())}"
            
        emb = outputs['embedding'].numpy()[0]
        
        # Cleanup
        del audio
        del inputs
        del outputs
        
        return emb
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Parallel Extraction of Perch embeddings.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Kaggle Auth
    if os.getenv("KAGGLE_API_TOKEN") and not os.getenv("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

    # Download if missing
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print(f"Data missing in {raw_dir}. Downloading via CLI...")
        os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if os.path.exists(zip_path):
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)

    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
    
    print(f"Successfully loaded {len(train_df)} rows of metadata.")
    
    final_output_path = os.path.join(processed_dir, "perch_embeddings.npz")
    final_csv_path = os.path.join(processed_dir, "train_with_perch.csv")
    
    # Hardware Detection
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = len(gpus) > 0
    if args.workers is None:
        # Default to safe cloud limits
        num_workers = 1 if use_gpu else min(16, os.cpu_count())
    else:
        num_workers = args.workers

    filenames = train_df['filename'].tolist()
    results = []

    if use_gpu:
        print(f"GPU Mode: Sequential processing...")
        setup_tf_environment()
        model = load_model_optimized()
        for f in tqdm(filenames):
            res = process_file_worker(f, raw_dir=raw_dir, model=model)
            results.append(None if (isinstance(res, str) and res.startswith("ERROR")) else res)
            if len(results) % args.batch_size == 0: gc.collect()
    else:
        print(f"CPU Mode: Parallel processing on {num_workers} workers...")
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers, initializer=worker_init) as pool:
            worker_fn = partial(process_file_worker, raw_dir=raw_dir)
            for i in range(0, len(filenames), args.batch_size):
                batch = filenames[i:i+args.batch_size]
                batch_results = list(tqdm(pool.imap(worker_fn, batch), total=len(batch), desc=f"Batch {i//args.batch_size}"))
                for res in batch_results:
                    if isinstance(res, str) and res.startswith("ERROR"):
                        print(res)
                        results.append(None)
                    else:
                        results.append(res)
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
        print(f"Uploading to gs://{args.gcs_bucket}...")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(args.gcs_bucket)
            bucket.blob("processed/perch_embeddings.npz").upload_from_filename(final_output_path)
            bucket.blob("processed/train_with_perch.csv").upload_from_filename(final_csv_path)
            print("Upload successful!")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
