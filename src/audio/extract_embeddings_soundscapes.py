import os
import warnings
import glob

# FORCE TF TO BE QUIET AND DISABLE XLA BEFORE ANY IMPORTS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import numpy as np
import librosa
import tensorflow as tf
import kagglehub
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

def load_model_v1(model_path=None):
    """Load stable Perch v1 (v4)."""
    if model_path is None:
        print("Checking/Downloading Perch v1 (version 4)...")
        model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4')
    
    model = tf.saved_model.load(model_path)
    return model, model_path

def worker_init(model_path):
    global model_instance
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    model_instance, _ = load_model_v1(model_path)

def process_soundscape_worker(filename, raw_dir, processed_dir):
    file_path = os.path.join(raw_dir, "train_soundscapes", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"
    
    out_path = os.path.join(processed_dir, f"{os.path.splitext(filename)[0]}.npz")
    if os.path.exists(out_path):
        return f"SKIPPED: {filename} already processed"

    try:
        # Load full 60s audio
        audio, _ = librosa.load(file_path, sr=32000, duration=60.0)
        
        # Pad if audio is shorter than 60s (unlikely for soundscapes, but safe)
        if len(audio) < 1920000:
            audio = np.pad(audio, (0, 1920000 - len(audio)))
        
        # Extract 56 overlapping 5-second windows
        window_size = 5 * 32000 # 160000 samples
        stride = 1 * 32000      # 32000 samples
        
        windows = []
        for i in range(56):
            start = i * stride
            end = start + window_size
            windows.append(audio[start:end])
            
        windows = np.array(windows, dtype=np.float32) # Shape: (56, 160000)
        
        infer = model_instance.signatures['serving_default']
        input_key = list(infer.structured_input_signature[1].keys())[0]
        
        # Batch inference in chunks of 8 to avoid memory spikes
        embeddings = []
        for i in range(0, 56, 8):
            batch = windows[i:i+8]
            inputs = tf.constant(batch)
            outputs = infer(**{input_key: inputs})
            
            if 'embedding' in outputs:
                emb = outputs['embedding'].numpy()
            elif 'output_1' in outputs:
                emb = outputs['output_1'].numpy()
            else:
                return f"ERROR: No embedding found for {filename}"
            embeddings.append(emb)
            
        embeddings = np.concatenate(embeddings, axis=0) # Shape: (56, 1280)
        
        np.savez_compressed(out_path, embeddings=embeddings)
        return "SUCCESS"
        
    except Exception as e:
        return f"ERROR: {filename} - {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed/soundscape_embeddings"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. DOWNLOAD DATA IF MISSING
    soundscapes_dir = os.path.join(raw_dir, "train_soundscapes")
    if not os.path.exists(soundscapes_dir):
        print(f"Data missing in {raw_dir}. Downloading via Kaggle CLI...")
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if not os.path.exists(zip_path):
            os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        if os.path.exists(zip_path):
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)

    # DOWNLOAD MODEL ONCE in the main process
    _, model_path = load_model_v1()
    
    # Get all .ogg files
    filenames = [os.path.basename(f) for f in glob.glob(os.path.join(soundscapes_dir, "*.ogg"))]
    if args.limit: 
        filenames = filenames[:args.limit]
    
    if not filenames:
        print(f"No .ogg files found in {soundscapes_dir}. Please ensure data is downloaded.")
        return

    print(f"Processing {len(filenames)} soundscapes using {args.workers} workers...")
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=worker_init, initargs=(model_path,)) as pool:
        process_func = partial(process_soundscape_worker, raw_dir=raw_dir, processed_dir=processed_dir)
        results = list(tqdm(pool.imap_unordered(process_func, filenames), total=len(filenames)))

    # Diagnostics
    errors = [res for res in results if res.startswith("ERROR")]
    successes = [res for res in results if res == "SUCCESS"]
    skipped = [res for res in results if res.startswith("SKIPPED")]
    
    print(f"\nFinished: {len(successes)} succeeded, {len(skipped)} skipped, {len(errors)} errors.")
    if errors:
        print(f"\nFirst 5 errors:")
        for err in errors[:5]:
            print(f"  - {err}")

if __name__ == "__main__":
    main()