import os
import warnings

# Force TF settings for BirdNET (TFLite or SavedModel)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import gc
import urllib.request

def setup_tf_environment():
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

def load_birdnet_model(model_path=None):
    """
    Downloads and loads the BirdNET-Analyzer V2.4 Model.
    """
    if model_path is None:
        model_path = "data/models/birdnet_v2.4.tflite"
        os.makedirs("data/models", exist_ok=True)
        
        if not os.path.exists(model_path):
            print("Model not found. On Vertex AI, ensure the model is provided in the image or bucket.")
            # Fallback download logic (might still fail due to 404/GitHub restrictions)
            model_url = "https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                print(f"Download failed: {e}")

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, model_path

def worker_init(model_path):
    global birdnet_interpreter
    setup_tf_environment()
    birdnet_interpreter, _ = load_birdnet_model(model_path)

def process_file_worker(filename, raw_dir):
    file_path = os.path.join(raw_dir, "train_audio", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"

    try:
        audio, _ = librosa.load(file_path, sr=48000, duration=3)
        if len(audio) < 144000: # 48000 * 3
            audio = np.pad(audio, (0, 144000 - len(audio)))
        
        input_details = birdnet_interpreter.get_input_details()
        output_details = birdnet_interpreter.get_output_details()
        input_data = np.array(audio[np.newaxis, :], dtype=np.float32)
        birdnet_interpreter.set_tensor(input_details[0]['index'], input_data)
        birdnet_interpreter.invoke()
        output_data = birdnet_interpreter.get_tensor(output_details[0]['index'])
        
        return output_data[0] 
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Extract BirdNET Embeddings.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. DOWNLOAD DATA IF MISSING
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print(f"Data missing in {raw_dir}. Downloading via Kaggle CLI...")
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if not os.path.exists(zip_path):
            os.system(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}")
        if os.path.exists(zip_path):
            os.system(f"unzip -qo {zip_path} -d {raw_dir}")
            os.remove(zip_path)

    # Load model once
    _, model_path = load_birdnet_model()
    
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit: train_df = train_df.head(args.limit)
    
    filenames = train_df['filename'].tolist()
    results = []

    print(f"BirdNET Mode: Using {args.workers} workers...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=worker_init, initargs=(model_path,)) as pool:
        for i in range(0, len(filenames), args.batch_size):
            batch = filenames[i:i+args.batch_size]
            batch_results = list(tqdm(pool.imap(partial(process_file_worker, raw_dir=raw_dir), batch), 
                                     total=len(batch), desc=f"Batch {i//args.batch_size + 1}"))
            results.extend(batch_results)
            gc.collect()

    # Diagnostics
    errors = [res for res in results if isinstance(res, str) and res.startswith("ERROR")]
    valid_results = [res for res in results if not (isinstance(res, str) and res.startswith("ERROR"))]

    if valid_results:
        valid_idx = [i for i, r in enumerate(results) if not (isinstance(r, str) and r.startswith("ERROR"))]
        embeddings = np.array(valid_results)
        final_output_path = os.path.join(processed_dir, "birdnet_embeddings.npz")
        
        np.savez_compressed(final_output_path, embeddings=embeddings, indices=valid_idx)
        print(f"Successfully saved {len(embeddings)} BirdNET embeddings.")

        if args.gcs_bucket:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(args.gcs_bucket)
                bucket.blob("processed/birdnet_embeddings.npz").upload_from_filename(final_output_path)
                print(f"Uploaded BirdNET results to gs://{args.gcs_bucket}")
            except Exception as e:
                print(f"GCS Upload Failed: {e}")
    else:
        print("No BirdNET embeddings extracted.")

if __name__ == "__main__":
    main()
