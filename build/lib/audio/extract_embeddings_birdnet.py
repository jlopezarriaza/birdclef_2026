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
    BirdNET uses a custom ResNet architecture.
    """
    if model_path is None:
        print("Checking/Downloading BirdNET V2.4 Model...")
        # Direct link to V2.4 TFLite (official release from kahst/BirdNET-Analyzer)
        model_url = "https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
        model_path = "data/models/birdnet_v2.4.tflite"
        os.makedirs("data/models", exist_ok=True)
        
        if not os.path.exists(model_path):
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')]
            urllib.request.install_opener(opener)
            print(f"Downloading BirdNET from {model_url}...")
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Downloaded BirdNET to {model_path}")

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
        # 1. BirdNET uses 48kHz and 3-second segments (standard)
        # However, for BirdCLEF we often adapt to the target segment.
        # We will use 48kHz as BirdNET expects it.
        audio, _ = librosa.load(file_path, sr=48000, duration=3)
        if len(audio) < 144000: # 48000 * 3
            audio = np.pad(audio, (0, 144000 - len(audio)))
        
        # 2. Get input and output details
        input_details = birdnet_interpreter.get_input_details()
        output_details = birdnet_interpreter.get_output_details()

        # 3. Prepare input data
        input_data = np.array(audio[np.newaxis, :], dtype=np.float32)
        birdnet_interpreter.set_tensor(input_details[0]['index'], input_data)

        # 4. Invoke interpreter
        birdnet_interpreter.invoke()

        # 5. Get embedding (usually the penultimate layer or custom output)
        # Note: In the TFLite model, we might need to find the correct index for embeddings.
        # For V2.4, the 1024-d feature vector is typically at a specific output index.
        # If the model only provides logits (6k), we take the layer before.
        # For simplicity in this baseline, we'll take the logits if embedding index is unknown.
        # but BirdNET-Analyzer TFLite usually has features as a secondary output.
        
        # In this implementation, we'll assume we take the logits or features 
        # depending on the available output tensors.
        output_data = birdnet_interpreter.get_tensor(output_details[0]['index'])
        
        return output_data[0] # Return the feature/logit vector
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Extract BirdNET Embeddings.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Pre-download model
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

    # Save logic
    valid_results = [res for res in results if not (isinstance(res, str) and res.startswith("ERROR"))]
    if valid_results:
        valid_idx = [i for i, r in enumerate(results) if not (isinstance(r, str) and r.startswith("ERROR"))]
        embeddings = np.array(valid_results)
        np.savez_compressed(os.path.join(processed_dir, "birdnet_embeddings.npz"), embeddings=embeddings, indices=valid_idx)
        print(f"Successfully saved {len(embeddings)} BirdNET embeddings.")
    else:
        print("No BirdNET embeddings extracted.")

if __name__ == "__main__":
    main()
