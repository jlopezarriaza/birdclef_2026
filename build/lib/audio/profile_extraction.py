import os
import time
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import kagglehub
from tqdm import tqdm

def load_perch_model():
    model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier')
    model = tf.saved_model.load(model_path)
    return model

def profile_extraction(num_samples=20):
    raw_dir = "data/raw"
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv")).head(num_samples)
    
    model = load_perch_model()
    infer = model.signatures['serving_default']
    input_key = list(infer.structured_input_signature[1].keys())[0]
    
    load_times = []
    infer_times = []
    
    print(f"Profiling {num_samples} files...")
    
    for _, row in train_df.iterrows():
        audio_path = os.path.join(raw_dir, "train_audio", row['filename'])
        
        # 1. Profile Loading
        start_load = time.perf_counter()
        audio, _ = librosa.load(audio_path, sr=32000, duration=5)
        num_samples_needed = 5 * 32000
        if len(audio) < num_samples_needed:
            audio = np.pad(audio, (0, num_samples_needed - len(audio)))
        audio_input = audio[np.newaxis, :].astype(np.float32)
        load_times.append(time.perf_counter() - start_load)
        
        # 2. Profile Inference
        inputs = tf.constant(audio_input)
        start_infer = time.perf_counter()
        _ = infer(**{input_key: inputs})
        infer_times.append(time.perf_counter() - start_infer)
        
    avg_load = np.mean(load_times)
    avg_infer = np.mean(infer_times)
    total_avg = avg_load + avg_infer
    
    print(f"\nResults (Averages per file):")
    print(f"- Audio Loading/Resampling: {avg_load:.4f}s ({avg_load/total_avg*100:.1f}%)")
    print(f"- Model Inference (TF):     {avg_infer:.4f}s ({avg_infer/total_avg*100:.1f}%)")
    print(f"- Total per file:           {total_avg:.4f}s")
    
    estimated_total_hours = (total_avg * 35550) / 3600
    print(f"\nEstimated time for 35,550 files: {estimated_total_hours:.2f} hours")

if __name__ == "__main__":
    profile_extraction()
