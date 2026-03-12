import os
# Disable XLA locally if on Intel Mac to avoid VHLO errors
# This script will still be slow on CPU, but now uses the correct V2 logic
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import kagglehub
from tqdm import tqdm

def load_perch_v2():
    """Download and load the Perch v2 model."""
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
    print(f"Loading Perch v2 from {model_url}...")
    model = hub.load(model_url)
    model_dir = hub.resolve(model_url)
    return model, model_dir

def preprocess_audio(audio_path, target_sr=32000):
    """Load and resample audio."""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio

def run_inference(model, audio, window_size_s=5, sr=32000):
    """Run inference on 5-second windows of audio."""
    num_samples = window_size_s * sr
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))
    
    audio_segment = audio[:num_samples].astype(np.float32)
    inputs = audio_segment[np.newaxis, :]
    
    # Perch v2 via hub.load can be called directly
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    # Setup Kaggle Auth
    os.environ["KAGGLE_USERNAME"] = "juanlopezarriaza"
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_45b265057d2a434f2e2eec260e818fe9"

    print("Initializing Perch v2 Baseline...")
    perch, model_dir = load_perch_v2()
    
    # Find a sample file
    train_audio_dir = "data/raw/train_audio"
    species_folders = sorted([f for f in os.listdir(train_audio_dir) if os.path.isdir(os.path.join(train_audio_dir, f))])
    sample_species = species_folders[0]
    sample_files = [f for f in os.listdir(os.path.join(train_audio_dir, sample_species)) if f.endswith('.ogg')]
    sample_path = os.path.join(train_audio_dir, sample_species, sample_files[0])
    
    print(f"Processing sample: {sample_path}")
    audio = preprocess_audio(sample_path)
    
    print("Running inference...")
    # NOTE: This may still fail locally on Intel Mac due to VHLO errors
    # but is the correct logic for Cloud/GPU runs.
    try:
        results = run_inference(perch, audio)
        print("\nInference Results:")
        print(f"- Embedding shape: {results['embedding'].shape}")
        if 'label' in results:
            probs = results['label'].numpy()[0]
            top_idx = np.argmax(probs)
            print(f"- Top Prediction Index: {top_idx}")
            
            label_csv = os.path.join(model_dir, "assets/perch_v2_ebird_classes.csv")
            if os.path.exists(label_csv):
                labels = pd.read_csv(label_csv)
                predicted_species = labels.iloc[top_idx]
                print(f"- Predicted Species: {predicted_species.to_dict()}")
    except Exception as e:
        print(f"\nInference failed locally: {e}")
        print("This is expected on Intel Mac for Perch v2. Use the Docker/GCP pipeline for full support.")
