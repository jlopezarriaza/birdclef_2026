import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import kagglehub
from tqdm import tqdm

def load_perch_model():
    """Download and load the compatible Perch v1 model from Kaggle."""
    # This is the version we saw working earlier (v1 SavedModel)
    # google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier
    model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier')
    model = tf.saved_model.load(model_path)
    return model, model_path

def preprocess_audio(audio_path, target_sr=32000):
    """Load and resample audio."""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio

def run_inference(model, audio, window_size_s=5, sr=32000):
    """Run inference on 5-second windows of audio."""
    num_samples = window_size_s * sr
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))
    
    # Model expects float32
    audio_segment = audio[:num_samples].astype(np.float32)
    # Add batch dimension: [1, 160000]
    inputs = tf.constant(audio_segment[np.newaxis, :])
    
    # Use the serving_default signature
    infer = model.signatures['serving_default']
    # For Perch v1, the input key is 'input_1' or similar, let's check
    input_key = list(infer.structured_input_signature[1].keys())[0]
    outputs = infer(**{input_key: inputs})
    
    return outputs

if __name__ == "__main__":
    # 1. Setup Kaggle Auth
    os.environ["KAGGLE_USERNAME"] = "juanlopezarriaza"
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_45b265057d2a434f2e2eec260e818fe9"

    print("Loading compatible Perch model...")
    perch, model_dir = load_perch_model()
    
    # 2. Find a sample file
    train_audio_dir = "data/raw/train_audio"
    species_folders = sorted([f for f in os.listdir(train_audio_dir) if os.path.isdir(os.path.join(train_audio_dir, f))])
    sample_species = species_folders[0]
    sample_files = [f for f in os.listdir(os.path.join(train_audio_dir, sample_species)) if f.endswith('.ogg')]
    sample_path = os.path.join(train_audio_dir, sample_species, sample_files[0])
    
    print(f"Processing sample: {sample_path}")
    audio = preprocess_audio(sample_path)
    
    print("Running inference...")
    results = run_inference(perch, audio)
    
    # 3. Explore outputs
    print("\nInference Results:")
    for key, val in results.items():
        print(f"- {key}: shape {val.shape}")
        
    # 'label' is the key for probabilities in Perch v1
    if 'label' in results:
        probs = results['label'].numpy()[0]
        top_idx = np.argmax(probs)
        print(f"\nTop Prediction Index: {top_idx}")
        print(f"Confidence: {probs[top_idx]:.4f}")
        
        # Perch v1 labels
        label_csv = os.path.join(model_dir, "assets/label.csv")
        if os.path.exists(label_csv):
            labels = pd.read_csv(label_csv)
            predicted_label = labels.iloc[top_idx]
            print(f"Predicted Species: {predicted_label.to_dict()}")
