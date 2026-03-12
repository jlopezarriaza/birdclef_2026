import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import kagglehub
from tqdm import tqdm
import argparse

# Hardware-aware environment setup
# On Linux/GPU (GCP), we want full speed. 
# On Intel Mac (local), we stay in safe mode.
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU detected. Enabling optimized execution.")
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
else:
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

def load_perch_v2():
    """Download and load the Perch v2 model."""
    # Using the v2 CPU variant which is optimized for broad compatibility
    model_url = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
    print(f"Loading Perch v2 from {model_url}...")
    model = hub.load(model_url)
    return model

def process_file(file_path, model, sr=32000, duration=5):
    """Loads first 5s of audio and extracts embedding."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        num_samples = duration * sr
        if len(audio) < num_samples:
            audio = np.pad(audio, (0, num_samples - len(audio)))
        
        # Model expects [1, 160000]
        inputs = audio[np.newaxis, :].astype(np.float32)
        
        # Perch v2 via hub.load can be called directly
        outputs = model(inputs)
        
        return outputs['embedding'].numpy()[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract Perch v2 embeddings for training set.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files for testing.")
    parser.add_argument("--batch_save", type=int, default=500, help="Save progress every N files.")
    parser.add_argument("--output_name", type=str, default="perch_v2_embeddings.npz")
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
        
    print(f"Extraction target: {len(train_df)} files.")
    model = load_perch_v2()
    
    embeddings = []
    valid_indices = []
    
    output_path = os.path.join(processed_dir, args.output_name)
    
    for i, (idx, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df))):
        audio_path = os.path.join(raw_dir, "train_audio", row['filename'])
        emb = process_file(audio_path, model)
        
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(idx)
            
        if (i + 1) % args.batch_save == 0:
            np.savez_compressed(
                output_path + ".tmp", 
                embeddings=np.array(embeddings), 
                metadata_indices=np.array(valid_indices)
            )

    embeddings = np.array(embeddings)
    valid_indices = np.array(valid_indices)
    
    print(f"Saving {len(embeddings)} embeddings to {output_path}...")
    np.savez_compressed(
        output_path, 
        embeddings=embeddings, 
        metadata_indices=valid_indices
    )
    
    mapped_df = train_df.loc[valid_indices].copy()
    mapped_df['embedding_idx'] = range(len(embeddings))
    mapped_df.to_csv(os.path.join(processed_dir, "train_with_perch_v2.csv"), index=False)
    
    if os.path.exists(output_path + ".tmp"):
        os.remove(output_path + ".tmp")
        
    print("Extraction Complete.")

if __name__ == "__main__":
    main()
