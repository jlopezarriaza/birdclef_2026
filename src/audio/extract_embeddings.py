import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import kagglehub
from tqdm import tqdm
import argparse

# Hardware-aware environment setup
if "COLAB_GPU" in os.environ or len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU detected. Enabling optimized execution.")
    # In Colab/GPU, we want XLA enabled for speed
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
else:
    # On Intel Mac CPU, keep XLA disabled to avoid deserialization errors
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

def load_perch_model():
    """Download and load the compatible Perch v1 model."""
    model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier')
    model = tf.saved_model.load(model_path)
    return model

def process_file(file_path, model, infer_fn, input_key, sr=32000, duration=5):
    """Loads first 5s of audio and extracts embedding."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad if shorter than 5s
        num_samples = duration * sr
        if len(audio) < num_samples:
            audio = np.pad(audio, (0, num_samples - len(audio)))
        
        # Model expects [1, 160000]
        inputs = tf.constant(audio[np.newaxis, :].astype(np.float32))
        outputs = infer_fn(**{input_key: inputs})
        
        return outputs['embedding'].numpy()[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract Perch embeddings for training set.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files for testing.")
    parser.add_argument("--batch_save", type=int, default=500, help="Save progress every N files.")
    args = parser.parse_args()

    # Paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
        
    print(f"Loading Perch model and preparing for extraction of {len(train_df)} files...")
    model = load_perch_model()
    infer = model.signatures['serving_default']
    input_key = list(infer.structured_input_signature[1].keys())[0]
    
    embeddings = []
    valid_indices = []
    
    output_path = os.path.join(processed_dir, "perch_v1_embeddings.npz")
    
    for i, (idx, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df))):
        audio_path = os.path.join(raw_dir, "train_audio", row['filename'])
        emb = process_file(audio_path, model, infer, input_key)
        
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(idx)
            
        # Periodic save/progress update could be added here for huge datasets
        if (i + 1) % args.batch_save == 0:
            # Checkpoint
            np.savez_compressed(
                output_path + ".tmp", 
                embeddings=np.array(embeddings), 
                metadata_indices=np.array(valid_indices)
            )

    # Final Save
    embeddings = np.array(embeddings)
    valid_indices = np.array(valid_indices)
    
    print(f"Saving {len(embeddings)} embeddings to {output_path}...")
    np.savez_compressed(
        output_path, 
        embeddings=embeddings, 
        metadata_indices=valid_indices
    )
    
    # Also save a 'mapped' CSV that only includes the rows we successfully processed
    mapped_df = train_df.loc[valid_indices].copy()
    mapped_df['embedding_idx'] = range(len(embeddings))
    mapped_df.to_csv(os.path.join(processed_dir, "train_with_embeddings.csv"), index=False)
    
    if os.path.exists(output_path + ".tmp"):
        os.remove(output_path + ".tmp")
        
    print("Done!")

if __name__ == "__main__":
    main()
