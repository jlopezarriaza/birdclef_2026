import os
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from datetime import datetime
import json
import gc

# Internal imports
from src.audio.spectrograms import SpectrogramGenerator

# FORCE TF TO BE QUIET
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

def load_perch_v1():
    """Load stable Perch v1 from kagglehub."""
    import kagglehub
    model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4')
    model = tf.saved_model.load(model_path)
    return model

def load_fusion_model(model_path):
    """Load the trained Fusion Model."""
    # Use compile=False because we don't need to train, and custom metrics might be missing
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def worker_init(fusion_model_path, known_species):
    global perch_instance, fusion_instance, spec_gen, species_list
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    perch_instance = load_perch_v1()
    fusion_instance = load_fusion_model(fusion_model_path)
    spec_gen = SpectrogramGenerator(img_size=224)
    species_list = known_species

def extract_metadata(row):
    """Same logic as FusionDataGenerator."""
    lat = row.get('latitude', 0.0) / 90.0
    lon = row.get('longitude', 0.0) / 180.0
    
    month = 0.5
    day_of_year = 0.5
    
    try:
        if 'url' in row and '?' in str(row['url']):
            ts_str = str(row['url']).split('?')[-1]
            if ts_str.isdigit():
                dt = datetime.fromtimestamp(int(ts_str))
                month = dt.month / 12.0
                day_of_year = dt.timetuple().tm_yday / 366.0
        elif 'BC2026_Train' in str(row['filename']):
            date_part = str(row['filename']).split('_')[4]
            dt = datetime.strptime(date_part, '%Y%m%d')
            month = dt.month / 12.0
            day_of_year = dt.timetuple().tm_yday / 366.0
    except:
        pass
        
    return np.array([lat, lon, month, day_of_year], dtype=np.float32)

def calculate_rms(audio, window_size=160000, stride=32000):
    """Calculate RMS energy for overlapping windows."""
    rms_scores = []
    for start in range(0, len(audio) - window_size + 1, stride):
        window = audio[start : start + window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_scores.append(rms)
    return rms_scores

def process_file_worker(row_dict, raw_dir):
    filename = row_dict['filename']
    target_species = row_dict['primary_label']
    file_path = os.path.join(raw_dir, "train_audio", filename)
    
    if not os.path.exists(file_path):
        return {"filename": filename, "error": f"Missing {file_path}"}

    try:
        # 1. Load Audio
        # Use soundfile for speed
        audio, sr = sf.read(file_path)
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            sr = 32000
        
        # Handle multi-channel
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        window_size = 160000 # 5s
        stride = 32000      # 1s
        
        # If file is shorter than 5s, only one window (padded)
        if len(audio) < window_size:
            audio = np.pad(audio, (0, window_size - len(audio)))
            num_windows = 1
        else:
            num_windows = (len(audio) - window_size) // stride + 1
        
        # 2. Decision Logic
        is_known = target_species in species_list
        method = "fusion" if is_known else "rms"
        
        best_start_idx = 0
        best_confidence = 0.0
        
        if is_known:
            # --- Fusion Guided ---
            target_idx = species_list.index(target_species)
            
            # Prepare batches for inference
            windows_audio = []
            for i in range(num_windows):
                start = i * stride
                windows_audio.append(audio[start : start + window_size])
            
            # Extract features for all windows
            X_spec = []
            X_perch = []
            X_meta = []
            
            meta = extract_metadata(row_dict)
            
            infer = perch_instance.signatures['serving_default']
            input_key = list(infer.structured_input_signature[1].keys())[0]
            
            # Process in small batches to avoid TF OOM
            batch_size = 8
            
            for i in range(0, num_windows, batch_size):
                batch_audio = windows_audio[i : i + batch_size]
                
                # Spectrograms
                batch_spec = [spec_gen.generate_from_audio(a) / 255.0 for a in batch_audio]
                X_spec.extend(batch_spec)
                
                # Perch Embeddings
                inputs = tf.constant(np.array(batch_audio, dtype=np.float32))
                outputs = infer(**{input_key: inputs})
                if 'embedding' in outputs:
                    batch_emb = outputs['embedding'].numpy()
                elif 'output_1' in outputs:
                    batch_emb = outputs['output_1'].numpy()
                X_perch.extend(batch_emb)
                
                # Metadata (identical for all windows in one file)
                X_meta.extend([meta] * len(batch_audio))
            
            # Run Fusion Inference
            # fusion_instance expectations: [input_spectrogram, input_perch_embedding, input_metadata]
            preds = fusion_instance.predict([np.array(X_spec), np.array(X_perch), np.array(X_meta)], verbose=0)
            
            # Find window with highest prob for target species
            probs_for_target = preds[:, target_idx]
            best_idx = np.argmax(probs_for_target)
            best_confidence = probs_for_target[best_idx]
            best_start_idx = best_idx
            
            # Fallback if confidence is too low
            if best_confidence < 0.15:
                method = "fusion_fallback_rms"
                rms_scores = calculate_rms(audio, window_size, stride)
                if len(rms_scores) > 0:
                    best_start_idx = np.argmax(rms_scores)
        else:
            # --- Energy Based (Gap Species) ---
            rms_scores = calculate_rms(audio, window_size, stride)
            if len(rms_scores) > 0:
                best_start_idx = np.argmax(rms_scores)
                best_confidence = rms_scores[best_start_idx]
            else:
                best_start_idx = 0
                best_confidence = 0.0
            
        start_time = best_start_idx
        end_time = start_time + 5.0
        
        return {
            "filename": filename,
            "primary_label": target_species,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "confidence": float(best_confidence),
            "method": method
        }
        
    except Exception as e:
        return {"filename": filename, "error": str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    model_path = "models/fusion_model_v1.keras"
    registry_path = "data/processed/species_registry.json"
    train_perch_csv = "data/processed/train_with_perch_v1.csv"
    output_path = os.path.join(processed_dir, "train_v2_peaks.csv")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Determine Known Species (the 202 used in training)
    print("Loading training metadata to identify known species...")
    df_train_perch = pd.read_csv(train_perch_csv)
    counts = df_train_perch['primary_label'].value_counts()
    known_species = sorted(df_train_perch[df_train_perch['primary_label'].isin(counts[counts >= 2].index)]['primary_label'].unique().tolist())
    print(f"Identified {len(known_species)} known species.")
    
    # 2. Load Registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # 3. Load full train.csv
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
    
    rows = train_df.to_dict('records')
    print(f"Processing {len(rows)} files with {args.workers} workers...")
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=worker_init, initargs=(model_path, known_species)) as pool:
        process_func = partial(process_file_worker, raw_dir=raw_dir)
        results = []
        
        checkpoint_interval = 500
        for i in range(0, len(rows), checkpoint_interval):
            batch_rows = rows[i : i + checkpoint_interval]
            batch_results = list(tqdm(pool.imap_unordered(process_func, batch_rows), 
                                     total=len(batch_rows), 
                                     desc=f"Batch {i//checkpoint_interval + 1}"))
            results.extend(batch_results)
            
            # Intermediate save
            valid_results = [r for r in results if "error" not in r]
            pd.DataFrame(valid_results).to_csv(output_path, index=False)
            gc.collect()

    # 4. Final results
    final_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(output_path, index=False)
    
    print(f"\nFinal: Saved {len(output_df)} peak selections to {output_path}")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err['filename']}: {err['error']}")

if __name__ == "__main__":
    main()
