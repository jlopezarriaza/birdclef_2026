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
import subprocess

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

def load_perch_v1(model_path=None):
    """Load stable Perch v1. If path is None, download via kagglehub."""
    if model_path is None:
        import kagglehub
        print("Downloading Perch v1...")
        model_path = kagglehub.model_download('google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/4')
    
    model = tf.saved_model.load(model_path)
    return model, model_path

def load_fusion_model(model_path):
    """Load the trained Fusion Model."""
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def worker_init(perch_model_path, fusion_model_path, known_species):
    global perch_instance, fusion_instance, spec_gen, species_list
    # Prevent all workers from hitting the disk at the exact same microsecond
    import time
    import random
    time.sleep(random.random() * 2) 
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    perch_instance, _ = load_perch_v1(perch_model_path)
    fusion_instance = load_fusion_model(fusion_model_path)
    spec_gen = SpectrogramGenerator(img_size=224)
    species_list = known_species

def extract_metadata(row):
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

def calculate_rms(audio, window_size=160000, stride=160000):
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
        audio, sr = sf.read(file_path)
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        window_size = 160000 # 5s
        stride = 160000      # 5s (CONSECUTIVE WINDOWS)
        
        if len(audio) < window_size:
            audio = np.pad(audio, (0, window_size - len(audio)))
            num_windows = 1
        else:
            num_windows = len(audio) // window_size
        
        is_known = target_species in species_list
        method = "fusion" if is_known else "rms"
        best_start_idx = 0
        best_confidence = 0.0
        
        if is_known:
            target_idx = species_list.index(target_species)
            windows_audio = []
            for i in range(num_windows):
                start = i * stride
                windows_audio.append(audio[start : start + window_size])
            
            X_spec, X_perch, X_meta = [], [], []
            meta = extract_metadata(row_dict)
            infer = perch_instance.signatures['serving_default']
            input_key = list(infer.structured_input_signature[1].keys())[0]
            
            # Use small batches
            batch_size = 8
            for i in range(0, num_windows, batch_size):
                batch_audio = windows_audio[i : i + batch_size]
                batch_spec = [spec_gen.generate_from_audio(a) / 255.0 for a in batch_audio]
                X_spec.extend(batch_spec)
                inputs = tf.constant(np.array(batch_audio, dtype=np.float32))
                outputs = infer(**{input_key: inputs})
                if 'embedding' in outputs:
                    X_perch.extend(outputs['embedding'].numpy())
                elif 'output_1' in outputs:
                    X_perch.extend(outputs['output_1'].numpy())
                X_meta.extend([meta] * len(batch_audio))
            
            preds = fusion_instance.predict([np.array(X_spec), np.array(X_perch), np.array(X_meta)], verbose=0)
            probs_for_target = preds[:, target_idx]
            best_idx = np.argmax(probs_for_target)
            best_confidence = probs_for_target[best_idx]
            best_start_idx = best_idx
            
            if best_confidence < 0.15:
                method = "fusion_fallback_rms"
                rms_scores = calculate_rms(audio, window_size, stride)
                if len(rms_scores) > 0:
                    best_start_idx = np.argmax(rms_scores)
        else:
            rms_scores = calculate_rms(audio, window_size, stride)
            if len(rms_scores) > 0:
                best_start_idx = np.argmax(rms_scores)
                best_confidence = rms_scores[best_start_idx]
            else:
                best_start_idx = 0
                best_confidence = 0.0
            
        start_time = best_start_idx * 5.0 # Consecutive windows, index * 5s
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

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return False
    return True

def sync_from_gcs(bucket, local_dir, remote_path):
    print(f"Syncing {remote_path} from gs://{bucket}...")
    os.makedirs(local_dir, exist_ok=True)
    # Use rsync for efficiency and robustness
    return run_command(f"gsutil -m rsync -r gs://{bucket}/{remote_path} {local_dir}")

def setup_kaggle_creds():
    """Setup kaggle.json if environment variables are present."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")
    if username and key:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        print("Kaggle credentials configured.")
        return True
    return False

def download_from_kaggle(raw_dir):
    """Download competition data if missing."""
    if not os.path.exists(os.path.join(raw_dir, "train.csv")):
        print("Data missing. Downloading from Kaggle...")
        setup_kaggle_creds()
        zip_path = os.path.join(raw_dir, "birdclef-2026.zip")
        if run_command(f"kaggle competitions download -c birdclef-2026 -p {raw_dir}"):
            if os.path.exists(zip_path):
                run_command(f"unzip -qo {zip_path} -d {raw_dir}")
                os.remove(zip_path)
                return True
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    args = parser.parse_args()

    raw_dir, processed_dir = "data/raw", "data/processed"
    model_path = "models/fusion_model_v1.keras"
    registry_path = "data/processed/species_registry.json"
    train_perch_csv = "data/processed/train_with_perch_v1.csv"
    output_path = os.path.join(processed_dir, "train_v2_peaks_fast.csv")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "train_audio"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    bucket = args.gcs_bucket or os.getenv("GCS_BUCKET")
    if bucket:
        print(f"Cloud mode enabled. Using bucket: {bucket}")
        # Sync Registry and Metadata
        run_command(f"gsutil cp gs://{bucket}/processed/species_registry.json {registry_path}")
        run_command(f"gsutil cp gs://{bucket}/processed/train_with_perch_v1.csv {train_perch_csv}")
        run_command(f"gsutil cp gs://{bucket}/models/fusion_model_v1.keras {model_path}")
        
        # Download Audio
        if not sync_from_gcs(bucket, os.path.join(raw_dir, "train_audio"), "raw/train_audio"):
            print("GCS Audio Sync failed. Falling back to Kaggle...")
            download_from_kaggle(raw_dir)
    else:
        # Local mode: check Kaggle anyway
        download_from_kaggle(raw_dir)

    if not os.path.exists(train_perch_csv):
        print(f"CRITICAL ERROR: {train_perch_csv} missing.")
        return

    # 1. Prepare Models and Data
    print("Preparing models...")
    setup_kaggle_creds()
    _, perch_model_path = load_perch_v1() # Download once
    
    df_train_perch = pd.read_csv(train_perch_csv)
    counts = df_train_perch['primary_label'].value_counts()
    known_species = sorted(df_train_perch[df_train_perch['primary_label'].isin(counts[counts >= 2].index)]['primary_label'].unique().tolist())
    
    train_csv_path = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        if bucket: run_command(f"gsutil cp gs://{bucket}/raw/train.csv {train_csv_path}")
        if not os.path.exists(train_csv_path):
            print("CRITICAL ERROR: train.csv missing.")
            return

    train_df = pd.read_csv(train_csv_path)
    if args.limit: train_df = train_df.head(args.limit)
    rows = train_df.to_dict('records')
    
    print(f"FAST Peak Selection ({len(rows)} files, 5s stride) with {args.workers} workers...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=worker_init, initargs=(perch_model_path, model_path, known_species)) as pool:
        process_func = partial(process_file_worker, raw_dir=raw_dir)
        results = []
        checkpoint_interval = 1000
        for i in range(0, len(rows), checkpoint_interval):
            batch_rows = rows[i : i + checkpoint_interval]
            batch_results = list(tqdm(pool.imap_unordered(process_func, batch_rows), total=len(batch_rows), desc=f"Batch {i//checkpoint_interval + 1}"))
            results.extend(batch_results)
            valid_results = [r for r in results if "error" not in r]
            pd.DataFrame(valid_results).to_csv(output_path, index=False)
            gc.collect()

    output_df = pd.DataFrame([r for r in results if "error" not in r])
    output_df.to_csv(output_path, index=False)
    print(f"\nFinal: Saved {len(output_df)} peak selections to {output_path}")
    if bucket: run_command(f"gsutil cp {output_path} gs://{bucket}/processed/train_v2_peaks_fast.csv")

if __name__ == "__main__":
    main()
