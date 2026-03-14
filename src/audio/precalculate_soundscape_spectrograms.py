import os
import warnings
import glob
import numpy as np
import librosa
import cv2
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

from src.audio.spectrograms import SpectrogramGenerator, save_spectrogram

def process_soundscape_worker(filename, raw_dir, processed_dir, img_size=224):
    file_path = os.path.join(raw_dir, "train_soundscapes", filename)
    if not os.path.exists(file_path): 
        return f"ERROR: Missing {file_path}"
    
    # Soundscape output folder
    soundscape_name = os.path.splitext(filename)[0]
    out_dir = os.path.join(processed_dir, soundscape_name)
    
    # Check if already processed (56 windows)
    if os.path.exists(out_dir) and len(glob.glob(os.path.join(out_dir, "*.png"))) == 56:
        return f"SKIPPED: {filename} already processed"

    try:
        # Load full 60s audio
        audio, _ = librosa.load(file_path, sr=32000, duration=60.0)
        
        # Pad if audio is shorter than 60s
        if len(audio) < 1920000:
            audio = np.pad(audio, (0, 1920000 - len(audio)))
        
        # Generator (pre-configured for 224x224)
        generator = SpectrogramGenerator(img_size=img_size)
        
        # Extract 56 overlapping 5-second windows
        window_size = 5 * 32000 # 160000 samples
        stride = 1 * 32000      # 32000 samples
        
        for i in range(56):
            start = i * stride
            end = start + window_size
            window_audio = audio[start:end]
            
            img = generator.generate_from_audio(window_audio)
            if img is not None:
                out_path = os.path.join(out_dir, f"{i:02d}.png")
                save_spectrogram(img, out_path)
            else:
                return f"ERROR: Failed to generate spectrogram for {filename} window {i}"
            
        return "SUCCESS"
        
    except Exception as e:
        return f"ERROR: {filename} - {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    raw_dir = "data/raw"
    processed_dir = "data/processed/soundscape_spectrograms"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all .ogg files
    soundscapes_dir = os.path.join(raw_dir, "train_soundscapes")
    filenames = [os.path.basename(f) for f in glob.glob(os.path.join(soundscapes_dir, "*.ogg"))]
    if args.limit: 
        filenames = filenames[:args.limit]
    
    if not filenames:
        print(f"No .ogg files found in {soundscapes_dir}. Please ensure data is downloaded.")
        return

    print(f"Processing {len(filenames)} soundscapes using {args.workers} workers...")
    
    # Use spawn for multiprocessing to be safe with CV2/Librosa
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers) as pool:
        process_func = partial(
            process_soundscape_worker, 
            raw_dir=raw_dir, 
            processed_dir=processed_dir,
            img_size=args.img_size
        )
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
