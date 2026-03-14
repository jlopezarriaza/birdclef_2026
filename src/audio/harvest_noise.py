import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path

def harvest_noise(
    labels_path="data/raw/train_soundscapes_labels.csv",
    audio_dir="data/raw/train_soundscapes/",
    output_dir="data/processed/noise_bank/",
    registry_path="data/processed/noise_bank_registry.csv",
    sample_rate=32000,
    segment_duration=5
):
    """
    Identifies and extracts segments from soundscapes that have no species labels.
    """
    # 1. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load labels
    df = pd.read_csv(labels_path)
    
    # Standardize time format if necessary (00:00:05 -> 5)
    def time_to_seconds(t_str):
        h, m, s = map(int, t_str.split(':'))
        return h * 3600 + m * 60 + s

    df['start_sec'] = df['start'].apply(time_to_seconds)
    df['end_sec'] = df['end'].apply(time_to_seconds)
    
    # 3. Identify noise segments
    # Get all unique soundscape files
    # Note: We assume all soundscape files are 60 seconds long based on competition info.
    soundscapes = [f for f in os.listdir(audio_dir) if f.endswith('.ogg')]
    
    noise_clips = []
    
    print(f"Analyzing {len(soundscapes)} soundscapes for noise...")
    
    for filename in tqdm(soundscapes):
        # Find labeled segments for this file
        file_labels = df[df['filename'] == filename]
        
        # Deduplicate labels (some files might have redundant rows)
        labeled_starts = set(file_labels['start_sec'].unique())
        
        # Check all 5s blocks in the 60s file
        # 0-5, 5-10, ..., 55-60
        for start in range(0, 60, segment_duration):
            if start not in labeled_starts:
                # This 5s block is a noise candidate
                noise_clips.append({
                    'filename': filename,
                    'start_sec': start,
                    'end_sec': start + segment_duration
                })
                
    print(f"Found {len(noise_clips)} noise segments.")
    
    # 4. Extract and save
    registry_data = []
    
    print("Extracting noise segments...")
    for i, clip in enumerate(tqdm(noise_clips)):
        src_path = os.path.join(audio_dir, clip['filename'])
        
        # Target filename for noise clip
        # Example: BC2026_Train_0006_S09_20250828_000000_noise_00-05.wav
        base_name = Path(clip['filename']).stem
        target_filename = f"{base_name}_noise_{clip['start_sec']:02d}-{clip['end_sec']:02d}.wav"
        target_path = os.path.join(output_dir, target_filename)
        
        try:
            # Load specific segment
            audio, _ = librosa.load(
                src_path, 
                sr=sample_rate, 
                offset=clip['start_sec'], 
                duration=segment_duration,
                mono=True
            )
            
            # Ensure it's exactly the right length
            target_samples = sample_rate * segment_duration
            if len(audio) < target_samples:
                # Pad with zeros if short (shouldn't happen for 60s files)
                audio = np.pad(audio, (0, target_samples - len(audio)))
            elif len(audio) > target_samples:
                audio = audio[:target_samples]
                
            # Save as WAV
            sf.write(target_path, audio, sample_rate)
            
            # Add to registry
            registry_data.append({
                'noise_id': f"noise_{i:04d}",
                'noise_filename': target_filename,
                'source_filename': clip['filename'],
                'start_sec': clip['start_sec'],
                'end_sec': clip['end_sec'],
                'sample_rate': sample_rate,
                'duration': segment_duration
            })
            
        except Exception as e:
            print(f"Error extracting {clip['filename']} at {clip['start_sec']}s: {e}")
            
    # 5. Save registry
    registry_df = pd.DataFrame(registry_data)
    registry_df.to_csv(registry_path, index=False)
    print(f"Noise harvesting complete. {len(registry_data)} clips saved to {output_dir}")
    print(f"Registry saved to {registry_path}")

if __name__ == "__main__":
    harvest_noise()
