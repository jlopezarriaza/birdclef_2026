import os
import numpy as np
import librosa
import cv2

class SpectrogramGenerator:
    """
    Generates Mel-spectrograms from audio files using parameters optimized 
    for bird vocalization identification (Visual Bird AI approach).
    """
    def __init__(self, 
                 sr=32000, 
                 n_mels=128, 
                 fmin=50, 
                 fmax=14000, 
                 n_fft=2048, 
                 hop_length=512, 
                 duration=5,
                 img_size=224):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.img_size = img_size
        self.samples = sr * duration

    def generate(self, audio_path):
        """
        Loads audio and returns a normalized RGB spectrogram image.
        """
        try:
            # 1. Load and Pad/Crop
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            if len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)))
            
            # 2. Generate Mel Spectrogram
            melspec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sr, 
                n_mels=self.n_mels, 
                fmin=self.fmin, 
                fmax=self.fmax, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # 3. Power to DB
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            
            # 4. Normalize to [0, 255] for image format
            melspec_norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-6)
            melspec_norm = (melspec_norm * 255).astype(np.uint8)
            
            # 5. Flip vertically (low frequencies at bottom)
            melspec_norm = np.flip(melspec_norm, axis=0)
            
            # 6. Resize to target size (Square for EfficientNet)
            if self.img_size:
                melspec_norm = cv2.resize(melspec_norm, (self.img_size, self.img_size))
            
            # 7. Convert to RGB (3 identical channels for pre-trained CV models)
            img_rgb = cv2.merge([melspec_norm, melspec_norm, melspec_norm])
            
            return img_rgb
        except Exception as e:
            print(f"Error generating spectrogram for {audio_path}: {e}")
            return None

def save_spectrogram(img, output_path):
    """Saves the spectrogram as an image file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
