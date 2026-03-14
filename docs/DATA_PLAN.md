# Data Strategy Plan - BirdCLEF 2026

## 1. Data Sources
- **Primary:** `train_audio/` (35,549 curated short recordings).
- **Validation/OOD:** `train_soundscapes/` (1-minute continuous field recordings).
- **External:** Potential use of Xeno-canto if specific species are under-represented.

## 2. Feature Engineering
- **Audio Embeddings:** 
    - **Perch v1 (1,280-dim):** Fast, stable baseline.
    - **Perch v2 (1,536-dim):** Higher quality, requires TFLite or TF 2.17+ in cloud.
    - **BirdNet (6,522-dim):** Broad taxonomic coverage, excellent for non-bird sounds.
- **Visual Features:** 
    - **Mel-Spectrograms:** 224x224 PNGs (50Hz - 14kHz).
- **Metadata:**
    - **Geospatial:** Latitude/Longitude (standardized).
    - **Temporal:** Cyclic encoding (Sin/Cos) for Month and Day of Year.

## 3. Data Augmentation (Future Experiments)
- **Time-Domain:** Pitch shifting, time stretching, and background noise injection (using Pantanal-specific noise from soundscapes).
- **Frequency-Domain:** SpecAugment (time/frequency masking) on spectrograms.
- **Mixup:** Blending two audio files to simulate simultaneous vocalizations.

## 4. Areas for Improvement
- **Class Imbalance:** Some species have <10 samples while others have hundreds. Use oversampling or synthetic minority oversampling (SMOTE) on embeddings.
- **Soundscape Alignment:** Create a "Noise Profile" from soundscape intervals where no labels are present to use as a "Negative" class.
