# Data Strategy Plan - BirdCLEF 2026

## 1. Data Sources
- **Primary (Clean):** `train_audio/` (35,549 curated short recordings).
- **Secondary (Noisy):** `train_soundscapes/` (1-minute continuous field recordings).
    - **Soundscape Slicing:** To utilize this data for training, we will slice 1-minute files into twelve 5-second windows aligned with `train_soundscapes_labels.csv`.
- **Master Species List:** We will maintain a fixed registry of all **234 unique species IDs** found across both `train.csv` and `train_soundscapes_labels.csv`.

## 2. Multi-Label Engineering
- **Secondary Labels:** The `secondary_labels` field in `train.csv` will be converted from a string list to a multi-hot encoded vector.
- **Combined Targets:** Each training sample will have a target vector where:
    - `primary_label` = 1.0
    - `secondary_labels` = 1.0 (or a weighted value like 0.5 to indicate background presence)
- **Audio Embeddings:** 
    - **Perch v1 (1,280-dim):** Fast, stable baseline.
    - **Perch v2 (1,536-dim):** Higher quality, requires TFLite or TF 2.17+ in cloud.
    - **BirdNet (6,522-dim):** Broad taxonomic coverage, excellent for non-bird sounds.
- **Visual Features:** 
    - **Mel-Spectrograms:** 224x224 PNGs (50Hz - 14kHz).
- **Metadata:**
    - **Geospatial:** Latitude/Longitude (standardized).
    - **Temporal:** Cyclic encoding (Sin/Cos) for Month and Day of Year.
## 3. Vocalization Centering (Model-Guided Refinement)
To avoid training on "mostly noise" windows where the bird is not the loudest sound (e.g., wind, insects), we will use **Model-Guided Slicing**:
- **Strategy:** We will use the model trained on curated `train_audio` to scan the 1-minute soundscapes.
- **Selection:** For a labeled species interval, we will select the 5-second window (within a ±2s buffer) that yields the **highest confidence score** from the model for that specific species.
- **Fallback:** For the 28 species missing from curated audio, we will use **BirdNET-guided** peak selection (BirdNET feature energy in the relevant frequency bands).
- **Goal:** This ensures the 5-second training sample actually contains the distinctive vocalization features of the target bird, even in low Signal-to-Noise Ratio (SNR) environments.

## 4. Data Augmentation (Future Experiments)
...
- **Time-Domain:** Pitch shifting, time stretching, and background noise injection (using Pantanal-specific noise from soundscapes).
- **Frequency-Domain:** SpecAugment (time/frequency masking) on spectrograms.
- **Mixup:** Blending two audio files to simulate simultaneous vocalizations.

## 4. Areas for Improvement
- **Class Imbalance:** Some species have <10 samples while others have hundreds. Use oversampling or synthetic minority oversampling (SMOTE) on embeddings.
- **Soundscape Alignment:** Create a "Noise Profile" from soundscape intervals where no labels are present to use as a "Negative" class.
