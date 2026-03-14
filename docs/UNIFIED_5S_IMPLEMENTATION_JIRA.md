# JIRA Epic: UNIFIED_5S_PIPELINE

**Goal:** Transform all training data into high-quality 5-second samples with real background context.

---

## [UNI-101] Pantanal Background Noise Harvesting
- **Type:** Task
- **Description:** Identify 5-second windows in `train_soundscapes` that have NO labels. Extract these as "Pure Noise" samples.
- **Purpose:** Used for padding short clips and as a "Negative" class to reduce false positives.
- **Deliverable:** `data/processed/pantanal_noise_bank.npy`

---

## [UNI-102] Short-Clip Contextual Stitcher
- **Type:** Task
- **Description:** For all curated recordings in `train_audio` with duration < 5s:
    - Randomly select a noise sample from [UNI-101].
    - "Stitch" the curated bird call into a random position within the 5s noise window.
- **Deliverable:** `src/audio/contextual_stitcher.py`

---

## [UNI-103] Dense Soundscape Feature Extraction (1s Stride)
- **Type:** Task
- **Description:** Run Perch/BirdNET extraction on `train_soundscapes` using a 5s window and 1s stride (56 windows per file).
- **Technical Note:** This is the "Audio-level stitching" described in the strategy.
- **Deliverable:** `data/processed/ss_dense_embeddings.npz`

---

## [UNI-104] Multi-Label Master Registry
- **Type:** Task
- **Description:** Create a CSV that maps every unified 5s sample (from UNI-102 and UNI-103) to a multi-hot vector of the 234 species IDs.
- **Deliverable:** `data/processed/unified_v2_metadata.csv`

---

## [UNI-105] Validation: Stitched-Clip EDA
- **Type:** Analysis
- **Description:** Compare spectrograms of "Zero-Padded" vs "Context-Stitched" samples. Verify that the transition between bird call and noise bank is smooth and doesn't introduce artifacts.
- **Deliverable:** `notebooks/07_contextual_stitching_eda.ipynb`

---

## [UNI-106] Unified 5s Model Training (V2)
- **Type:** Story
- **Description:** Train the Fusion Model using the entire unified 5s dataset.
- **Architecture:** Sigmoid output + BinaryCrossentropy.
- **Validation:** GroupKFold based on original file IDs.
- **Deliverable:** `models/fusion_model_v2_unified.keras`
