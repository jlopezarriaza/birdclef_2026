# JIRA Epic: SOUNDSCAPE_DATA_REFINEMENT

**Goal:** Bridge the domain gap between curated 5s audio and 1-minute noisy soundscapes by generating high-quality, verified 5s training samples using Model-Guided and Energy-Based selection.

---

## [BIR-101] Soundscape Multi-Resolution Feature Extraction
- **Type:** Task
- **Priority:** High
- **Description:** Run full-minute feature extraction on `train_soundscapes` with a high temporal resolution (1s stride) to enable downstream guided selection.
- **Deliverables:** 
    - Script: `src/audio/extract_soundscape_features.py`
    - Output: `data/processed/ss_features_1s_stride.npz` (Perch + BirdNet)
- **Technical Note:** This ensures we have features for every possible 5s window starting at `t=0, 1, 2...55`.

---

## [BIR-102] Baseline: Energy-Based Peak Detection
- **Type:** Task
- **Priority:** Medium
- **Description:** Implement a script to identify the loudest 5s window within each labeled interval using RMS energy. This serves as our "Fallback" and "Lower Bound" for data quality.
- **Deliverables:**
    - Script: `src/audio/detect_energy_peaks.py`
    - Output: `data/processed/energy_peak_metadata.csv` (file, species, offset, energy)
- **Validation:** Must handle overlapping labels by centering the window on the highest energy sum.

---

## [BIR-103] Advanced: Mini-Model Guided Labeling
- **Type:** Task
- **Priority:** High
- **Description:** Use the `fusion_model_v1` (currently training) to scan the soundscapes. For each species labeled in the grid, find the 5s offset where the model's confidence is highest.
- **Deliverables:**
    - Script: `src/inference/generate_guided_labels.py`
    - Output: `data/processed/guided_slice_metadata.csv` (file, species, offset, confidence)
- **Refinement:** Use BirdNET as a "Consultant" for the 28 species the mini-model has never seen.

---

## [BIR-104] EDA: Slicing Strategy Validation & Manual Audit
- **Type:** Task/Analysis
- **Priority:** Critical
- **Description:** A notebook to compare Grid vs. Energy vs. Model-Guided slicing. 
- **Tasks:**
    - Plot "Energy Curves" vs "Confidence Curves" for sample soundscapes.
    - Generate side-by-side spectrograms of the "best window" selected by each method.
    - **Manual Check:** Export a random sample of 50 windows where Energy and Model disagree for manual audio review.
- **Deliverables:** `notebooks/06_soundscape_slicing_eda.ipynb`

---

## [BIR-105] Unified 5s Resolution Training Assembly
- **Type:** Story
- **Priority:** High
- **Description:** Final data preparation. Instead of disparate 1s and 5s logic, we "standardize" all data (Short Audio + Soundscape Slices) into a single 5s resolution pipeline.
- **Alternative Implementation (Stitching):** 
    - If specific short audio recordings are < 5s, they will be padded or "stitched" with Pantanal background noise to maintain the 5s resolution.
    - All model inputs across all experiments will be strictly 5s @ 32kHz.
- **Deliverables:** 
    - Script: `src/training/assemble_v2_dataset.py`
    - Metadata: `data/processed/train_v2_master.csv` (Includes `group_id` for GroupKFold).

---

## [BIR-106] Implementation of Multi-Label GroupKFold
- **Type:** Task
- **Priority:** High
- **Description:** Ensure that the 1-minute files are never split across train/val folds.
- **Constraints:** All 5s slices from `BC2026_Train_0039` must be either 100% Training or 100% Validation.
