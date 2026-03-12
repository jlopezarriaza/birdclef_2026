# Implementation & Experiment Log

This log tracks the development progress, experiments, and architectural decisions for the BirdCLEF 2026 competition.

---

## [2026-03-11] - Project Kickoff & Baseline

### Completed
- **Environment Setup:** 
    - Initialized project with `uv`.
    - Configured Python 3.11 environment.
    - Resolved Intel macOS compatibility issues for TensorFlow (`2.15.0`) and NumPy (`<2.0`).
- **Data Acquisition:** 
    - Downloaded 15GB of raw competition data from Kaggle.
    - Unzipped and organized into `data/raw/`.
- **Baseline Implementation:** 
    - Implemented `src/inference/perch_baseline.py`.
    - Verified **Google Perch v1** SavedModel on CPU.
    - Successfully generated 1,280-dim embeddings and species predictions from raw `.ogg` files.
- **Documentation:**
    - Created `GEMINI.md` for instructional context.
    - Aggregated competition rules and code requirements in `docs/COMPETITION_INFO.md`.

### Trials & Observations
- **Perch v2:** Attempted to use `perch_v2_cpu`, but encountered XLA/VHLO deserialization errors on TensorFlow 2.15 (standard for Intel Mac wheels). Reverted to **Perch v1** which works natively and reliably on this hardware.
- **Inference Speed:** Perch v1 inference on a single 5-second window is fast on CPU, which is promising for the 90-minute Kaggle limit.

### Next Steps
- [x] Create Exploratory Data Analysis (EDA) notebook for Perch baseline.
- [ ] Pre-calculate embeddings for the full training set to enable fast training of custom heads.
- [ ] Setup validation strategy (Cross-Validation) based on `train.csv`.

## [2026-03-11] - Baseline EDA

### Completed
- **EDA Notebook:** Created `notebooks/01_perch_baseline_eda.ipynb`.
    - Integrated audio visualization (waveform/spectrogram).
    - Implemented Perch inference loop for batch processing.
    - Added PCA projection for embedding visualization.
