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
- **Perch v2:** Attempted multiple methods to load `perch_v2_cpu` (SavedModel, hub.load, signature-based). Encountered persistent `InvalidArgumentError` related to `XlaCallModule` and `vhlo.func_v1` deserialization. This is a known issue when loading models exported with newer StableHLO versions into older TensorFlow environments (TF 2.15 on Intel Mac). 
- **Fallback to Perch v1:** Reverted to **Perch v1** (`bird-vocalization-classifier`) which is verified stable and efficient on this hardware. We will use v1 for initial EDA and embedding generation to maintain project momentum.

### Next Steps
- [x] Create Exploratory Data Analysis (EDA) notebook for Perch baseline.
- [x] Implement embedding extraction script (`src/audio/extract_embeddings.py`).
- [ ] Run embedding extraction for the full training set (Target: `data/processed/perch_v1_embeddings.npz`).
- [ ] Setup validation strategy (Cross-Validation) based on `train.csv`.

## [2026-03-11] - Data Processing Foundation

### Completed
- **Embedding Extraction:** 
    - Created `src/audio/extract_embeddings.py` for batch processing of training audio.
    - Implemented `docs/DATA_PIPELINE.md` to document the extraction process and artifact usage.
    - Script handles 32kHz resampling, 5s windowing, and Perch v1 inference.
