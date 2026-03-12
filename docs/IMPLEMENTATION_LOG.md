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
- [x] Create Spectrogram EDA notebook for Visual Modeling.
- [x] Implement Perch v2 embedding extraction.
- [x] Implement Visual Modeling Approach (EfficientNet + Spectrograms).
- [x] Formulate Multi-Modal Fusion Implementation Plan (`docs/FUSION_MODEL_PLAN.md`).
- [ ] Run Perch v2 embedding extraction on GCP (Job ID 3475193536821854208).
- [ ] Train a baseline classifier on Perch embeddings.
- [ ] Train a baseline classifier on Spectrograms (Visual Approach).
- [ ] Implement Fusion Model based on the new roadmap.

## [2026-03-11] - Visual EDA

### Completed
- **Spectrogram EDA Notebook:** Created `notebooks/02_spectrogram_eda.ipynb`.
    - Integrated `SpectrogramGenerator` for visual verification.
    - Added quality-based comparisons (Rating analysis).
    - Verified consistency of visual patterns within species.

## [2026-03-11] - Future Roadmap: Multi-Modal Fusion

### Completed
- **Fusion Strategy:**
    - Authored `docs/FUSION_MODEL_PLAN.md`.
    - Defined architecture for concatenating Perch embeddings with EfficientNet visual features.
    - Outlined a two-stage training strategy (feature alignment followed by fine-tuning).
    - Mapped out expected advantages in noise robustness and classification accuracy.


## [2026-03-11] - Optimization & Hybrid Pipeline

### Completed
- **Parallel Extraction:** 
    - Rewrote `src/audio/extract_embeddings.py` to use a `multiprocessing` Pool.
    - Optimized for 32-core cloud instances, reducing estimated runtime from ~29 hours to ~1 hour.
- **Hybrid Hardware Awareness:**
    - Script now auto-detects CPU vs GPU environments.
    - Implemented single-process mode for GPUs to prevent VRAM OOM while keeping high-concurrency mode for CPUs.
- **GCS Integration:**
    - Finalized automatic upload logic to move results from Vertex AI ephemeral storage to `gs://birdclef-2026-data-birdclef-490003`.
- **Documentation:**
    - Updated `CLOUD_COMPUTE.md` and `DATA_PIPELINE.md` with parallelization and GPU-ready instructions.
