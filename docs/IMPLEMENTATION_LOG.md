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

- [x] Run Perch v2 embedding extraction on GCP (Launched: Job ID 3475193536821854208).
- [ ] Download extracted embeddings from GCS bucket.
- [ ] Setup validation strategy (Cross-Validation) based on `train.csv`.

## [2026-03-11] - Cloud & Perch v2 Upgrade

### Completed
- **Perch v2 Upgrade:** 
    - Updated `src/audio/extract_embeddings.py` and `src/inference/perch_baseline.py` to use **Perch v2**.
    - Model now produces higher-quality embeddings based on EfficientNet-B3.
- **Dockerization:**
    - Created `Dockerfile` based on `tensorflow/tensorflow:2.16.1-gpu`.
    - Integrated `uv` for container builds.
- **Vertex AI Deployment:**
    - Successfully launched serverless extraction job on a 32-core high-CPU instance (`n1-highcpu-32`).
    - Configured automatic GCS upload of results to `gs://birdclef-2026-data-birdclef-490003`.
