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
- **Perch v2:** Attempted multiple methods to load `perch_v2_cpu`. Local execution fails on Intel Mac due to `vhlo.func_v1` deserialization.
- **Memory Exhaustion (GCP):** Initial parallel run on `n1-highcpu-32` (28.8GB RAM) failed with `Replicas low on memory`. Spawning 32 parallel Perch v2 instances exceeded the available RAM.
- **Resolution:** Switched to **`n1-standard-32`** which provides **120GB of RAM**, successfully accommodating the parallel model instances.
### Next Steps
- [x] Create Exploratory Data Analysis (EDA) notebook for Perch baseline.
- [x] Create Spectrogram EDA notebook for Visual Modeling.
- [x] Create Temporal & Migratory EDA notebook.
- [x] Implement Perch v2 embedding extraction.
- [x] Implement Visual Modeling Approach (EfficientNet + Spectrograms).
- [x] Formulate Multi-Modal Fusion Implementation Plan (`docs/FUSION_MODEL_PLAN.md`).
- [ ] Run Perch v2 embedding extraction on GCP (Job ID 3475193536821854208).
- [ ] Develop Validation Pipeline using `train_soundscapes`.
- [ ] Train a baseline classifier on Perch embeddings.
- [ ] Train a baseline classifier on Spectrograms (Visual Approach).
- [ ] Implement Fusion Model based on the new roadmap.

## [2026-03-11] - Temporal & Migratory Dynamics

### Completed
- **Temporal EDA Notebook:** Created `notebooks/03_temporal_migratory_eda.ipynb`.
    - Identified migratory genera (Hirundo, Tringa, etc.).
    - Visualized seasonal presence of top migratory species in soundscapes.
    - Confirmed the necessity of a "Temporal Branch" in the fusion architecture.

## [2026-03-11] - Data Understanding: Samples vs Soundscapes

### Observations
- **`train_audio` (Samples):** These are curated recordings of specific species. They are "cleaner" and used for core training. However, they don't perfectly represent the test environment.
- **`train_soundscapes`:** These 1-minute recordings from the Pantanal contain multiple species and background noise (wind, rain, insects). These are **critical** for validation because they match the test set distribution.
- **Strategy:** We will use `train_audio` for supervised training and `train_soundscapes` for local validation (Out-of-Distribution check) and potentially for fine-tuning/domain adaptation.

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

## [2026-03-12] - Multi-Model Extraction & Vertex AI Stabilization

### Completed
- **Multi-Model Pipeline:**
    - Established independent extraction workflows for **Perch v1**, **Perch v2**, and **BirdNET**.
    - Created a master deployment script `deploy_all.sh` to launch tasks on Vertex AI `n1-standard-32` instances.
    - Implemented GCS auto-upload for all models to `gs://birdclef-2026-data-birdclef-490003/processed/`.
- **Perch v1 Stabilization:**
    - Created `src/audio/extract_embeddings_v1.py`.
    - Successfully ran locally and on Vertex AI.
    - Resolved race conditions in model downloading during parallel initialization.
- **BirdNET Integration:**
    - Integrated **BirdNET-Analyzer V2.4** (Global 6K species).
    - Created `src/audio/extract_embeddings_birdnet.py`.
    - Resolved model download issues by baking the `.tflite` model directly into the Docker image (`Dockerfile.birdnet`).
    - Verified extraction of 6,522-dimensional feature vectors.
- **BirdNET EDA:**
    - Created `notebooks/04_birdnet_embeddings_eda.ipynb`.
    - Implemented PCA-preconditioned t-SNE for visualizing the high-dimensional BirdNET space.
- **Perch v2 Cloud Optimization:**
    - Upgraded cloud environment to **TensorFlow 2.18.0** to resolve StableHLO/XLA deserialization errors (`attribute code: 22`).
    - Implemented dynamic input signature detection in `src/audio/extract_embeddings_v2.py`.
    - Transitioned to **TFLite** path for Perch v2 to bypass VHLO/SavedModel version mismatches on CPU.

### Trials & Observations
- **XLA Deserialization:** Perch v2 SavedModel exhibits strict forward-compatibility requirements. Vertex AI CPU environments require matching StableHLO versions found in TF 2.17+.
- **TFLite Stability:** TFLite proved significantly more robust than SavedModel for cross-platform (Mac vs. Cloud Linux) consistency.
- **Feature Space Expansion:** Combined feature vector (Perch 1 + Perch 2 + BirdNET + Visual) now exceeds 10,000 dimensions, providing a rich signal for the final classifier.

### Reversions
- **Reverted Local Environment:** Reverted `pyproject.toml` to stable baseline (TF 2.15.0, Numpy < 2.0) after `perch-hoplite` attempt failed due to Intel Mac wheel availability for TF 2.20.0.

## [2026-03-13] - Multi-Modal Fusion Implementation

### Completed
- **Fusion Model Architecture:**
    - Created `src/models/fusion_model.py`.
    - Implemented a 3-branch Keras model:
        - **Branch A (Audio):** Pre-calculated Perch v1 embeddings (1,280-dim).
        - **Branch B (Visual):** EfficientNetB0 backbone for on-the-fly Mel-spectrogram processing.
        - **Branch C (Metadata):** Spatio-Temporal context (Latitude, Longitude, Month, Day of Year).
- **Fusion Data Generator:**
    - Created `src/training/train_fusion.py`.
    - Implemented `FusionDataGenerator` to synchronize spectrogram generation, embedding retrieval, and metadata extraction.
    - Automated temporal feature extraction from iNaturalist URLs (Unix timestamps) and soundscape filenames.
- **Verification:**
    - Successfully ran a test training loop with a small subset of the data (500 samples).
    - Confirmed that accuracy increases and loss decreases across epochs.
    - Verified `embedding_idx` mapping between CSV metadata and `.npz` embedding arrays.
- **Optimization & Cloud Deployment:**
    - Created `src/audio/precalculate_spectrograms.py` for multi-core conversion of audio to 224x224 PNGs.
    - Updated `FusionDataGenerator` to support loading precalculated images, reducing CPU overhead during training by 10-20x.
    - Containerized the fusion training pipeline with `Dockerfile.fusion`.
    - Created `job_config_fusion.yaml` and `deploy_fusion.sh` for Vertex AI `n1-standard-32` deployment.
    - Automated cloud data setup (Kaggle download + GCS embedding sync) in `src/training/train_fusion.py`.

## [2026-03-13] - Perch v2 Migration & Extraction Saga

### Completed
- **Technical Discovery:**
    - Confirmed **Perch v2** SavedModel (default) is CUDA-locked and fails on CPU nodes with `The current platform CPU is not among the platforms required by the module: [CUDA]`.
    - Identified CPU-specific variant via Kaggle public notebooks: `google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1`.
    - Verified that **TensorFlow 2.20.0** is strictly required to deserialize the VHLO bytecode in Perch v2.
    - Confirmed embedding dimension has increased from 1,280 (v1) to **1,536 (v2)**.
- **Environment Stabilization:**
    - Created a "Bootstrap" strategy for Vertex AI to install the heavy TF 2.20 stack at runtime, bypassing local build/space issues.
    - Successfully verified the CPU variant with a "Smoke Test" on a standard Vertex AI node (`perch-v2-smoke-test-20260313-1859`).
- **Robust Extraction Script:**
    - Updated `src/audio/extract_embeddings_v2.py` with automatic `kaggle.json` creation and multi-worker safety fixes.

### Trials & Observations (Final Blockers)
- **Kaggle Auth persistent failure:** Despite injecting `KAGGLE_USERNAME` and `KAGGLE_KEY` and creating `.kaggle/kaggle.json` at runtime, the Vertex AI extraction job (`3712730929374953472`) failed with `401 Unauthorized` during the competition data download.
- **Probable Cause:** The service account identity or the linked Kaggle API key may not have accepted the competition rules, or the CLI requires an interactive login refresh that isn't happening on the headless node.

### Status
- **Halted:** Perch v2 extraction is technically verified (smoke test pass) but blocked by authentication at scale. Using Perch v1 embeddings for now to maintain momentum on the Fusion Model.

### Next Steps
- [ ] Pre-download competition data to a GCS bucket instead of relying on Kaggle CLI at runtime on Vertex.
- [ ] Update Fusion Model `Branch A` to support the 1,536-dim Perch v2 vector if/when extracted.
- [ ] Train the full fusion model on the complete dataset.
- [ ] Implement a "Late Fusion" ensemble for comparison (averaging probabilities).
- [ ] Integrate BirdNET embeddings into the fusion architecture (High-dim branch).
- [ ] Develop a dedicated validation script for `train_soundscapes`.

## [2026-03-14] - Shared Infrastructure: Master Registry

### Completed
- **[SHR-101] Master 234-Species Registry:**
    - Created `src/training/create_master_registry.py`.
    - Extracted 234 unique species IDs from `train.csv` (primary + secondary) and `train_soundscapes_labels.csv`.
    - Verified the count matches exactly 234 as per requirements.
    - Saved alphabetized mapping to `data/processed/species_registry.json`.
    - Updated `docs/IMPLEMENTATION_ROADMAP.md` and `docs/HANDOFF_NOTES.md`.
- **[SHR-102] Dense Feature Scan:**
    - Created `src/audio/extract_embeddings_soundscapes.py`.
    - Implemented high-density feature extraction using Perch v1.
    - Added 1s-stride sliding window (56 overlapping 5-second windows per 60s soundscape file).
    - Output saved as compressed `.npz` files (shape: `(56, 1280)`) for each soundscape.
    - CPU optimized implementation with multiprocessing.
- **[SHR-102-S] Soundscape Spectrogram Scan:**
    - Created `src/audio/precalculate_soundscape_spectrograms.py`.
    - Modified `src/audio/spectrograms.py` to support generating from audio arrays directly.
    - Implemented 1s-stride Mel-spectrogram generation (224x224 RGB PNGs).
    - CPU optimized with multiprocessing.
- **[SHR-103] GroupKFold Registry:**
    - Created `src/training/create_soundscape_kfold_registry.py`.
    - Mapped 56 overlapping 5s windows (1s stride) per soundscape to their labels from `train_soundscapes_labels.csv`.
    - Implemented 5-fold GroupKFold split by `filename` to prevent data leakage.
    - Generated a multi-label encoded registry at `data/processed/soundscape_kfold_registry.csv` with 234 species columns.
    - Verified fold distribution and label density (approx. 4.2 species per window).

### Next Steps
- [ ] Implement **[SHR-104] Noise Harvesting**: Extract "Pure Noise" banks from unlabeled soundscape segments.
