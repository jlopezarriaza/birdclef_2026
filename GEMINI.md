# BirdCLEF 2026 - Gemini Context

## Project Overview
This repository is dedicated to the [Kaggle BirdCLEF 2026 Competition](https://www.kaggle.com/competitions/birdclef-2026), focusing on identifying bird species in the Pantanal wetlands of Brazil using passive acoustic monitoring.

### Main Technologies
- **Language:** Python 3.11
- **Package Manager:** `uv`
- **Machine Learning:** TensorFlow (for Perch baseline), PyTorch (target for custom models)
- **Audio Processing:** `librosa`, `soundfile`
- **Baseline Model:** Google's **Perch** (Bird Vocalization Classifier), specifically the v1 SavedModel optimized for CPU inference on Intel Mac.

### Architecture
- `data/`: Contains raw competition data (audio, metadata) and processed artifacts.
- `src/`: Core source code.
    - `audio/`: Logic for audio loading and preprocessing.
    - `inference/`: Scripts for running models on new data.
    - `models/`: Custom model architectures.
    - `training/`: Training loops and cross-validation logic.
- `notebooks/`: Prototyping and EDA.
- `docs/`: Repository documentation and implementation logs.
    - `COMPETITION_INFO.md`: Rules and evaluation details.
    - `IMPLEMENTATION_LOG.md`: Running record of progress.
    - `DATA_PIPELINE.md`: Documentation for processed data artifacts (e.g., embeddings).
    - `CLOUD_COMPUTE.md`: Guide for running jobs on Vertex AI and GCP.
    - `FUSION_MODEL_PLAN.md`: Strategic roadmap for Multi-Modal (Perch + Visual) fusion.

## Building and Running

### Environment Setup
The project uses `uv` for dependency management. To sync the environment:
```bash
uv sync
```

### Running the Baseline
To run the Perch v1 baseline inference on a sample file:
```bash
# Ensure Kaggle credentials are set in the environment
export KAGGLE_USERNAME="your_username"
export KAGGLE_API_TOKEN="your_token"

uv run python3 src/inference/perch_baseline.py
```

### Data Management
The dataset is 15GB. It is downloaded via the Kaggle CLI:
```bash
uv run kaggle competitions download birdclef-2026 -p data/raw
```

## Development Conventions

### Hardware Constraints
- **Target Platform:** Intel macOS (x86_64).
- **GPU:** None. All models and processing must be optimized for **CPU**.
- **TensorFlow Version:** Pinned to `2.15.0` for compatibility with Intel Mac wheels.
- **PyTorch Version:** Pinned to `2.2.2`.

### Coding Style
- Follow PEP 8 standards.
- Use `uv run` for executing scripts to ensure the virtual environment is used.
- For audio, default to **32 kHz** sampling rate as required by the competition and Perch model.

### Submission Requirements
- Inference must run within 90 minutes on a CPU-only Kaggle environment.
- Final output must be a `submission.csv` with probability scores for each species in 5-second windows.
