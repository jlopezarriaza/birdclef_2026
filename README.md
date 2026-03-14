# BirdCLEF 2026 - Bird Species Identification in the Pantanal Wetlands

Repository for the [Kaggle BirdCLEF 2026 Competition](https://www.kaggle.com/competitions/birdclef-2026).

## Goal
Develop machine learning models to identify bird species from continuous audio recordings in the Pantanal wetlands of Brazil.

## Competition Details
- **Target:** Identify species (primarily birds) from 5-second windows of audio.
- **Metric:** Macro-averaged ROC-AUC (skipping classes with no true positive labels).
- **Data:** 32 kHz `.ogg` files.
- **Submission:** A CSV file with probabilities for each target species for each 5-second window.

## Usage

### 1. Precalculate Spectrograms (Recommended for Local Training)
To speed up local training by 10-20x, convert the raw audio to PNG spectrograms first:
```bash
PYTHONPATH=. uv run python3 src/audio/precalculate_spectrograms.py --workers 8
```

### 2. Multi-Modal Fusion Model Training
Train the model combining Perch embeddings, Visual spectrograms, and Metadata:
```bash
# Start background training
PYTHONPATH=. uv run python3 src/training/train_fusion.py > training.log 2>&1 &

# Monitor with TensorBoard
uv run tensorboard --logdir logs/
```

### 3. Cloud Deployment (Vertex AI)
To train on Vertex AI using 32 cores:
```bash
# Full training (6-8 hours)
./deploy_fusion.sh

# Smoke test (5 mins)
./smoke_test_fusion.sh
```

## Project Structure
- `data/`: Raw and processed audio data, metadata, and models (ignored by git).
- `notebooks/`: Exploratory Data Analysis (EDA) and prototyping.
- `src/`: Core logic for audio processing, model architecture, training, and inference.
    - `models/fusion_model.py`: 3-branch Fusion Model (Audio + Visual + Metadata).
    - `training/train_fusion.py`: Main training loop and data generator.

## Setup
Using `uv` for package management:
```bash
uv sync
```
