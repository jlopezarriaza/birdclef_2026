# BirdCLEF 2026 - Bird Species Identification in the Pantanal Wetlands

Repository for the [Kaggle BirdCLEF 2026 Competition](https://www.kaggle.com/competitions/birdclef-2026).

## Goal
Develop machine learning models to identify bird species from continuous audio recordings in the Pantanal wetlands of Brazil.

## Competition Details
- **Target:** Identify species (primarily birds) from 5-second windows of audio.
- **Metric:** Macro-averaged ROC-AUC (skipping classes with no true positive labels).
- **Data:** 32 kHz `.ogg` files.
- **Submission:** A CSV file with probabilities for each target species for each 5-second window.

## Project Structure
- `data/`: Raw and processed audio data, metadata, and models (ignored by git).
- `notebooks/`: Exploratory Data Analysis (EDA) and prototyping.
- `src/`: Core logic for audio processing, model architecture, training, and inference.

## Setup
Using `uv` for package management:
```bash
uv sync
```
