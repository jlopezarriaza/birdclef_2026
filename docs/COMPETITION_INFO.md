# BirdCLEF 2026 - Wildlife Detection in the Pantanal

## Overview
- **Goal:** Develop machine learning models to identify wildlife species (primarily birds) from continuous audio recordings collected via **Passive Acoustic Monitoring (PAM)**.
- **Location:** The Pantanal wetlands of Brazil, one of the world's most diverse and threatened ecosystems with over 650 bird species.
- **Host:** Cornell Lab of Ornithology and LifeCLEF.
- **Challenge:** Automated biodiversity monitoring in "messy" field data across different habitats and seasons. Models must handle noisy environments and simultaneous vocalizations.

## Timeline (2026)
- **Start Date:** March 11
- **Entry/Merger Deadline:** May 27
- **Final Submission Deadline:** June 3
- **Working Note Deadline:** June 17 (for CLEF conference proceedings)

## Evaluation
- **Metric:** Macro-averaged **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve).
- **Nuance:** The metric **skips classes that have no true positive labels** in the test set evaluation slice. This prevents scores from being skewed by species absence in particular soundscapes.
- **Submission Format:** A CSV file containing probability scores (0.0 to 1.0) for every species for every **5-second window** of audio.
- **Row ID Format:** `soundscape_[file_id]_[end_time]` (e.g., `soundscape_12345_5`).

## Code Requirements
- **Training (Off-platform):** You **are allowed** to use external cloud compute (e.g., AWS, GCP, local GPUs) to train your models and process data. There are no restrictions on the compute used for training.
- **Submission via Notebook:** All final submissions (inference) must be made through a Kaggle Notebook.
- **Runtime Limits (Inference):**
    - **CPU Notebook:** $\le$ 90 minutes.
    - **GPU Notebook:** Effectively **disabled** (granted only 1 minute of runtime). Models must be optimized for CPU.
- **Internet Access:** Must be **disabled** during the inference/submission run.
- **External Data:** Freely and publicly available external data is allowed, including pre-trained models. If you train on private data, you must make that dataset public on Kaggle before the deadline.
- **Submission File:** Output must be exactly named `submission.csv`.
- **No Hand Labeling:** Strictly prohibited on validation or test datasets.

## Data Details
- **Source:** Network of ~1,000 acoustic recorders in the Pantanal, plus crowdsourced recordings (Xeno-canto, iNaturalist).
- **Target Classes:** 234 species/sonotypes across Class Aves, Amphibia, Mammalia, Reptilia, and Insecta.

### Training Data
1. **`train_audio/` (Short Recordings):**
    - Thousands of short recordings of individual species.
    - Labels: Found in `train.csv` (`primary_label`, `secondary_labels`).
    - Metadata: Includes `latitude`, `longitude`, `rating` (1-5), and `author`.
    - Purpose: Primary source for learning species-specific features.

2. **`train_soundscapes/` (Long Recordings):**
    - 1-minute continuous recordings from the Pantanal.
    - Labels: Found in `train_soundscapes_labels.csv`.
    - Format: Provided for **5-second windows** (`start`, `end`).
    - Purpose: Realistic evaluation and domain adaptation (handling noise, overlapping calls).

### Metadata Files
- **`train.csv`**: Main metadata for short recordings.
- **`train_soundscapes_labels.csv`**: Labels for the 1-minute recordings.
- **`taxonomy.csv`**: Mapping of codes to scientific names and higher taxonomic ranks.
- **`recording_location.txt`**: High-level geographic info for the Pantanal sites.
- **`sample_submission.csv`**: Template for the required output format.

### Format
- **Audio:** 32 kHz `.ogg` files.
- **Submission:** Probability scores for each of the 234 target species in 5-second windows.

## Links
- [Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2026)
- [Perch Model (Google)](https://www.kaggle.com/models/google/bird-vocalization-classifier)
- [Perch GitHub](https://github.com/google-research/perch)
