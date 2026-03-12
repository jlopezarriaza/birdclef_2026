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
- **Submission via Notebook:** All submissions must be made through a Kaggle Notebook.
- **Runtime Limits:**
    - **CPU Notebook:** $\le$ 90 minutes.
    - **GPU Notebook:** Effectively **disabled** (granted only 1 minute of runtime).
- **Internet Access:** Must be **disabled** during inference/submission run.
- **External Data:** Freely and publicly available external data is allowed, including pre-trained models.
- **Submission File:** Output must be exactly named `submission.csv`.
- **No Hand Labeling:** Strictly prohibited on validation or test datasets.

## Data Details
- **Source:** Network of ~1,000 acoustic recorders in the Pantanal.
- **Training Audio:**
    - Thousands of short recordings of individual species (primary/secondary labels).
    - Metadata includes latitude, longitude, author, and recording quality.
- **Test Audio:** 
    - Continuous soundscapes (typically provided as 1-minute or 10-minute files in the hidden test set).
    - Hidden test set only accessible during the submission scoring process.
- **Format:** 32 kHz `.ogg` files.

## Links
- [Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2026)
- [Perch Model (Google)](https://www.kaggle.com/models/google/bird-vocalization-classifier)
- [Perch GitHub](https://github.com/google-research/perch)
