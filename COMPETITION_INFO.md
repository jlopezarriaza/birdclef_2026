# BirdCLEF 2026 Competition Information

## Overview
- **Goal:** Identify bird species in 5-second windows of continuous audio.
- **Location:** Pantanal wetlands of Brazil.
- **Timeline:** March 11 - June 3, 2026.
- **Prizes:** $50,000 total.

## Evaluation
- **Metric:** Macro-averaged ROC-AUC (skipping classes with no true positive labels).
- **Submission:** Probability scores for each species in each 5-second window.
- **Constraints:** CPU $\le$ 90 mins, Internet disabled for submission.

## Data
- **Format:** 32 kHz `.ogg` files.
- **Metadata:** `train_metadata.csv` (primary/secondary labels, coordinates).
- **Taxonomy:** `taxonomy.csv` (species mappings).

## Links
- [Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2026)
- [Perch Model (Google)](https://www.kaggle.com/models/google/bird-vocalization-classifier)
- [Perch GitHub](https://github.com/google-research/perch)
