# Strategy: Lightweight Ensemble (Inference Optimization)

**Objective:** Ensure the model meets the 90-minute CPU inference constraint by providing a fast, non-visual alternative/companion to the deep fusion model.

## 1. High-Level Concept
While deep multi-modal models (EfficientNet) are powerful, they are computationally expensive on CPU due to spectrogram generation. This strategy focuses on a "Featurized" approach that uses pre-calculated embeddings and metadata without on-the-fly image processing.

## 2. Core Pillars
### A. The "Sklearn" Ensemble
- **Architecture:** Gradient Boosted Trees (XGBoost, LightGBM) or Random Forest.
- **Inputs:** Concatenated vectors of:
    - Perch v2 Embeddings (1,536-dim)
    - BirdNET Embeddings (6,522-dim)
    - Metadata (Lat/Lon/Month/Day)
- **Benefit:** Matrix and Tree operations are 10-50x faster on CPU than decoding audio and generating spectrograms.

### B. Meta-Classifier (Stacking)
- Use a small neural net or Logistic Regression to combine the probabilities from the **Deep Fusion Model** and the **Lightweight Ensemble**.
- This handles "Disagreement" between the models and improves overall ROC-AUC.

### C. Tiered Inference
- **Logic:** Use the Lightweight Ensemble to scan the full soundscape first.
- **Trigger:** Only run the "Heavyweight" Deep Fusion model on 5s segments where the lightweight model is uncertain or detects high-interest species.
- **Benefit:** Maximizes accuracy while staying safely within the 90-minute limit.
