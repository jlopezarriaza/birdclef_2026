# Modeling Strategy Plan - BirdCLEF 2026

## 1. Deep Learning Fusion Model (Heavyweight)
Current 3-branch architecture (EfficientNetB0 + Perch v1 + Metadata). This model is powerful but computationally expensive for 90-minute CPU-only inference.
- **Improvement:** Test EfficientNetB3 or B4 for more complex visual patterns.
- **Improvement:** Switch to Perch v2 (requires version compatibility fixes).

## 2. Lightweight Ensemble (Efficient Alternative)
To meet Kaggle's 90-minute CPU constraint, we will develop a model that skips on-the-fly spectrogram generation during inference.
- **Architecture:** `sklearn` ensemble (Random Forest, XGBoost, or LightGBM).
- **Inputs:** Concatenated vectors of:
    - Pre-calculated Perch Embeddings (1,280-dim).
    - Pre-calculated BirdNet Embeddings (6,522-dim).
    - Geospatial/Temporal Metadata (4-dim).
- **Rationale:** This ensemble only performs matrix/tree operations, which are 10-50x faster on CPU than decoding audio and generating spectrogram images.

## 3. Ensemble Strategy (Stacking)
- **Top Layer:** Meta-classifier (e.g., Logistic Regression or a small Neural Net) that combines predictions from the Deep Fusion model and the Lightweight Ensemble.
- **Goal:** Higher robustness. If the Deep Fusion model is too slow for the full soundscape, it could be used only on "high-confidence" segments identified by the lightweight model.

## 4. Future Experiments
- **Semi-Supervised Learning:** Pseudo-labeling the 1-minute `train_soundscapes` using a preliminary model and adding high-confidence samples back into training.
- **Multitask Learning:** Add a head to predict "Site ID" or "Habitat" (from metadata) to help regularize the primary species classification.
- **Transformer Branch:** Experiment with an AST (Audio Spectrogram Transformer) in place of EfficientNet.
