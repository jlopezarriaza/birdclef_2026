# Training & Validation Plan - BirdCLEF 2026

## 1. Validation Strategy
- **Phase 1:** Stratified 5-Fold Cross-Validation on `train_audio` (short recordings).
- **Phase 2:** Out-of-Distribution (OOD) Validation on `train_soundscapes` (the "real world" test).
- **Metric:** Macro-averaged ROC-AUC, tracking both "Overall" and "Per-Site" performance to ensure the model generalizes across the Pantanal.

## 2. Training Workflow
1. **Fold Training:** Split the 35k recordings into 5 folds, ensuring no species is missing from any validation set.
2. **Loss Function:** Categorical Cross-Entropy (CCE) or Binary Cross-Entropy (BCE) for multi-label support (handling `secondary_labels`).
3. **Threshold Optimization:** Since we submit probabilities (0.0 to 1.0), we will optimize the decision threshold for each species to maximize F1-score on validation data.

## 3. Regularization & Stabilization
- **Learning Rate Scheduler:** Use `ReduceLROnPlateau` (already implemented) and potentially a `CosineAnnealing` scheduler for smoother convergence.
- **Early Stopping:** Monitor validation AUC rather than accuracy to stay aligned with competition metrics.
- **Weight Decay:** Apply L2 regularization to dense fusion layers.

## 4. Future Experiments
- **Knowledge Distillation:** Use the Deep Fusion model (Teacher) to train a smaller, faster model (Student) on the soundscape data.
- **Ensemble Averaging:** Compute the geometric mean of probabilities across all 5 folds.
- **Inference Speed Benchmarking:** Profile the end-to-end inference (audio load -> feature extract -> ensemble prediction) to ensure we hit the 90-minute limit.
