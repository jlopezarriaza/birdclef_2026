# Training & Validation Plan - BirdCLEF 2026

## 1. Validation Strategy
- **Phase 1:** Stratified 5-Fold Cross-Validation on `train_audio` (short recordings).
- **Phase 2:** Out-of-Distribution (OOD) Validation on `train_soundscapes` (the "real world" test).
- **Metric:** **Macro-averaged ROC-AUC**. 
    - *Nuance:* The competition metric skips classes that have no true positive labels in the evaluation slice. 
    - *Action:* In future training scripts, we must use `tf.keras.metrics.AUC(name='macro_auc', multi_class='ovr')` to better align with the leaderboard.
- **Metric Tracking:** Track both "Overall" and "Per-Site" performance to ensure the model generalizes across the Pantanal.

## 2. Training Workflow
1. **Multi-Hot Label Encoder:** Create a persistent mapping of all 234 species IDs. For each sample, the target is a vector of 234 binary values representing the union of primary and secondary labels.
2. **Loss Function:** **Binary Cross-Entropy (BCE)**.
3. **Data Balancing:** Since soundscapes contain the 28 "missing" species, training must include the sliced soundscape windows to ensure the model sees all 234 classes.

## 3. Regularization & Stabilization
- **Learning Rate Scheduler:** Use `ReduceLROnPlateau` (already implemented) and potentially a `CosineAnnealing` scheduler for smoother convergence.
- **Early Stopping:** Monitor validation AUC rather than accuracy to stay aligned with competition metrics.
- **Weight Decay:** Apply L2 regularization to dense fusion layers.

## 4. Future Experiments
- **Knowledge Distillation:** Use the Deep Fusion model (Teacher) to train a smaller, faster model (Student) on the soundscape data.
- **Ensemble Averaging:** Compute the geometric mean of probabilities across all 5 folds.
- **Inference Speed Benchmarking:** Profile the end-to-end inference (audio load -> feature extract -> ensemble prediction) to ensure we hit the 90-minute limit.
