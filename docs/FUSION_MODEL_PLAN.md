# Proposed Implementation: Multi-Modal Fusion Model

This document outlines the architecture and strategy for building a "Breakthrough" model that leverages both Google's **Perch v2** (Bioacoustic Embeddings) and **EfficientNet** (Visual Spectrogram Features) in a unified classification system.

## 1. Architectural Concept
The goal is to combine the "Global Bioacoustic Knowledge" of Perch with the "Localized Pattern Recognition" of a vision model trained on the specific dataset.

### Fusion Architecture
```text
Audio Input (5s @ 32kHz)
│
├── Branch A: Global Bioacoustics (Perch v2)
│   └── Output: 1,280-dim Embedding Vector
│
├── Branch B: Visual Patterns (EfficientNet)
│   └── Output: 1,280-dim (B0) or 1,536-dim (B3) Global Average Pooled Vector
│
└── Fusion Layer (Concatenation)
    └── Feature Vector (~2,560 to 2,816 dimensions)
        │
        ├── Dense Layer (1,024 units + Dropout)
        ├── Dense Layer (512 units + Dropout)
        └── Output Layer (Softmax for Bird Species)
```

## 2. Implementation Strategy

### Step 1: Feature Alignment
Ensure both branches process the **exact same 5-second window**.
- Current scripts already standardize to 32kHz and 160,000 samples.
- The `SpectrogramGenerator` and `Perch` inference must be synchronized.

### Step 2: Training Workflow (Phase 1 - Two-Stage)
1. **Pre-calculate Perch Embeddings:** (Already implemented).
2. **Train Visual Model:** (Already implemented).
3. **Freeze Backbones:** Extract features from both and train only the Dense Fusion Head.
   - This is the fastest way to verify if fusion improves AUC over standalone models.

### Step 3: Training Workflow (Phase 2 - End-to-End)
Create a unified model where only the EfficientNet backbone is fine-tuned while the Perch branch remains fixed (as a feature extractor).
- **Tooling:** Use `tf.keras.layers.Concatenate`.
- **Custom Generator:** A `Sequence` that returns `[Spectrogram, Perch_Embedding]` as input pairs.

## 3. Anticipated Breakthroughs
- **Noise Robustness:** Perch is trained on massive datasets and may handle background noise better, while the visual model focuses on specific Pantanal call textures.
- **Ensemble-in-a-Model:** Concatenation at the feature level is often more powerful than simple averaging of final probabilities (Late Fusion).

## 4. Potential Challenges
- **Memory Limits:** Running two heavy backbones simultaneously during inference might hit the Kaggle 90-minute limit or memory constraints.
- **Optimization:** Balancing the learning rates so the Fusion head doesn't overpower the EfficientNet fine-tuning.

## 5. Next Steps for Developer
1. Create `src/models/fusion_model.py`.
2. Implement a `FusionDataGenerator` that loads pre-calculated Perch `.npz` files and generates on-the-fly spectrograms.
3. Train the head on the combined feature set.
