# Implementation Roadmap: Soundscape Data Refinement

This roadmap combines both the **5s Unified** and **1s Stitching** strategies into an optimized, non-redundant execution plan.

---

## Phase 1: Shared Infrastructure (Immediate Priority)
These tasks provide the foundation for both strategies.

- **[SHR-101] Master 234-Species Registry:** Create a fixed JSON mapping for all competition species.
- **[SHR-102] Dense Feature Scan:** Run Perch/BirdNET extraction on all soundscapes with 1s stride (56 windows/file).
- **[SHR-103] GroupKFold Registry:** Map soundscape slices to parent file IDs to prevent data leakage.
- **[SHR-104] Noise Harvesting:** Extract "Pure Noise" banks from unlabeled soundscape segments.

---

## Phase 2: Strategy Branching
Once Phase 1 is complete, implement the specific logic for each strategy.

### Branch A: 5s Unified (Refinement)
- **[5S-201] Model-Guided Peak Selection:** Use V1 model to find highest confidence 5s windows.
- **[5S-202] Contextual Stitcher:** Pad curated audio < 5s using harvested noise.
- **[5S-203] Refinement EDA:** Visualize Confidence Peaks vs. Energy Peaks.

### Branch B: 1s Stitching (Augmentation)
- **[1S-201] High-Res Segment Labeling:** Propagate grid labels to 1s temporal resolution.
- **[1S-202] Sliding Window Assembly:** Generate all valid 5s windows from dense feature scan.

---

## Phase 3: Assembly & V2 Training
- **[V2-301] Unified Dataset Assembly:** Merge all samples (Clean, Stitched, and Guided) into a master CSV.
- **[V2-302] V2 Fusion Model Training:** Train the 4-branch model (Adding BirdNET branch) using multi-label sigmoid head.
- **[V2-303] Soundscape Validation:** Final OOD evaluation on hold-out soundscape folds.

---

## Phase 4: Optimization & Ensembling (Lightweight Path)
- **[ENS-401] Feature Concatenation Pipeline:** Create a script to merge Perch and BirdNET embeddings into a flat vector.
- **[ENS-402] XGBoost/LightGBM Training:** Train an efficient ensemble on the flat feature vectors.
- **[ENS-403] Meta-Classifier (Stacking):** Implement the logic to blend Deep Fusion and Ensemble probabilities.
- **[ENS-404] Inference Speed Benchmark:** Measure the CPU runtime of both models to ensure 90-minute compliance.
