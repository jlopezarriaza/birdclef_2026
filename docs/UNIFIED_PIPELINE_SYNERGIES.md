# Unified Pipeline Synergies & Dependencies

To avoid redundant work, this document maps the shared components between the **Soundscape Refinement** (BIR) and **Unified 5s Resolution** (UNI) plans.

## 1. The Foundation (Shared by BOTH)
The following tasks are strictly identical and provide the data needed for all downstream experiments.

| Task ID | Component | Dependency | Shared Benefit |
| :--- | :--- | :--- | :--- |
| **BIR-101 / UNI-103** | **Dense Feature Extraction** | None | Provides 56 embeddings per soundscape file (1s stride). Essential for both peak selection and segment stitching. |
| **BIR-105 / UNI-104** | **234-Species Master Registry** | None | Defines the multi-hot mapping for the final model head. |
| **BIR-106 / UNI-106** | **GroupKFold Registry** | None | Ensures 1-minute files are used exclusively for Train or Val to prevent leakage. |

## 2. Complementary Logic (No Overlap)
These tasks are unique to their respective plans but can be performed in parallel once the Foundation is ready.

- **[UNI-101] Noise Harvesting:** Extracting silent segments from soundscapes. *Unique to UNI.*
- **[UNI-102] Contextual Stitcher:** Using harvested noise to pad short clips. *Unique to UNI.*
- **[BIR-103] Model-Guided Labeling:** Using current model to pick "best" windows. *Unique to BIR.*

## 3. The Non-Redundant Execution Order
1. **FOUNDATION:** Extract Dense Features (1s stride) + Create Master Registry.
2. **HARVESTING:** Identify Pure Noise segments from soundscapes.
3. **REFINEMENT:** Run both Model-Guided Selection (BIR) AND Contextual Stitching (UNI).
4. **ASSEMBLY:** Create a single training CSV that includes slices selected by both strategies.
5. **TRAINING:** Train the V2 model on this combined "Super-Dataset."
