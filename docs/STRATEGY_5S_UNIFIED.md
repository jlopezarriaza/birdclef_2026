# Strategy: 5s Unified Resolution (Vocalization Centering)

**Objective:** Transform weak soundscape labels into "Strong Samples" by finding the exact 5-second window where a bird call is most prominent.

## 1. High-Level Concept
The competition labels are provided in 5s grid blocks (0-5, 5-10, etc.). However, a bird might call at second 4.5, resulting in two noisy labels with only fragments of the call. This strategy uses intelligence (Model-Guided) and physics (Energy-Based) to "re-center" the training window on the bird.

## 2. Core Pillars
### A. Model-Guided Selection
- Use the current V1 Model (trained on clean audio) to scan the soundscape.
- Select the 5s window (with 1s stride) that yields the highest confidence for the target species.
- This ensures the model learns from the cleanest possible field data.

### B. Contextual Stitching
- Short curated recordings (duration < 5s) are never zero-padded.
- They are "stitched" onto real Pantanal background noise harvested from empty soundscape segments.
- This eliminates the "Silence Bias" where the model expects birds to only appear in quiet environments.

### C. Standardized Feature Space
- All inputs (Clean clips + Centered soundscape windows) are converted into unified Perch/BirdNET embeddings at a strict 5s resolution.
