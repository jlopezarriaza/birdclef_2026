# Strategy: 5s Unified Resolution (Vocalization Centering)

**Objective:** Transform weak soundscape labels into "Strong Samples" by finding the exact 5-second window where a bird call is most prominent.

## 1. High-Level Concept
The competition labels are provided in 5s grid blocks (0-5, 5-10, etc.). However, a bird might call at second 4.5, resulting in two noisy labels with only fragments of the call. This strategy uses intelligence (Model-Guided) and physics (Energy-Based) to "re-center" the training window on the bird.

## 2. Hybrid Selection Logic (202 vs. 32 Species)
Because the initial Fusion Model was trained on `train.csv` (covering 202-206 species), it cannot provide confidence scores for the ~32 species that appear only in soundscapes or the test set. 

**Selection Protocol:**
- **For Known Species (202):** Use the **Fusion-Guided** approach. Scan the audio and select the 5s window with the highest Softmax probability for the target species.
- **For Gap Species (32):** Use the **Energy-Based** approach. Calculate the RMS energy (root-mean-square) across the audio and select the 5s window with the highest peak volume.
- **Fallback Rule:** If the Fusion model's peak confidence for a known species is critically low (< 0.15), default to the Energy-Based selection to ensure we aren't training on model "hallucinations."

## 3. Core Pillars

### A. Contextual Stitching
- Short curated recordings (duration < 5s) are never zero-padded.
- They are "stitched" onto real Pantanal background noise harvested from empty soundscape segments.
- This eliminates the "Silence Bias" where the model expects birds to only appear in quiet environments.

### B. Standardized Feature Space
- All inputs (Clean clips + Centered soundscape windows) are converted into unified Perch/BirdNET embeddings at a strict 5s resolution.
