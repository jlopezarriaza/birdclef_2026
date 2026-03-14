# Strategy: Unified 5-Second Resolution Training

**Goal:** Standardize every training sample (Short Audio and Soundscapes) into a high-fidelity 5-second window to ensure optimal embedding quality and model robustness.

## 1. The Core Philosophy
Machine learning models for bioacoustics perform best when training samples match the resolution of the final task. In BirdCLEF 2026, the task is strictly **5-second window classification**. By standardizing all input data at the audio level *before* embedding extraction, we ensure that:
1. The model learns the temporal rhythm of calls over a full 5s context.
2. We maximize data augmentation via temporal sliding.
3. We eliminate the domain gap between "fragments" and "soundscapes."

## 2. Audio-Level Stitching & Padding
We reject zero-padding and mathematical embedding stitching. Instead, we use **Real-World Context Stitching**:

### A. Short Curated Audio (Duration < 5s)
- **Logic:** If a clip is 2s long, we don't pad with silence.
- **Stitching:** We concatenate the 2s clip with 3s of "True Background Noise" harvested from labeled-empty soundscape segments.
- **Benefit:** The model learns to find the bird *within* the noisy environment of the Pantanal.

### B. Soundscapes (Dense Sliding Window)
- **Logic:** We slide a 5s window across the 1-minute file with a 1s stride.
- **Result:** This effectively "stitches" consecutive seconds into 56 different 5s views.
- **Benefit:** Provides **Translation Invariance**—the bird can appear at the start, middle, or end of the 5s window.

## 3. Embedding Consistency
Every training sample, regardless of source, is processed through the same pipeline:
1. **Raw Audio (5s @ 32kHz)** -> 
2. **Perch/BirdNET Extractor** -> 
3. **Unified Feature Vector (Dim: 1280 or 6522)**.

This ensures the "feature space" is perfectly aligned across the entire training set.

## 4. Multi-Label Training
With unified 5s samples, we use a **Sigmoid + BinaryCrossentropy** head. This allows the model to predict all 234 species simultaneously, capturing the reality of the multi-species soundscapes.
