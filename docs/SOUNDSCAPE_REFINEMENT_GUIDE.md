# Technical Guide: Soundscape Refinement & Vocalization Centering

This document provides the detailed technical implementation steps for bridging the domain gap between short curated audio and long field soundscapes.

## 1. High-Resolution Feature Mapping (BIR-101)
Instead of processing the 1-minute soundscape in discrete 5-second blocks, we perform a dense scan to capture the exact onset of vocalizations.

- **Window Size:** 5 seconds.
- **Stride:** 1 second.
- **Windows per file:** 56 (Windows starting at `t=0, 1, 2, ..., 55`).
- **Feature Extraction:** Both **Perch v1** (1,280-dim) and **BirdNET** (6,522-dim) features are extracted for all 56 windows.
- **Storage:** Results are saved as `(56, Dim)` arrays in `.npz` files, indexed by the original soundscape filename.

## 2. Model-Guided Peak Selection (BIR-103)
We use our "mini-model" (trained on high-quality short audio) as a "search engine" to find the cleanest samples within the noisy soundscapes.

### Selection Algorithm:
1. **Identify Target Species:** For a given soundscape and species `S`, find all intervals labeled as positive in `train_soundscapes_labels.csv`.
2. **Scan Neighborhood:** For a label interval `[T_start, T_end]`:
    - Define a search buffer: `[T_start - 2s, T_start + 2s]`.
    - Retrieve model confidence scores for species `S` for all windows starting within this buffer.
3. **Select Peak:** The 5-second window yielding the **highest confidence score** for species `S` is selected as the "Strong Sample."
4. **Fallback:** If the species was never seen by the mini-model, fall back to the window with the highest **RMS Energy** (BIR-102).

## 3. High-Resolution 1s Segment Stitching (BIR-107)
To further augment the data and teach the model "Translation Invariance" (recognizing the bird no matter where it is in the window), we use segment stitching.

- **Concept:** Slice the 1-minute file into sixty 1-second chunks.
- **Augmentation:** A positive 5-second window can be composed of segments `[1,2,3,4,5]`, `[2,3,4,5,6]`, etc.
- **Implementation:** We treat the 1-minute feature map as a sequence and use a sliding window of size 5 to generate diverse 5-second "views" of the same vocalization event.

## 4. Dataset Assembly (BIR-105)
All refined samples are pooled into a single master training set.

- **Resolution:** Uniform 5 seconds @ 32kHz.
- **Multi-Label:** All samples use Sigmoid activation + BinaryCrossentropy.
- **Metadata Integration:** Lat/Lon and Temporal cyclic features are attached to every slice.
- **GroupKFold Registry:** A `group_id` (the parent 1-minute filename) is assigned to prevent data leakage between training and validation folds.

## 5. Quality Control & Audit (BIR-104)
Before full training, we must validate that the centering logic isn't picking up loud background noise (wind/rain).

- **Energy vs. Confidence:** Plot both curves. A successful selection shows a "Confidence Peak" that matches the vocalization visual in the spectrogram, even if an "Energy Peak" (noise) exists elsewhere.
- **Manual Audio Review:** Export a sample of "disagreement" cases (where Energy and Model pick different windows) for human ear verification.
