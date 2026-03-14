# Strategy: 1s Segment Stitching (High-Res Augmentation)

**Objective:** Maximize training data volume and achieve "Translation Invariance" by treating the soundscape as a sequence of 1-second building blocks.

## 1. High-Level Concept
Instead of selecting the "best" 5s window, we use *every* 5s window that contains a call. By sliding a 5s window with a 1s stride over active segments, we create multiple "views" of the same bird call.

## 2. Core Pillars
### A. Sliding Window Augmentation
- If a bird calls for 5 seconds, we generate 5 different samples where the call starts at $t=0, t=1, t=2, t=3, \text{ and } t=4$.
- This teaches the model to recognize the species regardless of its position within the 5s classification window.

### B. Segment-Level Labeling
- We propagate labels down to the 1-second resolution.
- A 5s window is considered "Positive" if any of its five 1s segments contain the target species.

### C. Sequence Integrity
- We ensure that embeddings are always extracted from the full 5s context. 
- We do NOT stitch embeddings; we stitch raw audio segments (or use dense windowing) to maintain the non-linear temporal features of Perch/BirdNET.
