# Strategy: Multi-Peak Training Expansion

**Objective:** Increase training data volume and diversity by extracting every high-confidence vocalization segment from the curated `train_audio` dataset, rather than just the single "best" 5-second window.

## 1. Logic & Thresholding
Instead of an `argmax` operation over all windows in a file, the Multi-Peak strategy employs a threshold-based extraction:

- **Primary Rule:** Extract every 5-second window where the Fusion Model confidence for the `primary_label` is **> 0.4**.
- **Coverage Rule:** If no window in a file meets the 0.4 threshold, fallback to the single highest-confidence window (or RMS peak if model confidence is < 0.15) to ensure every file contributes at least one sample.
- **Stride:** Can be run with 5s consecutive windows (Fast) or 1s sliding windows (High-Res/Augmented) depending on compute availability.

## 2. Technical Implementation
- **Manifest Schema:** The `train_v2_peaks.csv` will expand from 35,550 rows to an estimated 80,000–120,000 rows.
- **Filename Handling:** Since multiple windows will map to the same source `.ogg` file, the `unified_filename` must include a window index (e.g., `1161364_iNat1216197_win02.wav`).
- **Deduplication:** Ensure that overlapping windows (if using 1s stride) do not lead to extreme overfitting on the same vocalization burst.

## 3. Potential Impact
- **Pros:** 
    - Significantly more training steps per epoch.
    - Model learns the start, middle, and end of various calls.
    - Captures secondary calls or distant birds that might be missed in a "Best 1" search.
- **Cons:**
    - Increased risk of "Label Noise" on lower-confidence windows.
    - Potential for dataset imbalance if certain species sing much longer than others.
    - Requires 3x-4x more storage and training time.

## 4. Execution Trigger
This strategy should be activated if:
1. The V2 model (trained on Single Peaks) plateaus with high training accuracy but poor validation performance.
2. The model exhibits high variance on long recordings in the soundscape validation set.
