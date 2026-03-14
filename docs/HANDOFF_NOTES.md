# Handoff Notes: Soundscape Refinement & Unified 5s Training

**Current Status (March 14, 2026):**
- **V1 Model:** A 3-branch Fusion Model (EfficientNetB0 + Perch v1 + Metadata) is currently training locally on curated `train_audio`.
- **Performance:** Reached ~0.98 AUC on clean data by Epoch 9. Expected to struggle on noisy soundscapes.
- **Repository:** Cleaned and reorganized. Core logic in `src/`, infrastructure in `infrastructure/`.

## 1. The Core Strategy: "Unified 5s Resolution"
The goal is to bridge the domain gap between clean clips and noisy field recordings by standardizing all training data to high-quality 5-second windows at the audio level.

### Key Technical Decisions:
- **Multi-Label:** Move from Softmax/CCE to **Sigmoid/BinaryCrossentropy** to handle multiple birds in soundscapes.
- **Output Classes:** Fixed registry of **234 species IDs** (unifies short audio and soundscape labels).
- **Vocalization Centering:** Using the V1 model to "search" for the best 5s window within 1-minute soundscapes (dense scan with 1s stride).
- **Contextual Stitching:** Padding short curated audio with **real Pantanal noise** harvested from empty soundscape windows instead of using zero-padding.

## 2. Shared Synergies (Non-Redundant Path)
Multiple plans (`DATA_STRATEGY_JIRA`, `UNIFIED_5S_JIRA`) have been merged into a single optimized path. **Follow this order:**

### Phase 1: The Foundation (Immediate Priority)
1. **[BIR-105/UNI-104] Master Registry:** Create `species_registry.json` mapping all 234 IDs to indices.
2. **[BIR-101/UNI-103] Dense Scan:** Run 1s-stride feature extraction on `train_soundscapes`. This is the most computationally expensive task.
3. **[BIR-106/UNI-106] GroupKFold:** Assign `group_id` based on 1-minute filenames to prevent leakage.

### Phase 2: Refinement & Harvesting
1. **[UNI-101] Noise Harvesting:** Extract 5s "pure noise" segments from unlabeled soundscape windows.
2. **[BIR-103] Model-Guided Selection:** Use the V1 model to pick the "strongest" 5s windows from soundscapes for training.
3. **[UNI-102] Contextual Stitching:** Use harvested noise to pad curated audio < 5s.

### Phase 3: Assembly & V2 Training
1. **[UNI-104] Dataset Assembly:** Create `train_v2_master.csv` combining all refined sources.
2. **[UNI-106] V2 Training:** Train the 4-branch model (adding BirdNET branch as planned in `MODELING_PLAN.md`).

## 3. Essential Documentation
Before starting, the next implementer MUST read:
1. `docs/UNIFIED_PIPELINE_SYNERGIES.md` (The "Source of Truth" for the path forward).
2. `docs/SOUNDSCAPE_REFINEMENT_GUIDE.md` (Technical implementation details).
3. `docs/DATA_STRATEGY_JIRA.md` & `docs/UNIFIED_5S_IMPLEMENTATION_JIRA.md`.

## 4. Environment Notes
- **Local:** Intel Mac (CPU-only). Use `PYTHONPATH=. uv run python3 src/...`.
- **Cloud:** Vertex AI scripts in `infrastructure/vertex-ai/scripts/`. Uses `n1-standard-32` for speed.
- **Metrics:** Use the custom `CompetitionAUC` callback in `src/training/metrics.py` to match the Kaggle leaderboard.
