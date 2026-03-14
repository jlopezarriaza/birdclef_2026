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
Follow the execution order in `docs/IMPLEMENTATION_ROADMAP.md`:

### Phase 1: Shared Infrastructure (Immediate Priority)
1. **[SHR-101] Master Registry:** Create mapping for all 234 species IDs.
2. **[SHR-102] Dense Scan:** Run 1s-stride feature extraction on `train_soundscapes`.
3. **[SHR-103] GroupKFold:** Map soundscapes to parent IDs to prevent leakage.
4. **[SHR-104] Noise Harvesting:** Extract 5s "pure noise" segments.

### Phase 2: Refinement & Augmentation
- Implement **Branch A (5s Unified)** for high-fidelity refinement.
- Implement **Branch B (1s Stitching)** for massive data augmentation.

### Phase 3: Assembly & V2 Training
1. **[V2-301] Dataset Assembly:** Merge all sources into `train_v2_master.csv`.
2. **[V2-302] V2 Training:** Train the 4-branch model (Adding BirdNET branch).

## 3. Essential Documentation
Before starting, the next implementer MUST read:
1. `docs/IMPLEMENTATION_ROADMAP.md` (The "Source of Truth" for the path forward).
2. `docs/STRATEGY_5S_UNIFIED.md` (Detailed centering & stitching strategy).
3. `docs/STRATEGY_1S_STITCHING.md` (Detailed high-res augmentation strategy).

## 4. Environment Notes
- **Local:** Intel Mac (CPU-only). Use `PYTHONPATH=. uv run python3 src/...`.
- **Cloud:** Vertex AI scripts in `infrastructure/vertex-ai/scripts/`. Uses `n1-standard-32` for speed.
- **Metrics:** Use the custom `CompetitionAUC` callback in `src/training/metrics.py` to match the Kaggle leaderboard.
