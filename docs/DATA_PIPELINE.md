# Data Pipeline Documentation

This document explains how the raw competition data is transformed into processed artifacts for training.

## 1. Perch Embedding Extraction

### Overview
To accelerate training, we extract 1,280-dimensional embeddings from the raw audio files using Google's **Perch v2** model. Perch v2 uses an EfficientNet-B3 backbone and is trained on ~15,000 classes, providing higher-quality features than v1.

### Script
- **Source:** `src/audio/extract_embeddings.py`
- **Logic:**
    1. Loads the first 5 seconds of each training recording (resampled to 32 kHz).
    2. Pads recordings shorter than 5 seconds.
    3. Passes the audio through the Perch v2 model.
    4. Extracts the `embedding` output.

### Artifacts Generated
These files are stored in `data/processed/` (ignored by git, must be generated locally or via GCP):

1.  **`perch_v2_embeddings.npz`**: A compressed NumPy archive containing:
    - `embeddings`: A matrix of shape `(N, 1280)`.
    - `metadata_indices`: The original indices from `train.csv` corresponding to each embedding.
2.  **`train_with_perch_v2.csv`**: A copy of the training metadata that only includes successfully processed files, with an additional `embedding_idx` column for easy lookup.

### Usage in Training
To load these embeddings in a PyTorch Dataset or training script:

```python
import numpy as np
import pandas as pd

# Load artifacts
data = np.load("data/processed/perch_v1_embeddings.npz")
embeddings = data['embeddings']
indices = data['metadata_indices']

# Load mapped metadata
train_df = pd.read_csv("data/processed/train_with_embeddings.csv")

# Get embedding for a specific row
idx = 0
vector = embeddings[idx]
label = train_df.iloc[idx]['primary_label']
```

### Execution
To (re)generate the embeddings:
```bash
uv run python3 src/audio/extract_embeddings.py
```
*Note: This process may take several hours on CPU for the full 35,000+ files.*
