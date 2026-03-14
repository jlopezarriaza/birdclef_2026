#!/bin/bash
set -e

echo "--- PRODUCTION RUNTIME BOOTSTRAP ---"
echo "Installing Perch v2 (CPU) dependencies..."

# Match the successful smoke test environment
pip install --no-cache-dir \
    "tensorflow-cpu~=2.20.0" \
    "tensorflow-text~=2.20.0" \
    tensorflow-hub \
    kagglehub \
    numpy \
    librosa \
    pandas \
    tqdm \
    kaggle

echo "✅ Dependencies installed. Starting Full Perch v2 Extraction..."

# Run the production extraction script
# The arguments ($@) allow passing flags from the Vertex job config (workers, bucket, etc.)
python3 src/audio/extract_embeddings_v2.py "$@"
