#!/bin/bash

# Local DOCKER-based Extraction for Perch v2
# This ensures we use a Linux environment with the correct TF version
# to avoid StableHLO/Bytecode errors on Intel Mac.

echo "🐳 Starting Perch v2 Extraction via Docker (Local)..."

# Build the v2 image if it doesn't exist
# Using the same Dockerfile as Vertex AI
docker build -f Dockerfile -t perch-v2-local .

# Mount local data directory to /app/data in container
# and run the extraction script.
# We override the ENTRYPOINT to use our local script.

WORKERS=4
GCS_BUCKET=$1

if [[ -n "$GCS_BUCKET" ]]; then
    GCS_ARG="--gcs_bucket $GCS_BUCKET"
    echo "☁️ Will upload to gs://$GCS_BUCKET"
fi

docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -e KAGGLE_USERNAME="juanlopezarriaza" \
    -e KAGGLE_API_TOKEN="KGAT_45b265057d2a434f2e2eec260e818fe9" \
    -e KAGGLE_KEY="KGAT_45b265057d2a434f2e2eec260e818fe9" \
    perch-v2-local \
    --workers $WORKERS $GCS_ARG

echo "✅ Docker-based extraction process finished."
echo "Check data/processed/ for 'perch_v2_embeddings.npz'."
