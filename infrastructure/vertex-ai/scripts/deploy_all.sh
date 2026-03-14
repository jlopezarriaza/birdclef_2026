#!/bin/bash
set -e

# Usage: ./deploy_all.sh [v1|v2|birdnet] [local|vertex] [gcs_bucket]
TASK=$1
TARGET=${2:-vertex}  # Default to vertex if not specified
BUCKET=$3

PROJECT_ID="birdclef-490003"
REPO_NAME="birdclef-repo"
REGION="us-central1"

if [[ -z "$TASK" ]]; then
    echo "Usage: ./deploy_all.sh [v1|v2|birdnet] [local|vertex] [gcs_bucket]"
    exit 1
fi

case $TASK in
  v1)
    DOCKERFILE="Dockerfile.v1"
    IMAGE_TAG="perch-v1-extractor"
    SCRIPT="src/audio/extract_embeddings_v1.py"
    WORKERS=4
    ;;
  v2)
    DOCKERFILE="Dockerfile"
    IMAGE_TAG="perch-v2-extractor"
    SCRIPT="src/audio/extract_embeddings_v2_local.py"
    WORKERS=4
    ;;
  birdnet)
    DOCKERFILE="Dockerfile.birdnet"
    IMAGE_TAG="birdnet-extractor"
    SCRIPT="src/audio/extract_embeddings_birdnet.py"
    WORKERS=8
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

# Common Image Tag for local use
LOCAL_IMAGE="birdclef-${TASK}-local"
# Full URI for Vertex AI
REMOTE_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_TAG:latest"

# ---------------------------------------------------------
# TARGET: LOCAL (Docker)
# ---------------------------------------------------------
if [[ "$TARGET" == "local" ]]; then
    echo "🐳 BUILDING LOCAL IMAGE: $LOCAL_IMAGE..."
    docker build -f "$DOCKERFILE" -t "$LOCAL_IMAGE" .

    echo "🚀 RUNNING $TASK LOCALLY via Docker..."
    
    # Optional GCS argument
    GCS_ARG=""
    if [[ -n "$BUCKET" ]]; then
        GCS_ARG="--gcs_bucket $BUCKET"
        echo "☁️ Results will be uploaded to gs://$BUCKET"
    fi

    # Run with local data volume mounted
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        -e KAGGLE_USERNAME="juanlopezarriaza" \
        -e KAGGLE_API_TOKEN="KGAT_45b265057d2a434f2e2eec260e818fe9" \
        -e KAGGLE_KEY="KGAT_45b265057d2a434f2e2eec260e818fe9" \
        "$LOCAL_IMAGE" \
        --workers "$WORKERS" $GCS_ARG

    echo "✅ Local $TASK extraction complete!"

# ---------------------------------------------------------
# TARGET: VERTEX (Cloud)
# ---------------------------------------------------------
else
    if [[ -z "$BUCKET" ]]; then
        BUCKET="birdclef-2026-data-birdclef-490003"
    fi

    echo "🚀 DEPLOYING $TASK to VERTEX AI..."
    echo "🔨 Building remote image..."
    docker build -f "$DOCKERFILE" -t "$REMOTE_IMAGE" .

    echo "📤 Pushing image to Artifact Registry..."
    docker push "$REMOTE_IMAGE"

    # Generate a clean YAML job config
    TEMP_YAML="job_config_${TASK}.yaml"
    echo "☁️ Creating job config $TEMP_YAML..."

    # Vertex AI uses more workers (n1-standard-32)
    VERTEX_WORKERS=16

    cat <<CONFIG > $TEMP_YAML
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-32
  replicaCount: 1
  containerSpec:
    imageUri: $REMOTE_IMAGE
    args:
    - --workers=$VERTEX_WORKERS
    - --gcs_bucket=$BUCKET
    env:
    - name: KAGGLE_USERNAME
      value: "juanlopezarriaza"
    - name: KAGGLE_KEY
      value: "KGAT_45b265057d2a434f2e2eec260e818fe9"
    - name: KAGGLE_API_TOKEN
      value: "KGAT_45b265057d2a434f2e2eec260e818fe9"
CONFIG

    echo "☁️ Submitting Vertex AI Job..."
    gcloud ai custom-jobs create \
        --region="$REGION" \
        --display-name="${TASK}-extraction-$(date +%Y%m%d-%H%M)" \
        --config="$TEMP_YAML"

    echo "✅ $TASK deployment to Vertex AI complete!"
fi
