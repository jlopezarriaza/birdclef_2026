#!/bin/bash
set -e

# Usage: ./deploy_all.sh [v1|v2|birdnet]
TASK=$1
PROJECT_ID="birdclef-490003"
REPO_NAME="birdclef-repo"
REGION="us-central1"
BUCKET="birdclef-2026-data-birdclef-490003"

if [[ -z "$TASK" ]]; then
    echo "Usage: ./deploy_all.sh [v1|v2|birdnet]"
    exit 1
fi

case $TASK in
  v1)
    DOCKERFILE="Dockerfile.v1"
    IMAGE_TAG="perch-v1-extractor"
    ;;
  v2)
    DOCKERFILE="Dockerfile"
    IMAGE_TAG="perch-v2-extractor"
    ;;
  birdnet)
    DOCKERFILE="Dockerfile.birdnet"
    IMAGE_TAG="birdnet-extractor"
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_TAG:latest"

echo "🚀 Deploying $TASK..."
echo "🔨 Building image..."
docker build -f "$DOCKERFILE" -t "$IMAGE_URI" .

echo "📤 Pushing image..."
docker push "$IMAGE_URI"

# Generate a clean YAML job config
TEMP_YAML="job_config_${TASK}.yaml"
echo "☁️ Creating job config $TEMP_YAML..."

cat <<CONFIG > $TEMP_YAML
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-32
  replicaCount: 1
  containerSpec:
    imageUri: $IMAGE_URI
    args:
    - --workers=16
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

echo "✅ $TASK deployment complete!"
