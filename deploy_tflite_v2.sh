#!/bin/bash
set -e

IMAGE_URI="us-central1-docker.pkg.dev/birdclef-490003/birdclef-repo/perch-extractor:tflite-v2"
REGION="us-central1"
CONFIG_FILE="job_config_test_v2.yaml"
JOB_NAME="perch-v2-tflite-smoke-test-$(date +%Y%m%d-%H%M)"

# 1. Update job config to CPU only for TFLite smoke test
cat <<EOF > job_config_tflite_test.yaml
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
  replicaCount: 1
  containerSpec:
    imageUri: $IMAGE_URI
    env:
    - name: KAGGLE_USERNAME
      value: "juanlopezarriaza"
    - name: KAGGLE_KEY
      value: "KGAT_45b265057d2a434f2e2eec260e818fe9"
EOF

echo "----------------------------------------------------------"
echo "🚀 Building TFLite Image locally (Lightweight)..."
docker build -f Dockerfile.tflite -t "$IMAGE_URI" .

echo "📤 Pushing Image..."
docker push "$IMAGE_URI"

echo "☁️ Submitting Vertex AI TFLite Smoke Test Job (CPU)..."
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="job_config_tflite_test.yaml"

echo "----------------------------------------------------------"
echo "✅ TFLite Test Job submitted!"
echo "Check logs here: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=birdclef-490003"
