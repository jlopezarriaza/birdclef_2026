#!/bin/bash
set -e

IMAGE_URI="us-central1-docker.pkg.dev/birdclef-490003/birdclef-repo/perch-extractor:v2-final"
REGION="us-central1"
CONFIG_FILE="job_config_v2.yaml"
JOB_NAME="perch-v2-full-extraction-$(date +%Y%m%d-%H%M)"

# 1. Update cloudbuild to use the new tag
cat <<EOF > cloudbuild_final.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '$IMAGE_URI', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '$IMAGE_URI']
images:
  - '$IMAGE_URI'
options:
  logging: CLOUD_LOGGING_ONLY
EOF

echo "----------------------------------------------------------"
echo "☁️ Submitting Cloud Build (Wait for finish)..."
gcloud builds submit --config cloudbuild_final.yaml .

echo "----------------------------------------------------------"
echo "☁️ Submitting Vertex AI FULL Extraction Job..."
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="$CONFIG_FILE"

echo "----------------------------------------------------------"
echo "✅ Done! Full extraction is now running."
echo "Logs: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=birdclef-490003"
