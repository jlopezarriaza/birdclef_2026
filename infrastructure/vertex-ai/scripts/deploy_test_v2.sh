#!/bin/bash
set -e

REGION="us-central1"
CONFIG_FILE="job_config_test_v2.yaml"
JOB_NAME="perch-v2-smoke-test-$(date +%Y%m%d-%H%M)"

echo "----------------------------------------------------------"
echo "☁️ Submitting Cloud Build for Docker Image..."
gcloud builds submit --config cloudbuild.yaml .

echo "----------------------------------------------------------"
echo "☁️ Submitting Vertex AI Smoke Test Job..."
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="$CONFIG_FILE"

echo "----------------------------------------------------------"
echo "✅ Done!"
echo "Check logs here: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=birdclef-490003"
