#!/bin/bash

# Configuration
PROJECT_ID="birdclef-490003"
REGION="us-central1"
JOB_NAME="stitcher-job-$(date +%Y%m%d-%H%M%S)"
CONFIG_PATH="infrastructure/vertex-ai/configs/job_config_stitcher.yaml"

echo "Submitting Custom Job to Vertex AI..."
echo "Job Name: $JOB_NAME"

gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=$JOB_NAME \
    --config=$CONFIG_PATH \
    --project=$PROJECT_ID

echo "Job submitted successfully."
