#!/bin/bash
set -e

# Configuration
IMAGE_URI="us-central1-docker.pkg.dev/birdclef-490003/birdclef-repo/perch-extractor:v2"
REGION="us-central1"
CONFIG_FILE="job_config.yaml"
JOB_NAME="perch-v2-extraction-$(date +%Y%m%d-%H%M)"

echo "----------------------------------------------------------"
echo "🚀 Starting Deployment for: $JOB_NAME"
echo "📦 Image: $IMAGE_URI"
echo "----------------------------------------------------------"

# 1. Build the container
echo "🔨 Step 1: Building Docker image..."
docker build -t "$IMAGE_URI" .

# 2. Push the container
echo "📤 Step 2: Pushing image to Artifact Registry..."
docker push "$IMAGE_URI"

# 3. Create the Vertex AI Job
echo "☁️  Step 3: Submitting Custom Job to Vertex AI..."
gcloud ai custom-jobs create \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="$CONFIG_FILE"

echo "----------------------------------------------------------"
echo "✅ Job submitted successfully!"
echo "📍 View logs: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=birdclef-490003"
echo "----------------------------------------------------------"
