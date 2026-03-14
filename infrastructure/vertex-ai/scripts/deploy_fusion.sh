#!/bin/bash
# deploy_fusion.sh: Build and submit the Fusion Model training job to Vertex AI

PROJECT_ID="birdclef-490003"
REGION="us-central1"
REPO_NAME="birdclef-repo"
IMAGE_NAME="fusion-trainer"
IMAGE_TAG="latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image: ${IMAGE_URI}..."
docker build -t ${IMAGE_URI} -f ../../docker/Dockerfile.fusion ../../../

echo "Pushing image to Artifact Registry..."
docker push ${IMAGE_URI}

echo "Submitting Custom Job to Vertex AI..."
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="birdclef-fusion-training" \
    --config=../configs/job_config_fusion.yaml

echo "Job submitted successfully."
