#!/bin/bash
# smoke_test_fusion.sh: Build and submit a tiny job to Vertex AI to verify the pipeline

PROJECT_ID="birdclef-490003"
REGION="us-central1"
REPO_NAME="birdclef-repo"
IMAGE_NAME="fusion-trainer"
IMAGE_TAG="smoke-test"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "1. Creating temporary job config for smoke test..."
cat <<EOF > job_config_smoke.yaml
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
  replicaCount: 1
  containerSpec:
    imageUri: ${IMAGE_URI}
    args:
    - --limit=50
    env:
    - name: KAGGLE_USERNAME
      value: "juanlopezarriaza"
    - name: KAGGLE_KEY
      value: "KGAT_45b265057d2a434f2e2eec260e818fe9"
    - name: KAGGLE_API_TOKEN
      value: "KGAT_45b265057d2a434f2e2eec260e818fe9"
    - name: GCS_BUCKET
      value: "birdclef-2026-data-birdclef-490003"
EOF

echo "2. Building Docker image: ${IMAGE_URI}..."
docker build -t ${IMAGE_URI} -f Dockerfile.fusion .

echo "3. Pushing image to Artifact Registry..."
docker push ${IMAGE_URI}

echo "4. Submitting Smoke Test Job to Vertex AI..."
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="birdclef-fusion-SMOKE-TEST" \
    --config=job_config_smoke.yaml

echo "Smoke test submitted. Monitor the job in the Vertex AI console."
echo "If successful, run ./deploy_fusion.sh for the full training."
rm job_config_smoke.yaml
