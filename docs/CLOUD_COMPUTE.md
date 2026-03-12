# Google Cloud & Vertex AI Pipeline

This document details the workflow for leveraging Google Cloud Platform (GCP) to perform high-speed data processing and model training for the BirdCLEF 2026 competition.

## 1. Overview
Due to the large dataset (15GB) and the complexity of the Perch v2 model, we use Vertex AI Custom Training jobs to perform batch processing. This allows us to use high-core CPU or GPU instances that are significantly faster than local hardware.

### Key Components
- **Artifact Registry:** Stores our custom Docker images.
- **Vertex AI Custom Jobs:** Serverless execution of our processing scripts.
- **Cloud Storage (GCS):** Permanent storage for processed artifacts (embeddings, mapped metadata).

## 2. Infrastructure Setup (One-Time)

### Environment Variables
```bash
export PROJECT_ID="birdclef-490003"
export BUCKET_NAME="birdclef-2026-data-$PROJECT_ID"
gcloud config set project $PROJECT_ID
```

### Enable APIs
```bash
gcloud services enable artifactregistry.googleapis.com aiplatform.googleapis.com storage.googleapis.com
```

### Resource Creation
```bash
# Create Docker Registry
gcloud artifacts repositories create birdclef-repo --repository-format=docker --location=us-central1

# Create Storage Bucket
gsutil mb -l us-central1 gs://$BUCKET_NAME
```

## 3. Build and Push Workflow

We use Docker to package our `uv`-managed environment. The `Dockerfile` is based on the official TensorFlow CPU image for maximum efficiency on Vertex AI worker nodes.

### Build Command
Run this from the project root to build for the cloud's architecture and push to the registry:
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

docker buildx build --platform linux/amd64 \
    -t us-central1-docker.pkg.dev/$PROJECT_ID/birdclef-repo/perch-extractor:v2 \
    --push .
```

## 4. Running Jobs on Vertex AI

Jobs are defined in `job_config.yaml`. This approach avoids complex CLI syntax and ensures reproducibility.

### Performance Optimization
The extraction script is **hardware-intelligent**:
- **CPU Parallelization:** On many-core machines (e.g., 32-core), it spawns parallel processes to maximize throughput.
- **GPU Readiness:** It automatically detects GPUs and switches to a single-process mode to avoid VRAM exhaustion, while enabling XLA JIT for maximum NVIDIA performance.

We currently use a **32-core high-RAM machine** (`n1-standard-32`) to achieve ~32x speedup over local sequential processing. 
- **Critical Requirement:** Parallel Perch v2 extraction requires ~120GB of RAM to prevent `Replicas low on memory` errors. The `highcpu` variants do not provide sufficient RAM for 32 parallel model instances.

### Submission Command
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --display-name=perch-v2-extraction-high-ram \
  --config=job_config.yaml
```

## 5. Data Retrieval

Once the job succeeds, artifacts are automatically uploaded to GCS. Download them locally for model training:

```bash
mkdir -p data/processed
gsutil cp gs://$BUCKET_NAME/processed/perch_v2_embeddings.npz data/processed/
gsutil cp gs://$BUCKET_NAME/processed/train_with_perch_v2.csv data/processed/
```

## 6. Troubleshooting

- **Memory Errors:** If you see `Replicas low on memory`, ensure you are using a `standard` machine type (e.g., `n1-standard-32`) rather than `highcpu`.
- **Quota Errors:** If you see `RESOURCE_EXHAUSTED` for GPUs, switch the `machineType` in `job_config.yaml` to the high-core CPU instance mentioned above.
- **Serialization Errors:** Perch v2 may fail locally on Intel Mac due to `vhlo.func_v1` errors. Always use the Docker/Cloud pipeline for V2 processing.
- **Log Monitoring:**
  ```bash
  gcloud ai custom-jobs stream-logs [JOB_ID]
  ```
