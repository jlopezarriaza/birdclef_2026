# Use official Python 3.11 image
FROM python:3.11-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Google Cloud SDK (for gsutil)
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

WORKDIR /app

# 3. Install EXACT Perch v2 stack (Discovery from successful smoke test)
# Bake these in so the Vertex node starts instantly and reliably
RUN pip install --no-cache-dir \
    "tensorflow-cpu~=2.20.0" \
    "tensorflow-text~=2.20.0" \
    tensorflow-hub \
    kagglehub \
    numpy \
    librosa \
    pandas \
    tqdm \
    kaggle

# 4. Copy source code
COPY src/ ./src/

# 5. Set environment variables
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"
ENV PYTHONUNBUFFERED=1

# 6. Entrypoint directly to the script
ENTRYPOINT ["python3", "src/audio/extract_embeddings_v2.py"]
