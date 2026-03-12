# Use official TensorFlow GPU base image for absolute compatibility
FROM tensorflow/tensorflow:2.16.1-gpu

# Install essential system tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /usr/bin/uv /usr/bin/uv

# Set working directory
WORKDIR /app

# Copy project configuration
COPY pyproject.toml uv.lock ./

# Install dependencies
# We use --system because we are inside a dedicated container
RUN uv pip install . --system

# Copy the source code
COPY src/ ./src/
COPY data/ ./data/

# Environment variables
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"
ENV PYTHONUNBUFFERED=1

# Default command: run embedding extraction for Perch v2
ENTRYPOINT ["python3", "src/audio/extract_embeddings.py"]
