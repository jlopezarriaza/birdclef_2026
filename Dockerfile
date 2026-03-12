# Use official TensorFlow CPU base image for efficiency on high-cpu machines
FROM tensorflow/tensorflow:2.16.1

# Install essential system tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project configuration
COPY pyproject.toml uv.lock ./

# CRITICAL FIX for "undefined symbol" errors:
# We use the system pip to install the exact TF packages that match this GPU image,
# and tell uv to install everything else WITHOUT touching the tensorflow ecosystem.
RUN pip install tensorflow-hub tensorflow-text==2.16.1
RUN uv pip install . --system --exclude tensorflow --exclude tensorflow-hub --exclude tensorflow-text

# Copy the source code
COPY src/ ./src/

# Environment variables
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"
ENV PYTHONUNBUFFERED=1

# Default command: run embedding extraction for Perch v2
ENTRYPOINT ["python3", "src/audio/extract_embeddings.py"]
