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
# We use the system pip to install the exact TF packages that match this GPU image.
RUN pip install tensorflow-hub tensorflow-text==2.16.1

# Tell uv to install all other dependencies from our project, 
# but specifically skip the ones we just handled via pip.
RUN uv pip install --system \
    "librosa<=0.11.0" \
    "soundfile>=0.13.1" \
    "pandas>=2.3.3" \
    "numpy<2.0" \
    "tqdm>=4.67.3" \
    "scikit-learn>=1.6.1" \
    "matplotlib>=3.9.4" \
    "kaggle==2.0.0" \
    "kagglehub>=1.0.0" \
    "google-cloud-storage" \
    "opencv-python"

# Copy the source code
COPY src/ ./src/

# Environment variables
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"
ENV PYTHONUNBUFFERED=1

# Default command: run embedding extraction for Perch v2
ENTRYPOINT ["python3", "src/audio/extract_embeddings.py"]
