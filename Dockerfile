##############################
# Stage 1: Builder (installs deps, trains model)
##############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRAIN_SEED=42 \
    PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only project metadata next (small changes here won't invalidate installed deps)
COPY pyproject.toml README.md /app/

# Copy code and data (these layers change more frequently)
COPY src /app/src
COPY configs /app/configs
COPY data /app/data

# Install package in editable mode so `python -m audio_diagnosis...` works
RUN pip install -e .

# Always preprocess and train during build (requires data present in build context)
RUN echo "[BUILD] Preprocessing audio" && \
    python -m audio_diagnosis.data.preprocessing --raw_dir "data/Audio Files" --out_dir data/processed --n_mels 128 --sr 4000 && \
    echo "[BUILD] Training CRNN" && \
    python -m audio_diagnosis.training.train_crnn --config configs/crnn_baseline.yaml && \
    ls -lh models && \
    echo "[BUILD] Training complete"

##############################
# Stage 2: Runtime (copy only what we need)
##############################
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model_crnn.pt \
    MC_SAMPLES=20 \
    CONFIDENCE_LEVEL=0.95 \
    PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

# Copy project metadata & source
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs

# Copy trained models from builder stage
COPY --from=builder /app/models /app/models

EXPOSE 8000
CMD ["uvicorn", "audio_diagnosis.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
