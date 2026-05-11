# multi-stage build: deps in stage 1, runtime in stage 2 (smaller final image)

# ---------- builder ----------
FROM python:3.11-slim AS builder

# system deps for compiling pyarrow / lightgbm / faiss wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps into a virtualenv we'll copy to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---------- runtime ----------
FROM python:3.11-slim AS runtime

# minimal runtime deps for lightgbm / faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

# code + model checkpoints
# data parquets are too large to bake into images — mounted as volume at runtime
COPY src/ ./src/
COPY checkpoints/two_tower.pt ./checkpoints/two_tower.pt
COPY checkpoints/ranker.lgb ./checkpoints/ranker.lgb

# the service expects the data dir at this path; user mounts a volume here
ENV RECSYS_DATA_DIR=/data/parquet \
    RECSYS_CKPT_DIR=/app/checkpoints

EXPOSE 8000

# health check: container restarts if /health fails repeatedly
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
