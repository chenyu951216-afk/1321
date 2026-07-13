FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    DATA_DIR=/data \
    MODEL_DIR=/models \
    MAX_UPLOAD_MB=15 \
    MAX_IMAGE_SIDE=2500 \
    TEMP_RETENTION_HOURS=24 \
    RESULT_RETENTION_HOURS=24 \
    MAX_CONCURRENT_JOBS=1 \
    FACE_DETECTION_SIZE=640 \
    JPEG_QUALITY=95

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

ARG ONNXRUNTIME_VARIANT=cpu
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && case "$ONNXRUNTIME_VARIANT" in \
        cpu) ;; \
        gpu) \
            python -m pip uninstall -y onnxruntime \
            && python -m pip install --no-cache-dir onnxruntime-gpu==1.27.0 \
            ;; \
        *) \
            echo "ONNXRUNTIME_VARIANT must be 'cpu' or 'gpu'." >&2 \
            && exit 2 \
            ;; \
    esac

COPY scripts ./scripts

ARG INSWAPPER_MODEL_URL=""
ARG BUFFALO_L_MODEL_URL=""
RUN mkdir -p /models /data/faces /data/temp /data/results \
    && INSWAPPER_MODEL_URL="$INSWAPPER_MODEL_URL" \
       BUFFALO_L_MODEL_URL="$BUFFALO_L_MODEL_URL" \
       python scripts/download_models.py --model-dir /models --allow-failure

COPY app ./app
COPY start.sh ./start.sh

RUN sed -i 's/\r$//' /app/start.sh \
    && chmod 0755 /app/start.sh

EXPOSE 8080

CMD ["/app/start.sh"]
