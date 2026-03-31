FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        fonts-dejavu-core \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libjpeg62-turbo \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "tensorflow[and-cuda]==2.15.1" \
        "mediapipe-model-maker==0.2.1.4" \
        pycocotools

WORKDIR /object-detection
