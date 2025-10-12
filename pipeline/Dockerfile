# pipeline/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# sys deps for regex speed, pdftools (if needed), faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# spaCy small model (быстро и достаточно для токенизации/negex)
RUN python -m spacy download en_core_web_sm

# код пайплайна и правила
COPY . /app

# по умолчанию контейнер запускает воркер
CMD ["python", "-m", "pipeline.worker"]
