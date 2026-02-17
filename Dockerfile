FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/storage/vectors/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/app/storage/vectors/hf-cache
ENV TRANSFORMERS_CACHE=/app/storage/vectors/hf-cache

RUN groupadd -g 1001 appuser && useradd -u 1001 -g appuser -s /usr/sbin/nologin appuser
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

COPY --from=builder /install /usr/local

COPY app app

RUN mkdir -p storage storage/vectors storage/logs /app/storage/vectors/hf-cache \
    && chown -R appuser:appuser storage /app/storage/vectors/hf-cache

USER appuser

EXPOSE 8080

ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
