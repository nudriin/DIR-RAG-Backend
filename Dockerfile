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

RUN groupadd -g 1001 appuser && useradd -u 1001 -g appuser -s /usr/sbin/nologin appuser

COPY --from=builder /install /usr/local

COPY app app

RUN mkdir -p storage storage/vectors storage/logs && chown -R appuser:appuser storage

USER appuser

EXPOSE 8080

ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

