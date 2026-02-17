#!/usr/bin/env bash
set -e

PROJECT_ID=${PROJECT_ID:-"gen-lang-client-0221576884"}
REGION=${REGION:-"asia-southeast2"}
SERVICE_NAME=${SERVICE_NAME:-"dir-rag-backend"}
REPO_NAME=${REPO_NAME:-"dir-rag-backend"}

IMAGE="asia-southeast2-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE}" .

gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --min-instances=0 \
  --max-instances=1 \
  --cpu=1 \
  --memory=4Gi \
  --ephemeral-storage=8Gi \
  --port=8080 \
  --allow-unauthenticated
