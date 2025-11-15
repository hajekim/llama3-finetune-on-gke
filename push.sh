#!/bin/bash

# Artifact Registry 설정
export PROJECT_ID="zeta-range-350705" # 본인의 GCP 프로젝트 ID로 변경하세요.
export REGION="us-central1" # GKE 클러스터가 위치한 리전으로 변경하세요 (예: us-central1).
export AR_REPO="llama3-finetune-repo" # Artifact Registry 리포지토리 이름
export IMAGE_NAME="llama3-finetune"
export TAG="latest"

export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${IMAGE_URI}"
docker build -t ${IMAGE_URI} .

echo "Pushing Docker image: ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Docker image build and push complete!"