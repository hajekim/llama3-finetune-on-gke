#!/bin/bash

# === 사용자 설정 변수 ===
export PROJECT_ID="<YOUR_PROJECT_ID>" # 🚨 본인의 GCP 프로젝트 ID로 변경하세요.
export REGION="<YOUR_GCP_REGION>"       # 🚨 GKE 클러스터가 위치한 리전으로 변경하세요 (예: us-central1).
export AR_REPO="llama3-finetune-repo"    # Artifact Registry 리포지토리 이름 (필요시 변경)
export IMAGE_NAME="llama3-finetune"      # Docker 이미지 이름 (필요시 변경)
export TAG="latest"                      # 이미지 태그
# =======================

export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${IMAGE_URI}"
docker build -t ${IMAGE_URI} .

echo "Pushing Docker image: ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Docker image build and push complete!"