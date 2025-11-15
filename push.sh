export PROJECT_ID="zeta-range-350705"
export REGION="us-central1"
export REPO_NAME="ml-containers"
export IMAGE_NAME="llama-fsdp-trainer:latest"

# 1. Artifact Registry 저장소 생성
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repo for ML training images"

# 2. gcloud CLI에 Docker 인증 구성
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker build -t ${REPO_NAME}/${IMAGE_NAME} .

# 3. Cloud Build를 사용하여 Dockerfile 빌드 및 푸시
gcloud builds submit . \
    --region=${REGION} \
    --tag=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}
