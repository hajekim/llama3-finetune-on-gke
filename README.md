# GKE(Google Kubernetes Engine)를 활용한 Llama 3 파인튜닝 가이드

이 문서는 Google Kubernetes Engine(GKE)의 `a3-megagpu-8g` (NVIDIA H100 8 chips) 인스턴스를 사용하여 `meta-llama/Meta-Llama-3-8B-Instruct` 모델을 파인튜닝 하는 전체 과정을 안내합니다.

## 프로젝트 목표

-   `databricks/databricks-dolly-15k` 데이터셋을 사용하여 Llama 3 모델을 효율적으로 파인튜닝합니다.
-   Workload Identity를 사용하여 GKE에서 Google Cloud Storage(GCS)로 안전하게 모델 아티팩트를 업로드합니다.
-   전체 과정을 재현할 수 있도록 Dockerfile, Kubernetes Job YAML, 스크립트를 제공합니다.

---

## 사전 준비 사항

1.  **Google Cloud Project**: 결제가 활성화된 GCP 프로젝트가 필요합니다.
2.  **CLI 도구**: `gcloud` CLI와 `kubectl`이 로컬 머신에 설치 및 인증되어 있어야 합니다.
3.  **GKE 클러스터**: `a3-megagpu-8g` 노드 풀이 구성된 GKE 클러스터가 필요합니다.
    -   **참고**: `a3-megagpu-8g`는 할당량이 필요하며, `us-central1`과 같은 특정 리전에서만 사용할 수 있습니다.
4.  **GCS Bucket**: 파인튜닝 된 모델 아티팩트를 저장할 GCS 버킷이 필요합니다.
5.  **Hugging Face 계정**: Llama 3 모델에 접근하려면 Hugging Face 계정과 `hf_...` 형식의 액세스 토큰이 필요합니다.

---

## 설정 단계

### 1. 서비스 계정 및 권한 설정 (Workload Identity)

GKE 파드가 GCS에 안전하게 접근할 수 있도록 Workload Identity를 설정합니다.

#### 1.1. IAM 서비스 계정 생성

```bash
# 변수 설정
export PROJECT_ID="<YOUR_PROJECT_ID>"
export GCS_BUCKET_NAME="<YOUR_GCS_BUCKET_NAME>"
export IAM_SA_NAME="llama3-finetuner-sa"

# IAM 서비스 계정 생성
gcloud iam service-accounts create ${IAM_SA_NAME} \
  --project=${PROJECT_ID} \
  --display-name="Llama3 Finetuner Service Account"
```

#### 1.2. GCS 버킷에 대한 권한 부여

생성한 IAM 서비스 계정에 GCS 버킷에 대한 `Storage Object Admin` 역할을 부여합니다.

```bash
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member "serviceAccount:${IAM_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role "roles/storage.objectAdmin"
```

#### 1.3. Kubernetes 서비스 계정 생성 및 연결

GKE 클러스터에 Kubernetes 서비스 계정(KSA)을 생성하고, IAM 서비스 계정과 연결합니다.

```bash
# 변수 설정
export K8S_SA_NAME="llama3-finetuner-ksa"
export NAMESPACE="default"

# Kubernetes 서비스 계정 생성
kubectl create serviceaccount ${K8S_SA_NAME} --namespace ${NAMESPACE}

# IAM 서비스 계정과 KSA를 연결하는 어노테이션 추가
gcloud iam service-accounts add-iam-policy-binding ${IAM_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${K8S_SA_NAME}]"
```

### 2. Hugging Face 액세스 토큰 Secret 생성

Hugging Face 토큰을 Kubernetes Secret으로 안전하게 저장합니다.

```bash
kubectl create secret generic huggingface-secret \
  --from-literal=token='<YOUR_HUGGING_FACE_TOKEN>'
```

### 3. Docker 이미지 빌드 및 푸시

제공된 `Dockerfile`을 사용하여 파인튜닝 환경을 포함하는 Docker 이미지를 빌드하고, Google Artifact Registry에 푸시합니다.

`push.sh` 스크립트는 이 과정을 자동화합니다. 스크립트 내의 `PROJECT_ID`와 `REPO`를 자신의 환경에 맞게 수정하세요.

```bash
# push.sh
export PROJECT_ID="<YOUR_PROJECT_ID>" # 자신의 GCP 프로젝트 ID로 변경하세요.
export REPO="ml-containers"
export IMAGE_NAME="llama-fsdp-trainer"
export TAG="latest"
export REGION="<YOUR_GCP_REGION>" # GKE 클러스터가 위치한 리전으로 변경하세요 (예: us-central1).

export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

# Docker 이미지 빌드 및 푸시
gcloud builds submit --tag ${IMAGE_URI} .
```

스크립트를 실행합니다.

```bash
bash push.sh
```

### 4. 파인 튜닝 작업 실행

모든 설정이 완료되면 `finetune-job.yaml` 파일을 사용하여 Kubernetes Job을 실행합니다. 이 파일은 올바른 서비스 계정, 노드 셀렉터, 리소스 등을 사용하도록 사전 구성되어 있습니다.

```bash
kubectl apply -f finetune-job.yaml
```

---

## 모니터링 및 결과 확인

#### 1. 파드 상태 확인

작업이 생성되면 파드의 상태를 확인하여 정상적으로 실행되는지 모니터링합니다. `Pending` -> `ContainerCreating` -> `Running` 순서로 상태가 변경됩니다.

```bash
kubectl get pods -l job-name=llama3-finetune-job -w
```

#### 2. 로그 스트리밍

파드가 `Running` 상태가 되면, 로그를 실시간으로 확인하여 모델 다운로드, 데이터 처리, 학습 진행 상황을 모니터링할 수 있습니다.

```bash
# <pod-name>을 위 명령어에서 확인한 실제 파드 이름으로 변경하세요.
export POD_NAME=$(kubectl get pods -l job-name=llama3-finetune-job -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME
```

학습이 완료되면 다음과 유사한 로그가 출력됩니다.

```
{'train_runtime': 8.7603, 'train_loss': 3.2107, 'epoch': 1.0}
...
Successfully uploaded model to gs://<your-bucket-name>/final_model
Fine-tuning and GCS upload complete!
```

#### 3. GCS 버킷 결과 확인

작업이 성공적으로 완료되면, 파인 튜닝 된 모델 아티팩트가 GCS 버킷에 저장됩니다. `gsutil` 명령어로 파일 목록을 확인할 수 있습니다.

```bash
gsutil ls gs://<YOUR_GCS_BUCKET_NAME>/final_model/
```

다음과 같은 파일들이 보여야 합니다.
-   `adapter_config.json`
-   `adapter_model.safetensors`
-   `README.md`
-   `special_tokens_map.json`
-   `tokenizer_config.json`
-   `tokenizer.json`
-   `training_args.bin`

---

## 파일 설명

-   **`Dockerfile`**: 파인 튜닝 환경을 위한 Docker 이미지를 빌드하는 파일입니다. (상세 설명은 위 `파일 상세 설명` 섹션 참조)
-   **`finetune-job.yaml`**: GKE에서 파인 튜닝 Job을 실행하기 위한 Kubernetes 명세 파일입니다. (상세 설명은 위 `파일 상세 설명` 섹션 참조)
-   **`scripts/finetune.py`**: 실제 모델 로드, 데이터 처리, 파인 튜닝 및 GCS 업로드를 수행하는 Python 스크립트입니다.
-   **`push.sh`**: `Dockerfile`을 빌드하고 Google Artifact Registry에 이미지를 푸시하는 과정을 자동화하는 셸 스크립트입니다.
