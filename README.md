# GKE를 활용한 Llama 3 파인 튜닝 가이드

이 문서는 GKE(Google Kubernetes Engine)의 `a3-megagpu-8g` (NVIDIA H100 8 chips) 인스턴스를 사용하여 `meta-llama/Meta-Llama-3-8B-Instruct` 모델을 파인 튜닝 하는 전체 과정을 안내합니다. 싱글 노드(8 GPUs) 및 멀티 노드(16 GPUs) 설정에 대한 지침을 모두 포함합니다.

## 프로젝트 목표

-   `databricks/databricks-dolly-15k` 데이터셋을 사용하여 Llama 3 모델을 효율적으로 파인 튜닝합니다.
-   Workload Identity를 사용하여 GKE에서 GCS(Google Cloud Storage)로 안전하게 모델 아티팩트를 업로드합니다.
-   싱글 노드 및 멀티 노드 학습을 재현할 수 있도록 Dockerfile, Kubernetes Job YAML, 스크립트를 제공합니다.

---

## 사전 준비 사항

1.  **Google Cloud Project**: 결제가 활성화된 GCP 프로젝트가 필요합니다.
2.  **CLI 도구**: `gcloud` CLI와 `kubectl`이 로컬 머신에 설치 및 인증되어 있어야 합니다.
3.  **GKE 클러스터**: `a3-megagpu-8g` 노드 풀이 구성된 GKE 클러스터가 필요합니다.
    -   **참고**: `a3-megagpu-8g` Quota가 필요하며, 할당 받은 리전에서 사용할 수 있습니다.
4.  **GCS Bucket**: 파인 튜닝 된 모델 아티팩트를 저장할 GCS 버킷이 필요합니다.
5.  **Hugging Face 계정**: Llama 3 모델에 접근하려면 Hugging Face 계정과 `hf_...` 형식의 액세스 토큰이 필요합니다.

---

## 설정 단계

### 1. 서비스 계정 및 권한 설정 (Workload Identity)

GKE 파드가 GCS에 안전하게 접근할 수 있도록 Workload Identity를 설정합니다.

#### 1.1. IAM 서비스 계정 생성

```bash
# 변수 설정 (본인의 환경에 맞게 수정하세요)
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

# KSA에 Workload Identity 어노테이션 추가
kubectl annotate serviceaccount ${K8S_SA_NAME} \
    --namespace ${NAMESPACE} \
    iam.gke.io/gcp-service-account=${IAM_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com
```

### 2. Hugging Face 액세스 토큰 Secret 생성

Hugging Face 토큰을 Kubernetes Secret으로 안전하게 저장합니다.

```bash
kubectl create secret generic huggingface-secret \
  --from-literal=token='<YOUR_HUGGING_FACE_TOKEN>'
```

### 3. Docker 이미지 빌드 및 푸시

제공된 `Dockerfile`을 사용하여 파인 튜닝 환경을 포함하는 Docker 이미지를 빌드하고, Google Artifact Registry에 푸시합니다.

#### 3.1. Artifact Registry 리포지토리 생성

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export REGION="<YOUR_GCP_REGION>" # 예: us-central1
export AR_REPO="llama3-finetune-repo"

gcloud artifacts repositories create ${AR_REPO} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Llama3 Finetuning Images"
```

#### 3.2. Docker 이미지 빌드 및 푸시

`push.sh` 스크립트는 이 과정을 자동화합니다. 스크립트 내의 변수를 자신의 환경에 맞게 수정하세요.

```bash
# push.sh (본인의 환경에 맞게 수정하세요)
export PROJECT_ID="<YOUR_PROJECT_ID>"
export REGION="<YOUR_GCP_REGION>"
export AR_REPO="llama3-finetune-repo"
export IMAGE_NAME="llama3-finetune"
export TAG="latest"

# 최종 이미지 URI
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${TAG}"

# Docker 이미지 빌드 및 푸시
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

스크립트를 실행합니다.

```bash
bash push.sh
```

---

## 🚀 Single Node Fine Tuning (H100 8 GPUs)

### 1. 파인 튜닝 작업 실행

`finetune-job.yaml` 파일을 사용하여 Kubernetes Job을 실행합니다. 이 파일은 싱글 `a3-megagpu-8g` 노드에서 8개의 GPU를 모두 사용하도록 설정되어 있습니다.

**참고:** `finetune-job.yaml` 파일 내의 `image` 경로를 위 단계에서 푸시한 본인의 Artifact Registry 이미지 경로로 수정해야 합니다.

```bash
kubectl apply -f finetune-job.yaml
```

### 2. 모니터링 및 결과 확인

#### 파드 상태 확인
```bash
kubectl get pods -l job-name=llama3-finetune-job -w
```

#### 로그 스트리밍
```bash
export POD_NAME=$(kubectl get pods -l job-name=llama3-finetune-job -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME
```

학습이 완료되면 `train_loss`와 함께 GCS 업로드 완료 메시지가 출력됩니다.

#### GCS 버킷 결과 확인
```bash
gsutil ls gs://<YOUR_GCS_BUCKET_NAME>/final_model/
```

---

## 🚀 Multi Node Fine Tuning (H100 16 GPUs)

이 설정은 2개의 `a3-megagpu-8g` 노드를 사용하여 총 16개의 GPU로 분산 학습을 수행합니다. PyTorch의 `torchrun`을 사용하여 분산 환경을 구성합니다.

### 1. GKE 노드 풀 확장

`a3-megagpu-8g`를 사용하는 노드 풀의 크기를 2로 확장합니다.

```bash
# 변수 설정
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export NODE_POOL_NAME="<YOUR_NODE_POOL_NAME>"
export REGION="<YOUR_GCP_REGION>"

# 노드 풀 크기 조정
gcloud container clusters resize ${CLUSTER_NAME} \
    --node-pool=${NODE_POOL_NAME} \
    --num-nodes=2 \
    --region=${REGION}
```

### 2. 멀티 노드 작업 실행

`finetune-job-multinode.yaml`은 `torchrun`을 사용하여 2개의 노드에서 분산 학습을 실행하도록 구성되어 있습니다.

**주요 설정:**
-   **Headless Service:** 파드 간의 안정적인 통신을 위해 `clusterIP: None`으로 설정된 헤드리스 서비스를 사용합니다.
-   **subdomain:** Pod spec에 `subdomain: llama3-finetune-job-headless`를 반드시 설정해야 합니다. 이 설정이 있어야 `{파드명}.{서비스명}.{네임스페이스}.svc.cluster.local` 형식의 파드별 DNS가 생성됩니다.
-   **Indexed Job:** `completionMode: Indexed`를 사용하여 각 파드에 고유한 인덱스(0 또는 1)를 부여합니다. `JOB_COMPLETION_INDEX` 환경 변수는 Kubernetes가 자동으로 주입하며, 이를 `torchrun`의 `--node_rank`로 사용합니다.
-   **MASTER_ADDR:** Headless Service DNS는 모든 파드 IP를 반환하므로 `torchrun`의 rendezvous에 불안정합니다. 반드시 rank-0 파드의 FQDN을 명시합니다:
    ```
    llama3-finetune-job-multinode-0.llama3-finetune-job-headless.default.svc.cluster.local
    ```
-   **torchrun:** `command` 섹션에서 `torchrun`을 직접 호출하여 `--nnodes=2`, `--nproc_per_node=8` 등의 분산 학습 파라미터를 명시적으로 설정합니다.
-   **NCCL 환경 변수:** `NCCL_SOCKET_IFNAME=eth0`를 설정하여 멀티 노드 통신에 사용할 네트워크 인터페이스를 지정합니다.

**참고:** `finetune-job-multinode.yaml` 파일 내의 `image` 경로를 본인의 Artifact Registry 이미지 경로로 수정해야 합니다.

```bash
kubectl apply -f finetune-job-multinode.yaml
```

### 3. 모니터링 및 결과 확인

#### 파드 상태 확인
```bash
kubectl get pods -l job-name=llama3-finetune-job-multinode -w
```

#### 각 파드 로그 스트리밍

두 파드의 로그를 각각 확인하여 학습이 동기화되어 진행되는지 확인할 수 있습니다.

```bash
# RANK 0 파드 로그
export POD_NAME_0=$(kubectl get pods -l job-name=llama3-finetune-job-multinode,batch.kubernetes.io/job-completion-index=0 -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME_0

# RANK 1 파드 로그
export POD_NAME_1=$(kubectl get pods -l job-name=llama3-finetune-job-multinode,batch.kubernetes.io/job-completion-index=1 -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME_1
```

학습이 성공적으로 완료되면, RANK 0 파드에서 모델을 GCS에 업로드하는 로그를 확인할 수 있습니다.

---

## 파일 설명

-   **`Dockerfile`**: 파인 튜닝 환경을 위한 Docker 이미지를 빌드하는 파일입니다.
-   **`finetune-job.yaml`**: 싱글 노드 GKE 파인 튜닝 Job을 위한 Kubernetes 명세 파일입니다.
-   **`finetune-job-multinode.yaml`**: 멀티 노드 분산 학습을 위한 Kubernetes Service 및 Job 명세 파일입니다.
-   **`scripts/finetune.py`**: 실제 모델 로드, 데이터 처리, 파인 튜닝 및 GCS 업로드를 수행하는 Python 스크립트입니다.
-   **`push.sh`**: `Dockerfile`을 빌드하고 Google Artifact Registry에 이미지를 푸시하는 과정을 자동화하는 셸 스크립트입니다. 실행 전 `PROJECT_ID`와 `REGION` 변수를 반드시 설정하세요.
