# llama3-finetune-on-gke

## 프로젝트 개요

GKE의 `a3-megagpu-8g` 노드(NVIDIA H100 8장)에서 `meta-llama/Meta-Llama-3-8B-Instruct` 모델을
`databricks/databricks-dolly-15k` 데이터셋으로 파인튜닝하는 파이프라인.
싱글 노드(8 GPU)와 멀티 노드(16 GPU, torchrun) 두 가지 설정을 모두 지원한다.

## 아키텍처

```
push.sh                         # 이미지 빌드 → Artifact Registry 푸시
└── Dockerfile                  # nvcr.io/nvidia/pytorch:24.05-py3 기반

finetune-job.yaml               # 싱글 노드 Kubernetes Job (8 GPU)
finetune-job-multinode.yaml     # 멀티 노드 Job + Headless Service (16 GPU)
└── scripts/finetune.py         # 실제 학습 로직 (LoRA + SFTTrainer + GCS 업로드)
```

## 학습 설정

- **기법**: LoRA (r=64, alpha=16, target: q/k/v/o/gate/up/down_proj)
- **dtype**: bfloat16 (양자화 없음)
- **옵티마이저**: paged_adamw_32bit (싱글노드 전용; 멀티노드는 adamw_torch 권장)
- **데이터**: dolly-15k 중 1,000 샘플, max_seq_length=512
- **저장**: `GCS_BUCKET_NAME` 환경변수로 지정된 버킷의 `final_model/` prefix

## 인증

- Workload Identity로 GCS 접근 (IAM SA `llama3-finetuner-sa` ↔ KSA `llama3-finetuner-ksa`)
- HuggingFace 토큰은 Kubernetes Secret `huggingface-secret`에서 주입

## 멀티노드 분산 학습 핵심 사항

- Headless Service(`llama3-finetune-job-headless`)로 파드 간 DNS 통신
- Pod spec에 `subdomain: llama3-finetune-job-headless` **필수** — 없으면 파드별 DNS 미생성
- `MASTER_ADDR`은 반드시 rank-0 파드 DNS를 명시해야 함:
  ```
  llama3-finetune-job-multinode-0.llama3-finetune-job-headless.default.svc.cluster.local
  ```
- Indexed Job은 `JOB_COMPLETION_INDEX`를 자동 주입 → 수동 fieldRef 불필요

## 알려진 패턴 및 주의사항

1. `SFTTrainer`에 `peft_config`를 넘기면 내부적으로 LoRA 적용 → `get_peft_model()` 별도 호출 금지 (이중 적용됨)
2. `Accelerator()`는 프로세스당 한 번만 인스턴스화 → 함수 내부 재생성 금지, 인자로 전달할 것
3. `push.sh` 실행 전 `PROJECT_ID`, `REGION` 변수 설정 필수 (`TAG`는 기본값 `latest`)
4. dolly-15k의 `context` 필드는 프롬프트 포맷에 포함시킬 것 (closed_qa, summarization 등에서 핵심 정보)
5. LoRA + 멀티노드 DDP 조합 시 `ddp_find_unused_parameters=False` 필수 (frozen params 때문)

## 자주 쓰는 커맨드

```bash
# 이미지 빌드/푸시 (PROJECT_ID, REGION 설정 후)
bash push.sh

# 싱글노드 학습
kubectl apply -f finetune-job.yaml
kubectl logs -f $(kubectl get pods -l job-name=llama3-finetune-job -o jsonpath='{.items[0].metadata.name}')

# 멀티노드 학습
kubectl apply -f finetune-job-multinode.yaml
kubectl get pods -l job-name=llama3-finetune-job-multinode -w

# 결과 확인
gsutil ls gs://<YOUR_GCS_BUCKET_NAME>/final_model/
```
