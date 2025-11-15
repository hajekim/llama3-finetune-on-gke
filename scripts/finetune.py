import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from google.cloud import storage

# =====================================================================================
# GCS 업로드 함수
# =====================================================================================
def upload_to_gcs(bucket_name, source_directory, destination_blob_prefix):
    """로컬 디렉터리의 내용을 GCS 버킷에 업로드합니다."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for local_file in os.walk(source_directory):
        dirpath, _, filenames = local_file
        for filename in filenames:
            local_file_path = os.path.join(dirpath, filename)
            
            # GCS 내의 상대 경로 생성
            relative_path = os.path.relpath(local_file_path, source_directory)
            destination_blob_name = os.path.join(destination_blob_prefix, relative_path)
            
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")

# =====================================================================================
# 파인튜닝 스크립트 시작
# =====================================================================================

# 1. 모델과 토크나이저 불러오기
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 토크나이저에 pad 토큰이 없는 경우 추가
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

# 2. BitsAndBytes를 사용한 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# 3. 양자화 설정으로 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # Accelerate가 GPU에 모델을 분산하도록 함
)
# 모델의 pad 토큰 ID 재설정 (필요한 경우)
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False # 학습 중에는 캐시 사용 안 함

# 4. LoRA 설정
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[ # Llama-3의 주의 레이어에 대한 일반적인 대상 모듈
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
# PEFT 모델 가져오기
model = get_peft_model(model, lora_config)

# 5. 데이터셋 불러오기 및 전처리
dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train")

# 학습을 더 빠르게 하기 위해 데이터셋의 작은 부분집합만 사용 (예: 1000개 샘플)
small_dataset = dataset.select(range(1000))

# 데이터셋 포맷팅 함수
def format_instruction(sample):
	return [f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"]

# 6. 학습 인자 설정
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # A100/H100 GPU에 권장
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# 7. SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=small_dataset,
    peft_config=lora_config,
    formatting_func=format_instruction, # 데이터셋 포맷팅 함수 제공
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# 8. 학습 시작
trainer.train()

# 9. 모델을 로컬에 저장
local_model_dir = "./results/final_model"
trainer.save_model(local_model_dir)
print(f"Model saved locally to {local_model_dir}")

# 10. GCS에 모델 업로드
gcs_bucket_name = "oreo-llama"
gcs_blob_prefix = "final_model"
upload_to_gcs(gcs_bucket_name, local_model_dir, gcs_blob_prefix)
print(f"Successfully uploaded model to gs://{gcs_bucket_name}/{gcs_blob_prefix}")


print("Fine-tuning and GCS upload complete!")