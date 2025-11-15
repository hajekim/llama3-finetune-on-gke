import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from google.cloud import storage
from accelerate import Accelerator

# =====================================================================================
# GCS 업로드 함수
# =====================================================================================
def upload_to_gcs(bucket_name, source_directory, destination_blob_prefix):
    """로컬 디렉터리의 내용을 GCS 버킷에 업로드합니다."""
    if Accelerator().is_main_process:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        for local_file in os.walk(source_directory):
            dirpath, _, filenames = local_file
            for filename in filenames:
                local_file_path = os.path.join(dirpath, filename)
                
                relative_path = os.path.relpath(local_file_path, source_directory)
                destination_blob_name = os.path.join(destination_blob_prefix, relative_path)
                
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(local_file_path)
                print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")
        print(f"Successfully uploaded model to gs://{bucket_name}/{destination_blob_prefix}")

# =====================================================================================
# 파인튜닝 스크립트 시작
# =====================================================================================

# Accelerator 초기화
accelerator = Accelerator()

# 1. 모델과 토크나이저 불러오기
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

# 2. 양자화 없이 bfloat16으로 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False

# 3. LoRA 설정
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)

# 4. 데이터셋 불러오기 및 전처리
dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train")
small_dataset = dataset.select(range(1000))

def format_instruction(sample):
	return [f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"]

# 5. 학습 인자 설정 (FSDP 설정 제거)
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
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# 6. SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=small_dataset,
    peft_config=lora_config,
    formatting_func=format_instruction,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# 7. 학습 시작
print(f"[{accelerator.process_index}] Starting training...")
trainer.train()
print(f"[{accelerator.process_index}] Training finished.")

# 8. 모델을 로컬에 저장 (메인 프로세스에서만 실행)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    local_model_dir = "./results/final_model"
    print(f"[{accelerator.process_index}] Saving model to {local_model_dir}...")
    trainer.save_model(local_model_dir)
    print(f"Model saved locally to {local_model_dir}")

    # 9. GCS에 모델 업로드
    # 환경 변수에서 GCS 버킷 이름을 읽어오고, 없으면 기본값을 사용합니다.
    gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME", "default-gcs-bucket-name")
    gcs_blob_prefix = "final_model"
    
    if gcs_bucket_name == "default-gcs-bucket-name":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: GCS_BUCKET_NAME env var not set.        !!!")
        print("!!! Skipping GCS upload.                           !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print(f"Attempting to upload to GCS bucket: {gcs_bucket_name}")
        upload_to_gcs(gcs_bucket_name, local_model_dir, gcs_blob_prefix)

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print("Fine-tuning and GCS upload complete!")