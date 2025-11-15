# 1. 기본 이미지: PyTorch, CUDA 12.x, cuDNN, NCCL 포함
# 'latest' 대신 특정 태그를 사용하여 재현성을 보장합니다. 
FROM nvcr.io/nvidia/pytorch:24.05-py3

# 2. pip 루트 사용자 경고 비활성화
ENV PIP_ROOT_USER_ACTION=ignore

# 3. pip 업그레이드 및 핵심 ML 라이브러리 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade typing_extensions && \
    pip install --no-cache-dir \
      transformers==4.42.3 \
      accelerate==0.31.0 \
      datasets==2.19.1 \
      peft==0.11.1 \
      trl==0.9.4 \
      bitsandbytes==0.43.1 \
      llama-recipes \
      gcsfs \
      google-cloud-storage
      
      # 4. 학습 스크립트 및 구성 파일을 컨테이너에 복사
      WORKDIR /app
      COPY ./scripts /app/scripts
      
      # 5. 기본 진입점 설정
ENTRYPOINT ["/bin/bash"]
