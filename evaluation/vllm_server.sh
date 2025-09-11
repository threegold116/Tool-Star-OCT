#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate toolstar
CUDA_VISIBLE_DEVICES=0 vllm serve /home/sxjiang/model/Qwen2.5-72B-Instruct-GPTQ-Int4 \
    --served-model-name Qwen2.5-72B-Instruct-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --uvicorn-log-level debug \
    --port 8888


