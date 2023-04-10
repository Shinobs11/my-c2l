#!/bin/bash
lsof -ti:29500 | xargs kill -9
killall python3 -9
killall python -9
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8 \
# CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUBLAS_WORKSPACE_CONFIG=:16:8 \
deepspeed --num_nodes 1 --num_gpus 1  ./autoRun.py --deepspeed_config ./src/configs/deepspeed_config.json

# python3 ./autoRun.py
