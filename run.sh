#!/bin/bash
killall python3 -9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8 \
# CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUBLAS_WORKSPACE_CONFIG=:16:8 \
python3 ./autoRun.py
# python3 ./autoRun.py
