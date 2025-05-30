#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <python_script> [gpu_id]"
    exit 1
fi

SCRIPT=$1
GPU_ID=$2

if [ -n "$GPU_ID" ]; then
    echo "Using specified GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT
    exit 0
fi

echo "Waiting for a free GPU..."

while true; do
    FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1 > 10000 {print NR-1; exit}')
    if [ ! -z "$FREE_GPU" ]; then
        echo "Using GPU $FREE_GPU"
        CUDA_VISIBLE_DEVICES=$FREE_GPU python $SCRIPT
        break
    fi
    sleep 120
done