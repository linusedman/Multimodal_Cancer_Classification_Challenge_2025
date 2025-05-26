#!/bin/bash

echo "Waiting for a free GPU..."

while true; do
    FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1 > 10000 {print NR-1; exit}')
    if [ ! -z "$FREE_GPU" ]; then
        echo "Using GPU $FREE_GPU"
        CUDA_VISIBLE_DEVICES=$FREE_GPU python train.py
        break
    fi
    sleep 120
done