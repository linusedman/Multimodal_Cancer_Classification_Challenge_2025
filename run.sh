#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <script.py>"
    exit 1
fi

SCRIPT="$1"


while true; do
    echo "[$(date '+%H:%M:%S')] Checking for free GPU..."
    FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1 > 10000 {print NR-1; exit}')
    if [ ! -z "$FREE_GPU" ]; then
        echo "Using GPU $FREE_GPU"
        CUDA_VISIBLE_DEVICES=$FREE_GPU python "$SCRIPT"
        break
    fi
    sleep 120
done
