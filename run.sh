#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <script.py>"
    exit 1
fi

SCRIPT="$1"
MAX_GPUS=1        # Maximum number of GPUs you're willing to use
MEM_THRESHOLD=10000  # Minimum free memory in MiB

while true; do
    echo "[$(date '+%H:%M:%S')] Checking for free GPUs..."

    # Find all GPUs with enough free memory
    mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk -v threshold=$MEM_THRESHOLD '{ if ($1 > threshold) print NR-1 }')

    NUM_FREE=${#FREE_GPUS[@]}

    if (( NUM_FREE > 0 )); then
        # Use up to MAX_GPUS available
        SELECTED_GPUS=$(IFS=,; echo "${FREE_GPUS[*]:0:$MAX_GPUS}")
        echo "✅ Found $NUM_FREE free GPU(s), using: $SELECTED_GPUS"
        CUDA_VISIBLE_DEVICES=$SELECTED_GPUS python "$SCRIPT"
        break
    else
        echo "❌ No suitable GPU found. Sleeping for 120s..."
        sleep 120
    fi
done