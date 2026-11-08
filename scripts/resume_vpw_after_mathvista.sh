#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?baseline scheduler PID is required}"
repo_root="/home/jncsnlp4/SSD2/ZJJ/VReST"
python_bin="/home/jncsnlp4/anaconda3/envs/rstar_zjj/bin/python"
output_dir="$repo_root/newoutput/mctsv8_vpw/mathvision"

echo "[$(date --iso-8601=seconds)] queued behind baseline scheduler PID $wait_pid"
while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
done

echo "[$(date --iso-8601=seconds)] baseline scheduler exited; waiting for GPU 0"
while true; do
    memory_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$memory_used" -le 1024 ]]; then
        break
    fi
    sleep 30
done

echo "[$(date --iso-8601=seconds)] resuming mctsv8_vpw on GPU 0"
cd "$repo_root"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$python_bin" scripts/run_resumable_inference.py \
    --repo-root="$repo_root" \
    --dataset=mathvision \
    --input-file="$repo_root/data/MATH-V/data/testmini.jsonl" \
    --method-name=mctsv8_vpw \
    --prompt-method=mctsv8_vpw \
    --rollout=10 \
    --skip-question-ids \
        4 46 86 123 193 242 295 390 522 675 870 1137 1394 1893 3033 \
    --output-dir="$output_dir" \
    --resume
