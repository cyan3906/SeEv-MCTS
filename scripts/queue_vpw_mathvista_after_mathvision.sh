#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?VPW MathVision controller PID is required}"
repo_root="/home/jncsnlp4/SSD2/ZJJ/VReST"
python_bin="/home/jncsnlp4/anaconda3/envs/rstar_zjj/bin/python"
output_dir="$repo_root/newoutput/mctsv8_vpw/mathvista"

echo "[$(date --iso-8601=seconds)] queued behind VPW MathVision PID $wait_pid"
while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
done

echo "[$(date --iso-8601=seconds)] VPW MathVision exited; waiting for GPU 0"
while true; do
    memory_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$memory_used" -le 1024 ]]; then
        break
    fi
    sleep 30
done

resume_args=()
if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    resume_args+=(--resume)
fi

echo "[$(date --iso-8601=seconds)] starting mctsv8_vpw MathVista on GPU 0"
cd "$repo_root"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$python_bin" scripts/run_resumable_inference.py \
    --repo-root="$repo_root" \
    --dataset=mathvista \
    --input-file="$repo_root/data/MathVista/data/testmini.json" \
    --method-name=mctsv8_vpw \
    --prompt-method=mctsv8_vpw \
    --rollout=8 \
    --output-dir="$output_dir" \
    "${resume_args[@]}"
