#!/usr/bin/env bash
# Stage 2b — instruction tuning on LLaVA-Instruct-150K (COCO images).
#
# Run this AFTER stage 2a. It resumes from the 2a projector so the projector
# starts already aligned to Qwen's embedding space, then teaches the model to
# answer questions about images (not just caption them).
#
# Key difference from 2a: labels are masked to ASSISTANT turns only, so the
# model learns to answer, not to echo the question. (Handled in
# data/vlm_dataset.build_llava_batch — unit-tested.)
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid

DATA=/home/work/keural-model/vlm-datasets
OUT=outputs/projector_2b
STAGE2A=outputs/projector_2a

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$USED" -gt 20000 ]; then
  echo "ABORT: GPUs still in use (${USED} MiB allocated)."
  exit 1
fi

if [ ! -f "$DATA/llava-instruct/train_150k.jsonl" ]; then
  echo "ABORT: run scripts/prepare_llava_jsonl.py first (converts LLaVA JSON -> JSONL"
  echo "       and drops records whose COCO image is missing)."
  exit 1
fi

# Warn (don't block) if 2a hasn't been run — 2b works from scratch but converges
# much better starting from an aligned projector.
if [ ! -d "$STAGE2A" ]; then
  echo "WARNING: no $STAGE2A found — running 2b without the 2a warm-up."
  echo "         Recommended: run scripts/run_stage2a_caption.sh first."
  sleep 5
fi

source venv_mid/bin/activate 2>/dev/null

# Lower LR than 2a: the projector is already aligned, and LoRA is now learning
# instruction-following on top. Higher LR here tends to wreck 2a's alignment.
nohup python3 scripts/train_projector.py \
  --vision-ckpt outputs/run/final \
  --llm-name Qwen/Qwen2.5-7B-Instruct \
  --no-4bit \
  --data-mode llava-json \
  --jsonl-path "$DATA/llava-instruct/train_150k.jsonl" \
  --image-root "$DATA/coco/train2017" \
  --output-dir "$OUT" \
  --token-budget 512 \
  --image-size 512 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 2e-5 \
  --warmup-steps 200 \
  --max-steps 8000 \
  --max-caption-length 512 \
  --save-every 1000 \
  --log-every 10 \
  > outputs/stage2b.log 2>&1 &

echo "Stage 2b launched (pid $!). Follow with:"
echo "  tail -f outputs/stage2b.log"
