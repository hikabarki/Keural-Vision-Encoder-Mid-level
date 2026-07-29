#!/usr/bin/env bash
# Stage 2a — projector warm-up on caption data (LLaVA-Pretrain 558K).
#
# Trains the LevelAwareProjector to map Keural Mid's 768-dim visual tokens into
# Qwen2.5-7B's embedding space. LLM is frozen except LoRA adapters.
#
# Run this FIRST, before stage 2b. Expect the loss to drop quickly early on —
# that's the projector finding the embedding space. Check outputs/projector_2a/
# training_log.jsonl.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid

DATA=/home/work/keural-model/vlm-datasets
OUT=outputs/projector_2a

# Pre-flight: don't launch onto busy GPUs.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
if [ "$USED" -gt 20000 ]; then
  echo "ABORT: GPUs still in use (${USED} MiB allocated). Wait for the other job to finish."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

source venv_mid/bin/activate 2>/dev/null

# NOTE: --no-4bit uses bf16 + LoRA. bitsandbytes 4-bit is broken in this env
# (CUDA binary mismatch) and unnecessary on H200s for a 7B model.
#
# LLaVA-Pretrain ships as JSON + loose JPEGs (NOT WebDataset tar shards), and
# its records use the same {image, conversations} schema as LLaVA-Instruct —
# so it goes through llava-json mode, not caption-tar.
nohup python3 scripts/train_projector.py \
  --vision-ckpt outputs/run/final \
  --llm-name Qwen/Qwen2.5-7B-Instruct \
  --no-4bit \
  --data-mode llava-json \
  --jsonl-path "$DATA/llava-pretrain/pretrain_558k.jsonl" \
  --image-root "$DATA/llava-pretrain/images" \
  --output-dir "$OUT" \
  --token-budget 512 \
  --image-size 512 \
  --batch-size 16 \
  --grad-accum 2 \
  --lr 1e-3 \
  --warmup-steps 200 \
  --max-steps 5000 \
  --save-every 1000 \
  --log-every 10 \
  > outputs/stage2a.log 2>&1 &

echo "Stage 2a launched (pid $!). Follow with:"
echo "  tail -f outputs/stage2a.log"
