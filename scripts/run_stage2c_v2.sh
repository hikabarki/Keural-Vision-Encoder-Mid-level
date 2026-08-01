#!/usr/bin/env bash
# Stage 2c v2 — OCR without the calibration regression.
#
# v1 taught OCR (MME OCR 110.0 -> 137.5) but broke binary-question calibration
# (POPE yes-ratio 0.503 -> 0.886, accuracy -11.3). Diagnosed cause: the mix
# carried only 22.4% yes/no records against the bilingual mix's 39.3%.
#
# Three single-variable corrections vs v1:
#   mix     ocr_mix_v2.jsonl  — 2:1 stratified replay, yes/no restored to 39.0%
#   steps   2,000 not 4,000   — v1 validation was flat from step 250
#   res     512 not 1024      — matched to what the projector was trained at;
#                               2.6 of v1's 5.2-point VQAv2 drop was resolution
#
# Warm-starts from the BILINGUAL checkpoint, not from v1, so this is a clean
# branch off the known-good model rather than a patch on a regressed one.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid
DATA=/home/work/keural-model/vlm-datasets
OUT=outputs/projector_2c_ocr_v2
BASE=outputs/projector_2b_bilingual/checkpoint-12000

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$USED" -gt 20000 ]; then
  echo "ABORT: GPUs busy (${USED} MiB)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1
fi
[ -s "$DATA/ocr_mix_v2.jsonl" ] || { echo "ABORT: ocr_mix_v2.jsonl missing"; exit 1; }

source venv_mid/bin/activate 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -s outputs/stage2c_v2.log ]; then
  mv outputs/stage2c_v2.log "outputs/stage2c_v2_$(date +%Y%m%d_%H%M%S).log"
fi

nohup torchrun --nproc_per_node=2 --master_port=29521 \
  scripts/train_projector.py \
  --vision-ckpt outputs/run/final --llm-name Qwen/Qwen2.5-7B-Instruct --no-4bit \
  --init-from "$BASE" \
  --data-mode llava-json \
  --jsonl-path "$DATA/ocr_mix_v2.jsonl" --image-root "$DATA" \
  --output-dir "$OUT" \
  --token-budget 512 --image-size 512 \
  --batch-size 8 --grad-accum 4 --seed 42 \
  --lr 1e-5 --warmup-steps 100 --max-steps 2000 \
  --save-every 500 --log-every 10 --max-caption-length 512 \
  --val-jsonl "en:$DATA/llava-instruct/train_150k_val_rooted.jsonl" \
  --val-image-root "$DATA" --val-every 250 --val-samples 64 \
  --loader-workers 32 \
  > outputs/stage2c_v2.log 2>&1 &

echo "Stage 2c v2 launched (pid $!). Follow with:"
echo "  tail -f outputs/stage2c_v2.log"
