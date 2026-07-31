#!/usr/bin/env bash
# Runs after gpu_queue.sh finishes. Waits for the GPU to free, then runs the
# evaluations the first queue got wrong or skipped.
#
# The first queue ran MME with --limit 500, which truncated the dataset to the
# first 3 of its 14 categories (artwork/celebrity/code_reasoning) and produced a
# "score" of 202.0 that is NOT a valid MME score. MME must be run whole.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid
source venv_mid/bin/activate 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=outputs/queue; mkdir -p "$Q"
LOG="$Q/followup.log"
B=outputs/projector_2b_bilingual/checkpoint-12000
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for the main queue to finish…"
while pgrep -f "gpu_queue.sh" >/dev/null; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')" -gt 20000 ]; do sleep 120; done
say "GPU free — starting follow-up evaluations"

say "START mme_full (all 2374 samples, 14 categories)"
python3 scripts/eval_vlm.py --benchmark mme \
  --projector-ckpt "$B/projector.pt" --lora-ckpt "$B/lora_adapter" \
  --output "$Q/mme_full.json" >> "$Q/mme_full.log" 2>&1
say "DONE  mme_full (rc=$?)"

say "START mmbench_full"
python3 scripts/eval_vlm.py --benchmark mmbench \
  --projector-ckpt "$B/projector.pt" --lora-ckpt "$B/lora_adapter" \
  --output "$Q/mmbench_full.json" >> "$Q/mmbench_full.log" 2>&1
say "DONE  mmbench_full (rc=$?)"

say "START mme_blind (isolates what vision contributes on MME)"
python3 scripts/eval_vlm.py --benchmark mme --no-vision \
  --projector-ckpt "$B/projector.pt" --lora-ckpt "$B/lora_adapter" \
  --output "$Q/mme_full_blind.json" >> "$Q/mme_blind.log" 2>&1
say "DONE  mme_blind (rc=$?)"
say "follow-up finished"
