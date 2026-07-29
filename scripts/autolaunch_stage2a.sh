#!/usr/bin/env bash
# Unattended launcher for Stage 2a.
#
# Waits for the GPUs to free up (another team's job is expected to finish
# ~03:00), launches Stage 2a, then watches the first steps for early failure so
# a dead run is not discovered hours later.
#
# It deliberately does NOT chain into Stage 2b: 2b trains on top of 2a's
# projector, so if 2a OOMs or diverges, an auto-chained 2b would spend hours
# training on a broken foundation. Check 2a by eye first —
# scripts/demo_vlm.py, then run scripts/run_stage2b_bilingual.sh.
#
# Usage:
#   nohup bash scripts/autolaunch_stage2a.sh > outputs/autolaunch.log 2>&1 &
#   tail -f outputs/autolaunch.log
#
# Safe to start now and leave running overnight.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid

FREE_THRESHOLD_MIB=20000     # consider GPUs free below this total allocation
POLL_SECONDS=120
MAX_WAIT_HOURS=12            # give up rather than hang forever
HEALTH_CHECK_SECONDS=900     # how long to watch for early failure after launch

log() { echo "[$(date '+%F %T')] $*"; }

log "armed — waiting for GPUs to drop below ${FREE_THRESHOLD_MIB} MiB"
log "current: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"

deadline=$(( $(date +%s) + MAX_WAIT_HOURS*3600 ))
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ "$used" -lt "$FREE_THRESHOLD_MIB" ]; then
    log "GPUs free (${used} MiB allocated)"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "ABORT: still busy (${used} MiB) after ${MAX_WAIT_HOURS}h. Not launching."
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

# Let the previous job's memory actually be released before allocating.
log "waiting 90s for VRAM to be released..."
sleep 90

log "launching Stage 2a"
bash scripts/run_stage2a_caption.sh
sleep 60

if [ ! -f outputs/stage2a.log ]; then
  log "ERROR: outputs/stage2a.log was never created — launch failed."
  exit 1
fi

# ---- early-failure watch -----------------------------------------------------
# Catch the things that kill a run in the first minutes: OOM, a crash, or a
# NaN loss. Anything later is for the morning check.
log "watching for early failure (${HEALTH_CHECK_SECONDS}s)..."
end=$(( $(date +%s) + HEALTH_CHECK_SECONDS ))
while [ "$(date +%s)" -lt "$end" ]; do
  if grep -qiE "out of memory|CUDA error|Traceback" outputs/stage2a.log; then
    log "FAILED — error detected in outputs/stage2a.log:"
    grep -iE -m3 -A3 "out of memory|CUDA error|Traceback" outputs/stage2a.log | sed 's/^/    /'
    log "Most likely CUDA OOM. Retry with: --batch-size 8 --grad-accum 4 (see docs/TRAINING_GUIDE.md 4.1)"
    exit 1
  fi
  if grep -qE "loss=nan|loss=inf" outputs/stage2a.log; then
    log "FAILED — non-finite loss detected. Stopping so it is not discovered hours later."
    exit 1
  fi
  if ! pgrep -f "train_projector.py" > /dev/null; then
    log "FAILED — training process exited during the health window. Tail:"
    tail -20 outputs/stage2a.log | sed 's/^/    /'
    exit 1
  fi
  sleep 30
done

steps=$(grep -c "^\[train\] step=" outputs/stage2a.log 2>/dev/null || echo 0)
log "HEALTHY — running, ${steps} logged steps so far"
grep -E "^\[train\] step=" outputs/stage2a.log | tail -3 | sed 's/^/    /'
log ""
log "Stage 2a is running unattended. When you are back:"
log "  1) tail outputs/projector_2a/training_log.jsonl     # confirm loss fell"
log "  2) python3 scripts/demo_vlm.py \\"
log "       --projector-ckpt outputs/projector_2a/checkpoint-5000/projector.pt \\"
log "       --image <photo.jpg> --question 'Describe this image.'"
log "     After 2a it should CAPTION, not follow instructions. Gibberish => stop."
log "  3) bash scripts/run_stage2b_bilingual.sh"
