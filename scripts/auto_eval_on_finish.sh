#!/usr/bin/env bash
# Waits for training to finish, then evaluates the final model on GPU.
# Emits status lines (each becomes a notification). Safe to run beside training:
# it only touches the GPU AFTER the training process has exited and freed VRAM.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid

echo "[auto-eval] armed $(date '+%F %T') — waiting for training to finish (final/ or process exit)"

# 1) wait for training to finish
while true; do
  if [ -d outputs/run/final ]; then echo "[auto-eval] final/ model detected"; break; fi
  if ! pgrep -f "torchrun --nproc_per_node=2 scripts/train_mid.py" >/dev/null 2>&1; then
    echo "[auto-eval] training process exited (no final/ yet) — will eval latest checkpoint"; break
  fi
  sleep 300
done

# 2) let the training process fully release GPU memory
sleep 90

# 3) pick eval target: prefer final/, else the highest-numbered checkpoint
TARGET=outputs/run/final
if [ ! -d "$TARGET" ]; then
  TARGET=$(ls -d outputs/run/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
echo "[auto-eval] target = $TARGET"
if [ ! -f "$TARGET/alignment_state.pt" ]; then
  echo "[auto-eval] WARNING: no alignment_state.pt in $TARGET — eval would use raw CLIP text (invalid)."
fi

# 4) run evaluation on GPU (both GPUs now free)
source venv_mid/bin/activate 2>/dev/null
echo "[auto-eval] starting evaluation $(date '+%F %T') ..."
python scripts/evaluate.py \
  --checkpoint "$TARGET" \
  --benchmarks imagenet flickr30k cifar100 \
  --device cuda > outputs/eval_final.log 2>&1
rc=$?
echo "[auto-eval] evaluation finished (rc=$rc) $(date '+%F %T')"
echo "[auto-eval] ===== RESULTS ====="
grep -E "Top-1 Accuracy|Top-5 Accuracy|Image→Text|Text→Image|CIFAR-100 Linear|Top-1 Accuracy:" outputs/eval_final.log | tail -12
echo "[auto-eval] full log: outputs/eval_final.log ; json: $TARGET/eval_results.json"
