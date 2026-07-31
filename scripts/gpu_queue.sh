#!/usr/bin/env bash
# Sequential GPU job queue — keeps both H200s busy until the Sunday deadline.
#
# Jobs run back-to-back in priority order: certain value first, speculative
# last. A failing job logs and the queue continues, so one bad job cannot idle
# the GPU for hours.
#
#   1. Benchmark suite on the bilingual model   ~3h   CERTAIN value
#   2. Stage 2c OCR training (if data present)  ~8h   SPECULATIVE
#   3. Evaluate the OCR model                   ~1h
#   4. High-sample re-evaluation                ~4h   tightens the headline numbers
#
# Everything writes to outputs/queue/. Follow with:
#   tail -f outputs/queue/queue.log
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid
source venv_mid/bin/activate 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Q=outputs/queue
mkdir -p "$Q"
LOG="$Q/queue.log"
B=outputs/projector_2b_bilingual/checkpoint-12000
DATA=/home/work/keural-model/vlm-datasets

say () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

job () {          # job <name> <command...>
  local name=$1; shift
  say "START  $name"
  local t0=$SECONDS
  "$@" >> "$Q/$name.log" 2>&1
  local rc=$?
  local mins=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ]; then say "DONE   $name  (${mins}m)"
  else say "FAILED $name  (rc=$rc, ${mins}m) — see $Q/$name.log, continuing"; fi
}

evl () {          # evl <tag> <extra args...>
  local tag=$1; shift
  python3 scripts/eval_vlm.py \
    --projector-ckpt "$B/projector.pt" --lora-ckpt "$B/lora_adapter" \
    --output "$Q/$tag.json" "$@"
}

say "queue starting — deadline Sunday Aug 2"

# ---------------------------------------------------------------- 1. benchmarks
# MME and MMBench have never been run. POPE blind completes the
# vision-contribution story; heldout is the only Korean measurement that exists.
job mme       evl mme      --benchmark mme      --limit 500
job mmbench   evl mmbench  --benchmark mmbench  --limit 500
job pope_blind evl pope_blind --benchmark pope  --limit 500 --no-vision
job heldout   evl heldout  --benchmark heldout  --limit 500

# ------------------------------------------------------------------- 2/3. OCR
# Speculative. Runs only if the data-prep step actually produced a mix, so a
# failed or incomplete download skips training instead of wasting hours.
OCR_MIX="$DATA/ocr_mix.jsonl"
if [ -s "$OCR_MIX" ]; then
  N=$(wc -l < "$OCR_MIX")
  say "OCR mix present ($N records) — running Stage 2c"
  job stage2c torchrun --nproc_per_node=2 --master_port=29519 \
      scripts/train_projector.py \
      --vision-ckpt outputs/run/final --llm-name Qwen/Qwen2.5-7B-Instruct --no-4bit \
      --init-from "$B" \
      --data-mode llava-json --jsonl-path "$OCR_MIX" --image-root "$DATA" \
      --output-dir outputs/projector_2c_ocr \
      --token-budget 1024 --image-size 1024 \
      --batch-size 4 --grad-accum 8 --lr 1e-5 --warmup-steps 200 \
      --max-steps 4000 --save-every 1000 --log-every 10 --max-caption-length 512 \
      --val-jsonl "en:$DATA/llava-instruct/train_150k_val_rooted.jsonl" \
      --val-image-root "$DATA" --val-every 250 --val-samples 64 --loader-workers 32

  C2=outputs/projector_2c_ocr/checkpoint-4000
  if [ -f "$C2/projector.pt" ]; then
    job ocr_eval python3 scripts/eval_vlm.py --benchmark vqav2 --limit 1000 \
        --projector-ckpt "$C2/projector.pt" --lora-ckpt "$C2/lora_adapter" \
        --token-budget 1024 --image-size 1024 --output "$Q/vqav2_2c.json"
  fi
else
  say "SKIP Stage 2c — no $OCR_MIX (download/prep not finished)"
fi

# ------------------------------------------------- 4. tighten headline numbers
# n=500 gives roughly +/-4 points of sampling error. The numbers that go in a
# report deserve better, and this is free GPU time.
job pope_5k   evl pope_n2000   --benchmark pope  --limit 2000
job vqa_5k    evl vqav2_n5000  --benchmark vqav2 --limit 5000
job vqa_5k_bl evl vqav2_n5000_blind --benchmark vqav2 --limit 5000 --no-vision

say "queue finished"
