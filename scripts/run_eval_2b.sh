#!/usr/bin/env bash
# Stage 2b evaluation — POPE (hallucination) + the VQAv2 sighted/blind pair.
#
# Runs SEQUENTIALLY on purpose. Each benchmark loads a full Qwen2.5-7B into host
# RAM before it reaches the GPU; running them concurrently is how you rediscover
# the 352 GiB cgroup ceiling that SIGKILLed Stage 2a.
#
# The number that matters is not any single score but the GAP:
#     vqav2_sighted - vqav2_blind  =  what the vision encoder actually contributes
# A blind Qwen2.5-7B scores ~40% on VQAv2 from language priors alone.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid

CKPT=${CKPT:-outputs/projector_2b/checkpoint-8000}
LIMIT_POPE=${LIMIT_POPE:-500}
LIMIT_VQA=${LIMIT_VQA:-1000}
RES=outputs/eval_2b
mkdir -p "$RES"

if [ ! -f "$CKPT/projector.pt" ]; then
  echo "ABORT: $CKPT/projector.pt not found"; exit 1
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$USED" -gt 20000 ]; then
  echo "ABORT: GPUs busy (${USED} MiB). Wait for the other job."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

source venv_mid/bin/activate 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # run <label> <output-json> <extra args...>
  local label=$1 out=$2; shift 2
  echo ""
  echo "============================================================"
  echo "  $label   ($(date '+%H:%M:%S'))"
  echo "============================================================"
  python3 scripts/eval_vlm.py \
    --projector-ckpt "$CKPT/projector.pt" \
    --lora-ckpt      "$CKPT/lora_adapter" \
    --output "$out" "$@"
  echo "--- $label exit=$? ---"
}

echo "Stage 2b evaluation"
echo "  checkpoint : $CKPT"
echo "  started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NOTE: POPE is not in the HF cache; the first run downloads it (~1-2 GB)."

# 1. Hallucination. Read the YES-RATIO, not just accuracy: a model that answers
#    "yes" to everything scores ~50% accuracy while being useless.
run "POPE (hallucination)" "$RES/pope.json" --benchmark pope --limit "$LIMIT_POPE"

# 2/3. The diagnostic pair. Neither number means much alone.
run "VQAv2 SIGHTED" "$RES/vqav2_sighted.json" --benchmark vqav2 --limit "$LIMIT_VQA"
run "VQAv2 BLIND (no vision)" "$RES/vqav2_blind.json" --benchmark vqav2 --limit "$LIMIT_VQA" --no-vision

echo ""
echo "============================================================"
echo "  SUMMARY   ($(date '+%H:%M:%S'))"
echo "============================================================"
python3 - <<'PY'
import json, os
R = "outputs/eval_2b"
def get(f):
    p = os.path.join(R, f)
    try:
        return json.load(open(p))
    except Exception:
        return None
pope, s, b = get("pope.json"), get("vqav2_sighted.json"), get("vqav2_blind.json")
if pope:
    print(f"POPE   accuracy {pope.get('accuracy',0)*100:.2f}%   F1 {pope.get('f1',0)*100:.2f}%")
    yr = pope.get('yes_ratio')
    if yr is not None:
        flag = "  <-- answers yes to nearly everything" if yr > 0.75 else ""
        print(f"       yes-ratio {yr:.3f}{flag}")
if s and b:
    sa, ba = s.get('accuracy',0)*100, b.get('accuracy',0)*100
    print(f"\nVQAv2  sighted {sa:.2f}%   blind {ba:.2f}%")
    print(f"       GAP {sa-ba:+.2f} points  <-- what the vision encoder contributes")
    if sa - ba < 5:
        print("       WARNING: gap under 5 points. The encoder is barely helping.")
print("\nJSON results in outputs/eval_2b/")
PY
echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
