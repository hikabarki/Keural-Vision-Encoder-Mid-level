#!/usr/bin/env bash
# Stage 2b (bilingual) — instruction tuning on mixed English + Korean data.
#
# Same as run_stage2b_instruct.sh but trains on a shuffled EN+KO mixture so the
# model answers in whichever language it is asked. Both datasets reference the
# SAME COCO train2017 images, so no extra image download is needed.
#
#   English : LLaVA-Instruct-150K   (156,712 records)
#   Korean  : KoLLaVA coco_ko       (361,953 records)
#
# NOTE the different --image-root vs. the English-only script: the Korean JSON
# stores paths as "coco/train2017/x.jpg" while the English one stores bare
# "x.jpg", so the merged file is normalized to the Korean convention below and
# the root is the dataset dir, not .../coco/train2017.
set -uo pipefail
cd /home/work/Hika/vision-encoder/mid

DATA=/home/work/keural-model/vlm-datasets
OUT=outputs/projector_2b_bilingual
MERGED=$DATA/mixed_en_ko_instruct.jsonl

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
if [ "$USED" -gt 20000 ]; then
  echo "ABORT: GPUs still in use (${USED} MiB allocated)."
  exit 1
fi

# Build the merged EN+KO file once (idempotent).
if [ ! -f "$MERGED" ]; then
  echo "Building bilingual mixture -> $MERGED"
  python3 - "$DATA" "$MERGED" <<'PY'
import json, random, sys
data_root, out = sys.argv[1], sys.argv[2]
rows = []
# English: image field is a bare filename -> prefix it to match the Korean convention
with open(f"{data_root}/llava-instruct/train_150k.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if not r["image"].startswith("coco/"):
            r["image"] = "coco/train2017/" + r["image"]
        r["lang"] = "en"
        rows.append(r)
n_en = len(rows)
# Korean: already "coco/train2017/..."
with open(f"{data_root}/kollava-instruct/train_ko_coco.jsonl") as f:
    for line in f:
        r = json.loads(line); r["lang"] = "ko"; rows.append(r)
n_ko = len(rows) - n_en
random.seed(42); random.shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  merged {n_en:,} EN + {n_ko:,} KO = {len(rows):,} records")
PY
fi

source venv_mid/bin/activate 2>/dev/null

nohup python3 scripts/train_projector.py \
  --vision-ckpt outputs/run/final \
  --llm-name Qwen/Qwen2.5-7B-Instruct \
  --no-4bit \
  --data-mode llava-json \
  --jsonl-path "$MERGED" \
  --image-root "$DATA" \
  --output-dir "$OUT" \
  --token-budget 512 \
  --image-size 384 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 2e-5 \
  --warmup-steps 200 \
  --max-steps 12000 \
  --max-caption-length 512 \
  --save-every 1000 \
  --log-every 10 \
  > outputs/stage2b_bilingual.log 2>&1 &

echo "Stage 2b (bilingual EN+KO) launched (pid $!). Follow with:"
echo "  tail -f outputs/stage2b_bilingual.log"
