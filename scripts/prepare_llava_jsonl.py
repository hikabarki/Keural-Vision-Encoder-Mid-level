"""
Convert LLaVA-Instruct JSON -> JSONL for data/vlm_dataset.llava_json_stream.

The released LLaVA files are a single giant JSON array of records shaped:
    {"id", "image": "000000033471.jpg", "conversations": [{"from","value"}, ...]}

llava_json_stream() reads line-delimited JSON, so this flattens the array to
JSONL. It also (a) drops records whose image is missing on disk, and (b) drops
records with no assistant turn — both would otherwise be silently skipped at
training time, so counting them here makes the real usable size visible.

Usage:
    python3 scripts/prepare_llava_jsonl.py \
        --input  /home/work/keural-model/vlm-datasets/llava-instruct/llava_instruct_150k.json \
        --image-root /home/work/keural-model/vlm-datasets/coco/train2017 \
        --output /home/work/keural-model/vlm-datasets/llava-instruct/train_150k.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="LLaVA JSON -> JSONL prep")
parser.add_argument("--input", required=True, help="LLaVA-Instruct .json file")
parser.add_argument("--image-root", required=True, help="Dir holding the referenced images")
parser.add_argument("--output", required=True, help="Destination .jsonl")
parser.add_argument("--val-split", type=int, default=1000,
                    help="Hold out this many records as a val set (0 to disable)")
parser.add_argument("--check-images", action="store_true", default=True,
                    help="Verify each referenced image exists on disk")
args = parser.parse_args()

image_root = Path(args.image_root)
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading {args.input} ...")
with open(args.input) as f:
    records = json.load(f)
print(f"  {len(records):,} records")

kept, missing_image, no_assistant, no_image_field = [], 0, 0, 0
for rec in records:
    img = rec.get("image")
    if not img:
        no_image_field += 1
        continue
    convs = rec.get("conversations") or []
    if not any(c.get("from") == "gpt" for c in convs):
        no_assistant += 1
        continue
    if args.check_images and not (image_root / img).exists():
        missing_image += 1
        continue
    kept.append(rec)

print(f"\nFiltering results:")
print(f"  kept              : {len(kept):,}")
print(f"  dropped (no image field)   : {no_image_field:,}")
print(f"  dropped (no assistant turn): {no_assistant:,}")
print(f"  dropped (image not on disk): {missing_image:,}")

if not kept:
    raise SystemExit("ERROR: no usable records — check --image-root points at the images.")

val_n = min(args.val_split, max(0, len(kept) // 20))
val, train = (kept[:val_n], kept[val_n:]) if val_n else ([], kept)


def dump(recs, path):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(recs):,} -> {path}")


print("\nWriting:")
dump(train, out_path)
if val:
    dump(val, out_path.with_name(out_path.stem + "_val.jsonl"))

print(f"\nDone. Train with:")
print(f"  --data-mode llava-json --jsonl-path {out_path} --image-root {image_root}")
