"""
Build the Stage 2b bilingual mix WITH short-answer VQA data.

Why this exists
---------------
Stage 2b (English) scored 1.73% strict on VQAv2 against a 48.60% lenient score.
The model answers "Yes, this is a bowl of rice noodle soup." where the benchmark
expects "yes" -- it was right and scored zero. The cause is the data: the mix
contained only verbose LLaVA-Instruct conversations and no short-answer VQA.
LLaVA-1.5 adds VQAv2/GQA/OKVQA train splits precisely to teach terseness.

The existing bilingual mix already carries 207,829 short-answer records -- but
207,821 of them are KOREAN and only 8 are English. So Korean already has the
format; English has none, which is exactly the language the VQAv2 result was
measured in. Adding more Korean short-answer would have deepened the imbalance
while leaving the measured failure untouched.

We therefore pull English short-answer records from llava_v1_5_mix665k.json (the
English LLaVA-1.5 mix). Both mixes reference gqa/, vg/, ocr_vqa/ and textvqa/
image directories that are NOT on this disk -- only coco/ is present -- so we
take only COCO-image records (198,473 English available) and skip the rest
rather than silently training on records whose images never load.

Output keeps the existing mix's schema exactly: {id, image, lang, conversations}
with `image` relative to DATA_ROOT.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

DATA = "/home/work/keural-model/vlm-datasets"
BASE = f"{DATA}/mixed_en_ko_instruct.jsonl"
LLAVA_EN = f"{DATA}/llava-instruct/llava_v1_5_mix665k.json"
OUT = f"{DATA}/mixed_en_ko_instruct_v2.jsonl"

SHORT_MAX_CHARS = 30      # mean assistant answer length that counts as "short"
TARGET_EN_FRAC = 0.40     # raise English from 22% toward parity
SEED = 42


def assistant_texts(rec):
    msgs = rec.get("conversations") or rec.get("messages") or []
    return [m.get("value") or m.get("content") or "" for m in msgs
            if m.get("from") in ("gpt", "assistant") or m.get("role") == "assistant"]


def is_short(rec):
    a = assistant_texts(rec)
    return bool(a) and sum(len(x) for x in a) / len(a) < SHORT_MAX_CHARS


def main():
    rng = random.Random(SEED)

    base = [json.loads(l) for l in open(BASE) if l.strip()]
    print(f"base mix                     : {len(base):,} records")

    base_short_by_lang = Counter(r.get("lang") for r in base if is_short(r))
    print(f"  already short-answer       : {dict(base_short_by_lang)}  "
          "<- English has essentially none; that is the gap")

    pool = json.load(open(LLAVA_EN))
    short_coco, skipped_missing_root = [], Counter()
    for r in pool:
        img = r.get("image") or ""
        if not img:
            skipped_missing_root["<text-only>"] += 1
            continue
        if not img.startswith("coco/"):
            skipped_missing_root[img.split("/")[0]] += 1
            continue
        if is_short(r):
            short_coco.append(r)
    print(f"English short-answer (COCO)  : {len(short_coco):,} usable")
    for root, n in skipped_missing_root.most_common():
        print(f"  SKIPPED {n:>7,} -> {root} images not on this disk")

    # Add English until English reaches TARGET_EN_FRAC of the total.
    #   (en + n) / (total + n) = f  =>  n = (f*total - en) / (1 - f)
    en_now = sum(1 for r in base if r.get("lang") == "en")
    want = int((TARGET_EN_FRAC * len(base) - en_now) / (1 - TARGET_EN_FRAC))
    take = min(max(want, 0), len(short_coco))
    if take < want:
        print(f"  NOTE: wanted {want:,}, only {take:,} available")
    add = rng.sample(short_coco, take)

    # Verify every image resolves before writing. A record whose image is missing
    # is silently dropped at train time by the loader, which quietly shrinks the
    # dataset and skews the language ratio without any error.
    missing = 0
    normalized = []
    for r in add:
        if not os.path.exists(os.path.join(DATA, r["image"])):
            missing += 1
            continue
        normalized.append({
            "id": str(r.get("id", "")),
            "image": r["image"],
            "lang": "en",
            "conversations": r["conversations"],
        })
    if missing:
        print(f"  dropped {missing:,} whose image did not resolve on disk")

    merged = base + normalized
    rng.shuffle(merged)

    with open(OUT, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    langs = Counter(r.get("lang", "?") for r in merged)
    shorts = sum(1 for r in merged if is_short(r))
    print(f"\nwrote {OUT}")
    print(f"  total          : {len(merged):,}")
    print(f"  language       : " + "  ".join(
        f"{k} {v/len(merged)*100:.0f}%" for k, v in langs.most_common()))
    print(f"  short-answer   : {shorts:,} ({shorts/len(merged)*100:.1f}%)")
    sl = Counter(r.get("lang") for r in merged if is_short(r))
    print(f"  short by lang  : " + "  ".join(f"{k} {v:,}" for k, v in sl.most_common()))
    print(f"  before         : {len(base):,} records, "
          f"en {en_now/len(base)*100:.0f}%, "
          f"english short-answer {base_short_by_lang.get('en', 0):,}")


if __name__ == "__main__":
    sys.exit(main())
