"""
Prepare an OCR training mix for Stage 2c.

Downloads TextVQA (scene text) and writes images + a LLaVA-format JSONL under
DATA_ROOT, then blends in general instruction data so the model does not forget
everything else while learning to read.

Why TextVQA and not SynthDoG
----------------------------
SynthDoG-en + SynthDoG-ko would be the stronger choice for document OCR and give
Korean coverage, but they are ~77 GB together and this disk has 184 GB free at
94% used. TextVQA is 7.5 GB and answers the question that actually matters right
now: can the FROZEN encoder represent glyphs at all? If Stage 2c moves the
needle, SynthDoG is the obvious follow-up on a machine with room.

The replay mixture
------------------
Training purely on OCR data would produce a model that reads text and has
forgotten how to describe images -- the standard catastrophic-forgetting result.
REPLAY_RATIO keeps general instruction data in the mix so the existing
capability survives.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATA = Path("/home/work/keural-model/vlm-datasets")
OUT_JSONL = DATA / "ocr_mix.jsonl"
IMG_DIR = DATA / "textvqa_images"
GENERAL = DATA / "mixed_en_ko_instruct_v2.jsonl"

REPLAY_RATIO = 1.0     # one general sample per OCR sample
MAX_OCR = 30000        # enough signal for 4k steps at effective batch 64
SEED = 42


def main() -> int:
    from datasets import load_dataset

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("Loading lmms-lab/textvqa (downloads ~7.5 GB on first run)…", flush=True)
    ds = load_dataset("lmms-lab/textvqa", split="train")
    print(f"  {len(ds):,} records", flush=True)

    records, skipped = [], 0
    n = min(MAX_OCR, len(ds))
    for i in range(n):
        it = ds[i]
        answers = it.get("answers") or []
        answers = [a["answer"] if isinstance(a, dict) else a for a in answers]
        answers = [a for a in answers if a and a.strip()]
        if not answers:
            skipped += 1
            continue
        # Majority answer — TextVQA gives 10 annotations per question.
        gold = max(set(answers), key=answers.count).strip()

        stem = f"textvqa_{it.get('question_id', i)}.jpg"
        p = IMG_DIR / stem
        if not p.exists():
            try:
                it["image"].convert("RGB").save(p, quality=92)
            except Exception:
                skipped += 1
                continue

        records.append({
            "id": str(it.get("question_id", i)),
            "image": f"textvqa_images/{stem}",
            "lang": "en",
            "conversations": [
                {"from": "human",
                 "value": "<image>\n" + it["question"].strip() +
                          "\nAnswer the question using a single word or phrase."},
                {"from": "gpt", "value": gold},
            ],
        })
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{n:,} written ({skipped} skipped)", flush=True)

    print(f"OCR records: {len(records):,}  (skipped {skipped:,})", flush=True)
    if not records:
        print("ERROR: no OCR records produced — not writing a mix", flush=True)
        return 1

    # Replay: keep general capability alive.
    general = []
    if GENERAL.exists():
        with open(GENERAL) as f:
            general = [l for l in f if l.strip()]
        take = min(int(len(records) * REPLAY_RATIO), len(general))
        general = [json.loads(l) for l in rng.sample(general, take)]
        print(f"replay records: {len(general):,} from {GENERAL.name}", flush=True)
    else:
        print(f"WARNING: {GENERAL} missing — OCR-only mix risks forgetting", flush=True)

    merged = records + general
    rng.shuffle(merged)
    with open(OUT_JSONL, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_JSONL}: {len(merged):,} records "
          f"({len(records):,} OCR + {len(general):,} replay)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
