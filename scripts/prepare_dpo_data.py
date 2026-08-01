"""
Prepare RLHF-V preference pairs for Stage 3 (DPO).

openbmb/RLHF-V-Dataset carries 5,733 human-annotated (image, question, chosen,
rejected) tuples where the rejected response contains a hallucination that a
human corrected. It is the canonical multimodal preference set and targets
exactly the failure this model still has: fluent long-form output that invents
objects.

Images are bundled inside the dataset, so nothing beyond the 0.82 GB download is
needed. This writes them to disk plus a JSONL the DPO trainer reads.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATA = Path("/home/work/keural-model/vlm-datasets")
IMG_DIR = DATA / "rlhfv_images"
OUT = DATA / "dpo_rlhfv.jsonl"
VAL_OUT = DATA / "dpo_rlhfv_val.jsonl"
VAL_N = 200
SEED = 42


def main() -> int:
    from datasets import load_dataset

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openbmb/RLHF-V-Dataset", split="train")
    print(f"RLHF-V records: {len(ds):,}")

    rows, skipped = [], 0
    for i, it in enumerate(ds):
        try:
            t = json.loads(it["text"])
            q, chosen, rejected = t["question"], t["chosen"], t["rejected"]
        except Exception:
            skipped += 1
            continue
        if not (q and chosen and rejected) or chosen.strip() == rejected.strip():
            skipped += 1
            continue

        stem = f"rlhfv_{it.get('idx', i)}.jpg"
        p = IMG_DIR / stem
        if not p.exists():
            try:
                it["image"].convert("RGB").save(p, quality=92)
            except Exception:
                skipped += 1
                continue

        rows.append({
            "id": str(it.get("idx", i)),
            "image": f"rlhfv_images/{stem}",
            "question": q.strip(),
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(ds):,}", flush=True)

    if not rows:
        print("ERROR: no usable pairs")
        return 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    val, train = rows[:VAL_N], rows[VAL_N:]
    for path, part in ((OUT, train), (VAL_OUT, val)):
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cl = sum(len(r["chosen"]) for r in rows) / len(rows)
    rl = sum(len(r["rejected"]) for r in rows) / len(rows)
    print(f"\nwrote {OUT}       {len(train):,} pairs")
    print(f"wrote {VAL_OUT}   {len(val):,} pairs (held out)")
    print(f"  skipped              : {skipped:,}")
    print(f"  mean chosen length   : {cl:.0f} chars")
    print(f"  mean rejected length : {rl:.0f} chars")
    # If rejected were systematically longer, DPO could learn "shorter is better"
    # instead of "truthful is better" -- worth knowing before training.
    print(f"  length ratio         : {rl/cl:.3f}  "
          f"({'BALANCED' if 0.9 < rl/cl < 1.1 else 'SKEWED — DPO may learn length, not truth'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
