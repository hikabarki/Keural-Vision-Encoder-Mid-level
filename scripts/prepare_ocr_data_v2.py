"""
Stage 2c v2 OCR mix — corrected for the calibration regression.

What went wrong in v1
---------------------
v1 taught OCR successfully (MME OCR 110.0 -> 137.5, +27.5) but destroyed
yes/no calibration: POPE yes-ratio moved 0.503 -> 0.886, i.e. the model began
answering "yes" to 89% of binary questions, and POPE accuracy fell 11.3 points.

The cause is a dilution of binary supervision. Measured over the two mixes:

    bilingual v2 (Stage 2b)   39.3% of records carry a yes/no answer
    OCR mix v1  (Stage 2c)    22.4%

TextVQA answers are almost never yes/no, so mixing it 1:1 with general data
halved the model's exposure to binary questions and it drifted to a default
"yes". Note the trap this created: POPE F1 was essentially UNCHANGED (70.59 ->
70.56) because F1 rewards recall, and a model that always says yes has perfect
recall. Only accuracy and the yes-ratio revealed the damage.

The fix
-------
Raise the replay ratio to 2:1 AND stratify it, so the final mix restores the
~39% yes/no proportion the model was calibrated on. Replay is drawn in two
strata - records that contain a yes/no answer, and records that do not - in
whatever proportion hits the target. This is a single-variable correction: the
OCR half is byte-identical to v1.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA = Path("/home/work/keural-model/vlm-datasets")
OCR_V1 = DATA / "ocr_mix.jsonl"                    # reuse: images already on disk
GENERAL = DATA / "mixed_en_ko_instruct_v2.jsonl"
OUT = DATA / "ocr_mix_v2.jsonl"

REPLAY_RATIO = 2.0        # general : OCR
TARGET_YESNO = 0.39       # match the bilingual mix the model was calibrated on
SEED = 42

YES = {"yes", "네", "예"}
NO = {"no", "아니요", "아니오"}


def assistant(rec):
    msgs = rec.get("conversations") or rec.get("messages") or []
    return [m.get("value") or m.get("content") or "" for m in msgs
            if m.get("from") in ("gpt", "assistant") or m.get("role") == "assistant"]


def is_yesno(rec):
    for a in assistant(rec):
        s = a.strip().lower().rstrip(".")
        if s in YES or s in NO:
            return True
    return False


def main() -> int:
    rng = random.Random(SEED)

    if not OCR_V1.exists():
        print(f"ERROR: {OCR_V1} missing — run prepare_ocr_data.py first")
        return 1

    # Recover the OCR half of v1 (v1 = OCR + 1:1 replay). OCR records are the
    # ones whose image lives under textvqa_images/.
    ocr = []
    for line in open(OCR_V1):
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("image", "")).startswith("textvqa_images/"):
            ocr.append(r)
    print(f"OCR records recovered from v1 : {len(ocr):,}")
    if not ocr:
        print("ERROR: no textvqa records found in v1 mix")
        return 1
    ocr_yesno = sum(1 for r in ocr if is_yesno(r))
    print(f"  of which yes/no             : {ocr_yesno:,} ({ocr_yesno/len(ocr)*100:.1f}%)")

    # Split the general pool into the two strata we need to sample from.
    yn_pool, other_pool = [], []
    with open(GENERAL) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (yn_pool if is_yesno(r) else other_pool).append(r)
    print(f"general pool                  : {len(yn_pool):,} yes/no + "
          f"{len(other_pool):,} other")

    n_replay = int(len(ocr) * REPLAY_RATIO)
    total = len(ocr) + n_replay
    # Solve for how many yes/no replay records land the FINAL mix on TARGET_YESNO.
    need_yn = int(round(TARGET_YESNO * total)) - ocr_yesno
    need_yn = max(0, min(need_yn, n_replay, len(yn_pool)))
    need_other = n_replay - need_yn
    if need_other > len(other_pool):
        need_other = len(other_pool)
    print(f"replay plan                   : {need_yn:,} yes/no + {need_other:,} other "
          f"= {need_yn + need_other:,} ({REPLAY_RATIO:g}:1)")

    replay = rng.sample(yn_pool, need_yn) + rng.sample(other_pool, need_other)
    merged = ocr + replay
    rng.shuffle(merged)

    with open(OUT, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    yn = sum(1 for r in merged if is_yesno(r))
    langs = Counter(r.get("lang", "?") for r in merged)
    print(f"\nwrote {OUT}")
    print(f"  total records   : {len(merged):,}")
    print(f"  yes/no coverage : {yn:,} ({yn/len(merged)*100:.1f}%)   "
          f"[v1 was 22.4%, bilingual 39.3%]")
    print(f"  language        : " + "  ".join(
        f"{k} {v/len(merged)*100:.0f}%" for k, v in langs.most_common()))
    print(f"  OCR fraction    : {len(ocr)/len(merged)*100:.1f}%")

    # Every OCR image must resolve or the loader drops it silently.
    missing = sum(1 for r in ocr[:500]
                  if not (DATA / r["image"]).exists())
    print(f"  image spot-check: {500 - missing}/500 OCR images resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
