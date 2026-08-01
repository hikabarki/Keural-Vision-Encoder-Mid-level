"""
Update the encoder's Hugging Face card in place.

Three additions, no rewrites — the existing prose, figures and structure are
preserved:

  1. model-index metadata, so the Stage 1 benchmark numbers render as a metrics
     table on the model page instead of sitting in prose. This is the single
     thing that most distinguishes a card that reads as a serious release.
  2. A "Downstream Results" section. The VLM numbers are the strongest available
     evidence that this encoder works in practice: a frozen 183.6M encoder
     contributing +25.95 POPE and +446.9 MME over a blind baseline is a claim
     about the ENCODER, not about the decoder attached to it.
  3. A link to the VLM repository, so the two artifacts point at each other.

    python3 scripts/update_encoder_card.py --dry-run
    HF_TOKEN=... python3 scripts/update_encoder_card.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "mkd-hika/keural-vision-encoder-mid"
VLM = "mkd-hika/keural-mid-vlm-bilingual"
LOCAL = ROOT / "docs/HF_ENCODER_CARD_updated.md"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

from huggingface_hub import HfApi, hf_hub_download

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
txt = Path(hf_hub_download(REPO, "README.md", token=token)).read_text()
orig_len = len(txt)

# ---------------------------------------------------------------- 1. metadata
MODEL_INDEX = """model-index:
  - name: keural-vision-encoder-mid
    results:
      - task:
          type: zero-shot-image-classification
          name: Zero-Shot Image Classification
        dataset:
          type: imagenet-1k
          name: ImageNet-1k (validation)
        metrics:
          - type: accuracy
            value: 37.2
            name: Top-1 accuracy
          - type: accuracy
            value: 66.8
            name: Top-5 accuracy
      - task:
          type: image-to-text
          name: Image-to-Text Retrieval
        dataset:
          type: nlphuji/flickr30k
          name: Flickr30K
        metrics:
          - type: recall
            value: 42.9
            name: Image-to-Text R@1
          - type: recall
            value: 80.6
            name: Image-to-Text R@10
          - type: recall
            value: 42.2
            name: Text-to-Image R@1
          - type: recall
            value: 79.4
            name: Text-to-Image R@10
      - task:
          type: image-classification
          name: Linear Probe
        dataset:
          type: cifar100
          name: CIFAR-100
        metrics:
          - type: accuracy
            value: 75.0
            name: Linear probe Top-1
"""
if "model-index:" not in txt:
    end = txt.index("---", 3)
    txt = txt[:end] + MODEL_INDEX + txt[end:]
    print("  + model-index metadata (9 metrics)")
else:
    print("  = model-index already present")

# ------------------------------------------------------- 2. downstream results
DOWNSTREAM = f"""
---

## Downstream Results — Vision-Language Model

The encoder is **frozen** and used as-is inside a bilingual (English + Korean)
vision-language model: [`{VLM}`](https://huggingface.co/{VLM}).
Only a projector (8.9M) and LoRA adapters (5.0M) are trained on top; the decoder is
Qwen2.5-7B-Instruct.

A vision-language model can score respectably while barely using the image, because
many questions are answerable from language priors alone. Each benchmark below is
therefore run **twice** — once normally, once with the visual tokens removed and
everything else held constant. **The difference is what this encoder contributes**,
and it is the only figure the language model cannot produce on its own.

| Benchmark | With this encoder | Blind | **Encoder contributes** |
|---|---|---|---|
| POPE accuracy (n=2,000) | **70.50%** | 44.55% | **+25.95** |
| POPE F1 (n=2,000) | **70.59%** | 23.46% | **+47.13** |
| VQAv2 strict (n=5,000) | **53.04%** | 39.83% | **+13.21** |
| MME total, 14 categories (n=2,374) | **1258.9** | 812.0 | **+446.9** |
| MMBench (n=4,377) | 60.20% | — | — |
| Held-out perplexity (KO / EN) | 2.15 / 2.99 | — | — |

The POPE yes-ratio is **0.492 against a gold ratio of 0.500**, so the model genuinely
discriminates on object-existence questions rather than defaulting to one answer —
accuracy alone is not interpretable without that check. The blind VQAv2 score of
39.83% lands where language-prior performance is expected to, which is corroborating
evidence that the measurement is sound.

Notably, the encoder reaches these numbers while scoring 37.2% zero-shot on ImageNet.
Zero-shot accuracy measures image–text *alignment*; the 75.0% CIFAR-100 linear probe
is the better indicator of representation quality for downstream use, because a
trained projector relearns the alignment from scratch.

"""
if "## Downstream Results" not in txt:
    anchor = "## Comparison with Existing Encoders"
    if anchor in txt:
        txt = txt.replace(anchor, DOWNSTREAM.lstrip("\n") + "---\n\n" + anchor, 1)
    else:
        txt += DOWNSTREAM
    print("  + Downstream Results section")
else:
    print("  = Downstream Results already present")

# ------------------------------------------------------------ 3. roadmap link
OLD_ROW = "| **Mid** | **183.5M** | **Training complete — evaluation published** |"
NEW_ROW = ("| **Mid** | **183.6M** | **Stage 1 complete; Stage 2 VLM published — "
           f"[{VLM.split('/')[-1]}](https://huggingface.co/{VLM})** |")
if OLD_ROW in txt:
    txt = txt.replace(OLD_ROW, NEW_ROW)
    print("  + roadmap row links to the VLM repo")

LOCAL.write_text(txt)
print(f"\nwrote {LOCAL.relative_to(ROOT)}  ({orig_len} -> {len(txt)} chars, "
      f"+{len(txt)-orig_len})")

if args.dry_run:
    print("--dry-run: not uploaded.")
    sys.exit(0)
if not token:
    print("ERROR: set HF_TOKEN to upload.")
    sys.exit(1)

api.upload_file(path_or_fileobj=str(LOCAL), path_in_repo="README.md",
                repo_id=REPO, repo_type="model")
print(f"uploaded -> https://huggingface.co/{REPO}")
