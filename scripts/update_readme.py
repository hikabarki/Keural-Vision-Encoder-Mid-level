"""
Bring README.md up to date with Stage 2 and Stage 3.

The repository README still describes this as a vision-encoder project whose
Stage 2 is prospective. Stage 2 is complete (2a, 2b English, 2b bilingual),
Stage 2c ran twice, and Stage 3 DPO ran and is documented as a negative result.
Several tools added since — the Streamlit studio, the video probe, the DPO
trainer, the evaluation queue — are absent from the repository map.

Edits are surgical: the badge row, a new Results section, the repository map,
the documentation list and the header links. Existing prose is untouched.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
txt = README.read_text()
before = len(txt)

# ---------------------------------------------------------------- 1. badges --
txt = txt.replace(
    '<img alt="status" src="https://img.shields.io/badge/status-training-orange">',
    '<img alt="status" src="https://img.shields.io/badge/status-Stage%202%20complete-success">\n'
    '<img alt="vlm" src="https://img.shields.io/badge/VLM-bilingual%20EN%2FKO-blueviolet">')

# ---------------------------------------------------------------- 2. links ---
txt = txt.replace(
    '[**🤗 Model**](https://huggingface.co/mkd-hika/keural-vision-encoder-mid) ·',
    '[**🤗 Encoder**](https://huggingface.co/mkd-hika/keural-vision-encoder-mid) ·\n'
    '[**🤗 VLM (EN/KO)**](https://huggingface.co/mkd-hika/keural-mid-vlm-bilingual) ·')

# ------------------------------------------------------------- 3. results ----
RESULTS = """
---

## Results

### Stage 1 — vision encoder

| Benchmark | Result |
|---|---|
| ImageNet zero-shot Top-1 / Top-5 | 37.2% / 66.8% |
| Flickr30K image→text R@1 / R@5 / R@10 | 42.9% / 72.7% / 80.6% |
| Flickr30K text→image R@1 / R@5 / R@10 | 42.2% / 70.0% / 79.4% |
| CIFAR-100 linear probe Top-1 | 75.0% |

Zero-shot accuracy measures image–text *alignment*. For downstream use the
CIFAR-100 linear probe is the better indicator of representation quality, because
a trained projector relearns the alignment from scratch — which the Stage 2
results below bear out.

### Stage 2 — vision-language model

The encoder is **frozen** inside the VLM; only a projector (8.9M) and LoRA
adapters (5.0M) train on top of Qwen2.5-7B-Instruct. A VLM can score respectably
while barely using the image, so every benchmark is run **twice** — once
normally, once with the visual tokens removed and everything else held constant.
**The difference is what the encoder contributes**, and it is the only figure a
language model cannot produce on its own.

| Benchmark | With image | Blind | **Encoder contributes** |
|---|---|---|---|
| POPE accuracy (n=2,000) | **70.50%** | 44.55% | **+25.95** |
| POPE F1 (n=2,000) | **70.59%** | 23.46% | **+47.13** |
| VQAv2 strict (n=5,000) | **53.04%** | 39.83% | **+13.21** |
| MME total, 14 categories (n=2,374) | **1258.9** | 812.0 | **+446.9** |
| MMBench (n=4,377) | 60.20% | — | — |
| Held-out perplexity, Korean / English | 2.15 / 2.99 | — | — |

POPE yes-ratio is **0.492 against a gold 0.500**, so the model discriminates on
object-existence questions rather than defaulting to one answer — accuracy alone
is not interpretable without that check.

![Vision contribution](docs/figures/stage2_vision_contribution.png)

### Training stages

| Stage | What it does | Data | Steps | Outcome |
|---|---|---|---|---|
| 1 | Contrastive pre-training + dual-teacher distillation | 20M pairs | 18,000 | ✅ encoder |
| 2a | Projector alignment on captions | LLaVA-Pretrain 557K | 5,000 | ✅ ppl 31.81 → 17.30 |
| 2b | Supervised instruction tuning (SFT), English | LLaVA-Instruct 157K | 8,000 | ✅ ppl 3.22 → 2.90 |
| 2b | SFT, bilingual EN+KO | Merged 603K | 12,000 | ✅ **recommended model** |
| 2c | OCR specialization (two runs) | TextVQA 30K + replay | 4,000 / 2,000 | ⚠️ trade-off |
| 3 | Direct Preference Optimization | RLHF-V 5.5K pairs | 2,000 | ❌ negative result |

Every stage after SFT traded one capability for another. Stage 2c raised MME
(1258.9 → 1337.8) and MME OCR (110.0 → 137.5) but drifted toward answering
"yes" on binary questions. Stage 3 DPO optimized its own objective successfully
(held-out reward accuracy 0.890) but collapsed toward answering "no"
(POPE yes-ratio 0.503 → 0.116). **Stage 2b bilingual is the recommended
checkpoint** — it is the only calibrated one.

![Model selection](docs/figures/model_selection.png)

Full analysis, including the diagnosis of each failure, is in
[the technical report](docs/reports/Keural_Mid_Technical_Documentation_Report.docx).

"""
if "## Results" not in txt:
    txt = txt.replace("\n---\n\n## Architecture", RESULTS + "---\n\n## Architecture", 1)

# ---------------------------------------------------- 4. repository structure -
NEW_ENTRIES = """│   ├── <a href="scripts/run_stage2c_v2.sh">run_stage2c_v2.sh</a>        # Stage 2c launcher — OCR, calibration-corrected
│   ├── <a href="scripts/train_dpo_vlm.py">train_dpo_vlm.py</a>         # Stage 3: DPO (precomputed reference policy)
│   ├── <a href="scripts/prepare_dpo_data.py">prepare_dpo_data.py</a>      # RLHF-V preference pairs -> JSONL
│   ├── <a href="scripts/prepare_ocr_data_v2.py">prepare_ocr_data_v2.py</a>   # OCR mix with stratified yes/no replay
│   ├── <a href="scripts/build_bilingual_mix_v2.py">build_bilingual_mix_v2.py</a> # EN/KO mix + English short-answer VQA
│   ├── <a href="scripts/vlm_studio.py">vlm_studio.py</a>            # Streamlit GUI — chat, compare, saliency, prompt lab
│   ├── <a href="scripts/demo_video.py">demo_video.py</a>            # Video understanding probe (frame aggregation)
│   ├── <a href="scripts/gpu_queue.sh">gpu_queue.sh</a>             # Sequential job queue so the GPU never idles
│   ├── <a href="scripts/run_eval_2b.sh">run_eval_2b.sh</a>           # POPE + VQAv2 sighted/blind in one pass
│   ├── <a href="scripts/publish_to_hf.py">publish_to_hf.py</a>         # Publish encoder + VLM to Hugging Face
"""
anchor = '│   ├── <a href="scripts/demo_vlm.py">demo_vlm.py</a>'
if anchor in txt and "vlm_studio.py" not in txt:
    txt = txt.replace(anchor, NEW_ENTRIES + anchor, 1)

# --------------------------------------------------------- 5. documentation --
DOCS = """- [**Technical Documentation Report**](docs/reports/Keural_Mid_Technical_Documentation_Report.docx) — full architecture, methodology, mathematics and results for Stages 1–3
- [**VLM model card**](docs/HF_VLM_MODEL_CARD.md) — the bilingual vision-language model
- [**Training guide**](docs/TRAINING_GUIDE.md) — how to reproduce each stage
"""
if "## Documentation" in txt and "Technical Documentation Report" not in txt:
    txt = txt.replace("## Documentation\n\n", "## Documentation\n\n" + DOCS, 1)

# quick consistency fix: measured parameter count is 183,575,297
txt = txt.replace("183.5M params ·", "183.6M params ·")

README.write_text(txt)
print(f"README.md: {before} -> {len(txt)} chars (+{len(txt)-before})")
for k in ("## Results", "vlm_studio.py", "train_dpo_vlm.py", "keural-mid-vlm-bilingual",
          "Stage%202%20complete", "Technical Documentation Report"):
    print(f"  {'OK ' if k in txt else 'MISS'} {k}")
