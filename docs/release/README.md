<div align="center">

# Keural VLM — Mid Level

**A bilingual vision-language model on a 183.6M encoder trained from scratch**

*Adaptive Token Budget · Hierarchical Concept Tokens · Continuous 2D-RoPE · Dual-Teacher Distillation*

<img alt="encoder" src="https://img.shields.io/badge/encoder-183.6M-blue">
<img alt="languages" src="https://img.shields.io/badge/languages-EN%20%2B%20KO-blueviolet">
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white">
<img alt="precision" src="https://img.shields.io/badge/precision-bfloat16-informational">
<img alt="license" src="https://img.shields.io/badge/license-proprietary-red">

[**🤗 Encoder**](https://huggingface.co/mkd-hika/keural-vision-encoder-mid) ·
[**🤗 VLM (EN/KO)**](https://huggingface.co/mkd-hika/keural-mid-vlm-bilingual) ·
[**📐 Technical Spec**](docs/PIPELINE_TECHNICAL_SPEC.md)

![Keural Mid architecture](assets/keural_mid_architecture.png)

</div>

---

## What this is

A vision encoder built from random initialization — not a CLIP, ViT or DINO fine-tune —
and the bilingual vision-language model built on top of it. The encoder combines three
mechanisms that do not commonly appear together:

- **Adaptive Token Budget (ATB)** — token *positions* are chosen by a learned saliency
  map through differentiable Gumbel-Top-K with non-maximum suppression, under a budget
  set at call time or scaled with input resolution (default 512, max 2,048).
- **Hierarchical Concept Tokenization (HCT)** — every token carries a semantic level
  (global / region / detail) with its own learned embedding and constructor.
- **Continuous 2D-RoPE** — position is encoded from each token's real normalized
  coordinates rather than a patch-grid index, so any resolution that is a multiple of
  16 works without interpolating a positional table.

## Results

### What the vision encoder contributes

A vision-language model can score respectably while barely using the image, because
many questions are answerable from language priors alone. Every benchmark below is run
**twice** — once normally, once with the visual tokens removed and everything else held
constant. **The difference is the encoder's contribution**, and it is the only figure a
language model cannot produce on its own.

| Benchmark | With image | Blind | **Encoder contributes** |
|---|---|---|---|
| POPE accuracy (n=2,000) | **70.50%** | 44.55% | **+25.95** |
| POPE F1 (n=2,000) | **70.59%** | 23.46% | **+47.13** |
| VQAv2 strict (n=5,000) | **53.04%** | 39.83% | **+13.21** |
| VQAv2 lenient (n=5,000) | **53.58%** | 40.27% | **+13.31** |
| MME total, 14 categories (n=2,374) | **1258.9** | 812.0 | **+446.9** |
| MMBench (n=4,377) | 60.20% | — | — |
| Held-out perplexity, Korean / English | 2.15 / 2.99 | — | — |

![Vision contribution](docs/figures/stage2_vision_contribution.png)

The POPE **yes-ratio is 0.492 against a gold ratio of 0.500**, so the model genuinely
discriminates on object-existence questions rather than defaulting to one answer — a
model replying "yes" indiscriminately scores near 50% accuracy while being useless, so
accuracy alone is not interpretable without it. The blind VQAv2 score of 39.83% lands
where language-prior performance is expected to, which is corroborating evidence that
the measurement itself is sound.

### Stage 1 — the encoder alone

| Benchmark | Result |
|---|---|
| ImageNet zero-shot Top-1 / Top-5 | 37.2% / 66.8% |
| Flickr30K image→text R@1 / R@5 / R@10 | 42.9% / 72.7% / 80.6% |
| Flickr30K text→image R@1 / R@5 / R@10 | 42.2% / 70.0% / 79.4% |
| CIFAR-100 linear probe Top-1 | 75.0% |

Zero-shot accuracy measures image–text *alignment*, and understates downstream
usefulness: a trained projector relearns that alignment from scratch, so the 75.0%
CIFAR-100 linear probe is the better indicator of representation quality — which the
Stage 2 numbers above bear out.

## Training pipeline

| Stage | Purpose | Data | Steps | Held-out perplexity |
|---|---|---|---|---|
| 1 | Contrastive pre-training + dual-teacher distillation | 20M image–text pairs | 18,000 | loss 21.56 → 2.87 |
| 2a | Projector alignment on captions | LLaVA-Pretrain 557K | 5,000 | 31.81 → 17.30 (EN) |
| 2b | Supervised instruction tuning (SFT) | LLaVA-Instruct 157K | 8,000 | 3.22 → 2.90 (EN) |
| 2b | Bilingual SFT (EN + KO) | Merged 603K | 12,000 | 2.90 → **2.14** (KO) |

![Stage 2b perplexity](docs/figures/stage2_val_perplexity.png)

The encoder is **frozen** throughout Stage 2. Only a LevelAwareProjector (8.9M) and
LoRA adapters (5.0M, r=16 α=32) train against a frozen Qwen2.5-7B-Instruct — 14.0M
trainable parameters against 7.8B frozen. Label masking supervises **assistant turns
only**, so the model learns to produce answers rather than to reproduce questions.

### Reproducibility mattered more than expected

Stage 2a originally set no random seed, and two runs on identical data and identical
hyperparameters reached perplexity 23.90 and 124.50 respectively, because projector and
LoRA initialization differed on every launch. Fixing the seed, halving the peak learning
rate to 5e-4 and lengthening warmup to 500 steps removed the instability and produced
the best result for the stage. Every config here carries `--seed 42`.

![Stage 2a stability](docs/figures/stage2a_val_perplexity.png)

### Answer format is a real variable

The English-only model scored 1.73% strict against 48.60% lenient on VQAv2 — a 47-point
divergence caused by answer *format*, not perception. Asked "Is this rice noodle soup?"
with gold answer "yes", it replied "Yes, this is a bowl of rice noodle soup", which
exact-match scoring credits as zero. The mixture contained only verbose conversational
data and no short-answer supervision in English. Adding 84,590 English short-answer
records raised strict accuracy to 53.04%, and strict and lenient now agree to within
0.54 points.

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/smoke_test_mid.py          # no data or teachers needed, ~1 min
```

### Inference

```bash
CKPT=outputs/projector_2b_bilingual/checkpoint-12000
python3 scripts/demo_vlm.py \
  --projector-ckpt $CKPT/projector.pt --lora-ckpt $CKPT/lora_adapter \
  --image photo.jpg --question "What is in this image?"
```

Input must be **512×512 with a 512-token budget** — what Stage 2 trained at. Neither is
auto-detected, and inferring at a different size is off-distribution with no error.

### Evaluation GUI

```bash
streamlit run scripts/vlm_studio.py --server.port 8501 --server.address 0.0.0.0
```

Multi-turn chat about an image, checkpoint comparison without reloading the decoder,
saliency and token-placement visualization, a prompt-sensitivity lab, and a
sighted-vs-blind view on every answer.

### Training

```bash
bash scripts/run_stage2a_caption.sh      # projector alignment
bash scripts/run_stage2b_bilingual.sh    # bilingual SFT
bash scripts/run_eval_2b.sh              # POPE + VQAv2, sighted and blind
```

See [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for data preparation and hardware notes.

## Limitations

- **Long-form output fabricates detail.** Asked for a detailed description the model
  invents plausible objects that are not present. Short answers are markedly more
  reliable, so answer length is a reliability control, not a verbosity setting.
- **Text recognition is partial.** No OCR data was used in any stage reported here. The
  model scores 110.0 on the MME OCR category against a blind 50.0 and reads large,
  high-contrast signage, but cannot read documents or small text.
- **Sensitive to prompt phrasing.** On POPE the instruction "Answer yes or no only" —
  absent from training — collapsed the yes-ratio to 0.050, while the trained phrasing
  gave 0.492. Prompts resembling the training format behave very differently from those
  that do not.
- **The encoder is the ceiling.** At 37.2% zero-shot ImageNet it is undertrained
  relative to CLIP ViT-L/14 (~75%), and it bounds every downstream number here. Its
  Stage 1 loss was still declining when the run ended.
- **No temporal modelling.** Single images only.

## Documentation

- [**PIPELINE_TECHNICAL_SPEC.md**](docs/PIPELINE_TECHNICAL_SPEC.md) — full architecture and data flow
- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) — module-level design
- [**BENCHMARK_COMPARISON.md**](docs/BENCHMARK_COMPARISON.md) — evaluation methodology and peer comparison
- [**TRAINING_GUIDE.md**](docs/TRAINING_GUIDE.md) — reproducing each stage
- [**KEURAL_TINY_VS_MID.md**](docs/KEURAL_TINY_VS_MID.md) — scaling from the 24.7M proof of concept

## Citation

```bibtex
@misc{keural_mid_vlm_2026,
  title  = {Keural VLM Mid-Level: A Bilingual Vision-Language Model on an Adaptive-Tokenization Encoder},
  author = {Barki, Hika},
  year   = {2026},
  howpublished = {\url{https://github.com/mkd-hika/Keural-VLM-Mid-level}},
  note   = {MKD Co., Ltd.}
}
```

## License

Proprietary — MKD Co., Ltd. Qwen2.5-7B-Instruct (Apache-2.0) is used as the decoder and
is not redistributed here.
