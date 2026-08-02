---
language: en
license: other
license_name: proprietary
license_link: LICENSE
tags:
  - vision
  - vision-encoder
  - image-text
  - contrastive-learning
  - knowledge-distillation
  - adaptive-tokenization
library_name: transformers
pipeline_tag: image-feature-extraction
model-index:
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
---

# Keural Mid-Level Vision Encoder

*A 183.5M-parameter image encoder built from scratch — adaptive tokenization, hierarchical concept tokens, and continuous 2D positional encoding. Not a CLIP/ViT fine-tune.*

![Keural Mid architecture](keural_mid_architecture.png)

<p>
  <img alt="params" src="https://img.shields.io/badge/params-183.6M-blue"/>
  <img alt="from scratch" src="https://img.shields.io/badge/trained-from%20scratch-success"/>
  <img alt="precision" src="https://img.shields.io/badge/precision-bfloat16-informational"/>
  <img alt="status" src="https://img.shields.io/badge/status-training%20complete-success"/>
</p>

> Developed by **MKD Co., Ltd.** · Code: [github.com/mkd-hika/Keural-Vision-Encoder-Mid-level](https://github.com/mkd-hika/Keural-Vision-Encoder-Mid-level)

---

## At a Glance

| | |
|---|---|
| **Parameters** | 183,575,297 (183.6M) |
| **Input** | RGB image, 384×384 (any multiple of 16 supported natively) |
| **Output** | 512 adaptive tokens (768-dim) + one pooled 768-dim image embedding |
| **Trained** | From random init on ~20M image–text pairs, 18,000 steps, 2× H200 |
| **Precision** | bfloat16 |

| Benchmark | Result |
|---|---|
| ImageNet zero-shot Top-1 / Top-5 | 37.2% / 66.8% |
| Flickr30K Image→Text R@1 / R@5 / R@10 | 42.9% / 72.7% / 80.6% |
| Flickr30K Text→Image R@1 / R@5 / R@10 | 42.2% / 70.0% / 79.4% |
| CIFAR-100 linear probe | 75.0% |

Measured at 384×384 with `token_budget=512`, matching training. Accuracy trails
same-size peers — see [Limitations](#limitations) and [Evaluation](#evaluation)
for why, and for the higher-k retrieval results where the gap closes.

---

## Downstream Results — Vision-Language Model

The encoder is **frozen** and used as-is inside a bilingual (English + Korean)
vision-language model: [`mkd-hika/keural-mid-vlm-bilingual`](https://huggingface.co/mkd-hika/keural-mid-vlm-bilingual).
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

---

## Model Description

**Keural Mid** is a mid-scale vision encoder that maps an image to a single 768-dim embedding aligned to a text embedding space, plus a variable-length sequence of semantically-typed tokens. It is trained from random initialization (no CLIP/ViT/DINO backbone) with a combination of sigmoid contrastive learning, hierarchical concept alignment, saliency regularization, and dual-teacher knowledge distillation.

It introduces three mechanisms not found together in existing encoders (CLIP, SigLIP, DINOv2):

| Innovation | What it does |
|---|---|
| **ATB — Adaptive Token Budget** | Content-aware token allocation. A learned saliency map drives a differentiable Gumbel-Top-K sampler, spending more tokens on complex regions and fewer on simple backgrounds. |
| **HCT — Hierarchical Concept Tokenization** | Every token carries a semantic level — `global`, `region`, or `detail` — via learned level embeddings and level-specific constructors. |
| **Continuous 2D-RoPE** | Rotary position encoding from actual normalized `(cx, cy)` coordinates plus a log-area scale bias, so variable resolutions and token scales are handled natively. |

---

## Intended Uses & Limitations

### Intended uses
- **Image feature extraction** — a global `pooled` embedding for retrieval, clustering, or as a frozen backbone for downstream heads (e.g. linear probing, detection, captioning front-ends).
- **Research** into adaptive tokenization, hierarchical token typing, and resolution-elastic position encoding.

### Limitations
- **Accuracy trails same-size peers.** Zero-shot ImageNet Top-1 is 37.2%, versus 68.6% for CLIP ViT-B/16 (~86M params) and 76.7% for SigLIP ViT-B/16 (~93M params) — both smaller than this model. This is attributed to a training-budget gap (18K steps on ~20M pairs, vs. peers trained on 12M-400M+ pairs over far more steps) rather than an architecture ceiling: training loss was still declining at the final step. See [Evaluation](#evaluation).
- **Saliency mechanism is real but token placement is still settling.** The learned saliency map produces stable, object-tracking contours on images with a clear dominant subject. However, the *discrete* tokens it samples continued shifting meaningfully until very late in training (only ~50% overlap with the final checkpoint's token positions as of step 13-14K, reaching ~73% by step 16K) — exact token placement had not fully converged even at the final checkpoint.
- **Domain/bias.** Trained on web image–text pairs; it inherits the coverage and biases of that distribution and is English-text aligned.
- **Not a safety-filtered model.** No content moderation or de-biasing has been applied.

---

## How to Use

```python
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "mkd-hika/keural-vision-encoder-mid",
    trust_remote_code=True,
).eval()

transform = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("example.jpg").convert("RGB")
pixel_values = transform(image).unsqueeze(0)          # (1, 3, 384, 384)

with torch.no_grad():
    out = model(pixel_values=pixel_values, token_budget=512)  # match training exactly

image_embedding = torch.nn.functional.normalize(out.pooled, dim=-1)  # (1, 768)

print(out.tokens.shape)            # (1, 512, 768)  per-token features
print(out.pooled.shape)            # (1, 768)       global image embedding
print(out.level_ids.shape)         # (1, 512)       0=global, 1=region, 2=detail
print(out.spatial_metadata.shape)  # (1, 512, 4)    cx, cy, scale_w, scale_h
print(out.saliency_scores.shape)   # (1, 512)       per-token importance
```

> **Checkpoints.** The repo root holds the final model (step 18,000). Earlier
> checkpoints are available as subfolders for studying training dynamics:
> `checkpoint-9000`, `checkpoint-10000`, `checkpoint-15000`, `checkpoint-18000`
> — load one with `AutoModel.from_pretrained(..., subfolder="checkpoint-15000")`.

> **Note:** pass `token_budget=512` explicitly. Training always used a fixed
> budget of 512 tokens; leaving `token_budget` unset falls back to
> resolution-based auto-scaling (e.g. 1,152 tokens at 384px), which the model
> was not trained on. Evaluation showed this mismatch moves results by <1
> point, so it's a minor effect, but 512 is the methodologically correct
> setting to match training.

### Output fields

| Field | Shape | Description |
|---|---|---|
| `pooled` | `(B, 768)` | Global image embedding (`[POOL]` token output). |
| `tokens` | `(B, N, 768)` | Per-token feature vectors (`N` = allocated token budget). |
| `level_ids` | `(B, N)` | Semantic level per token: `0`=global, `1`=region, `2`=detail. |
| `spatial_metadata` | `(B, N, 4)` | `(cx, cy, scale_w, scale_h)`, normalized to `[0, 1]`. |
| `saliency_scores` | `(B, N)` | Per-token importance in `[0, 1]`. |
| `attention_mask` | `(B, N)` | Valid-token mask. |

---

## Architecture

The figure above is the complete specification: **(a)** the tokenizer stack
(CNN stem → saliency → ATB → HCT), **(b)** the spatial transformer block and
continuous 2D-RoPE, **(c)** the output contract, dual-teacher distillation, and
the training objective.


The `pooled` image embedding is the `[POOL]` token after the final RMSNorm (no separate projection head). For contrastive training it is aligned against a trainable projection of a frozen CLIP text encoder.

### Model specifications

| | |
|---|---|
| Parameters | 183,575,297 (183.5M) |
| Embedding dimension | 768 |
| Transformer depth | 24 blocks |
| Attention heads | 12 (head_dim = 64) |
| FFN | SwiGLU (hidden = 3072) |
| Normalization | RMSNorm |
| Position encoding | Continuous 2D-RoPE + scale bias |
| Token budget | 512 default (training-matched), up to 2048 |
| Precision | bfloat16 |
| Input resolution | 384×384, 448×448 |

---

## Training

| | |
|---|---|
| Data | ~20M image–text pairs |
| Hardware | 2× NVIDIA H200 (141 GB) |
| Objective | SigLIP contrastive + HCT contrastive + saliency regularizer + dual-teacher cosine distillation |
| Teachers (frozen) | SigLIP-SO400M (1152-d) + InternViT-300M (1024-d) |
| Optimizer | AdamW, lr 5e-4, cosine schedule, 2K warmup, weight decay 0.05 |
| Effective batch | 3,072 (768 × 2 grad-accum × 2 GPUs, distributed `all_gather` negatives) |
| Schedule | 18,000 optimizer steps, complete |
| Precision | bfloat16 (mixed) |
| Final loss | 2.87 total (down from 21.56 at step 0), still declining at completion |

**Loss.** `L_total = L_primary + λ_hct·L_hct + λ_sal·β_sal·L_saliency + L_distill`, where `L_primary` and `L_hct` are sigmoid image↔text losses (on the pooled embedding and on the mean of global tokens respectively), `L_saliency` is an anti-collapse + total-variation regularizer on the saliency map, and `L_distill` is cosine distance to the two frozen teachers.

![Training loss curves](training_loss_curves.png)

---

## Distillation Teachers

| Teacher | Params | Embed dim | Projection to 768 |
|---|---|---|---|
| SigLIP-SO400M | 400M | 1152 | `Linear(1152 → 768)` |
| InternViT-300M | 300M | 1024 | `Linear(1024 → 768)` |

Both teachers are frozen and used only during training; they are not required for inference.

---

## Evaluation

Evaluated on the final checkpoint with `token_budget=512` (exactly matching training).

| Benchmark | Result |
|---|---|
| ImageNet zero-shot Top-1 | **37.2%** |
| ImageNet zero-shot Top-5 | 66.8% |
| Flickr30K Image→Text R@1 / R@5 / R@10 | 42.9% / 72.7% / 80.6% |
| Flickr30K Text→Image R@1 / R@5 / R@10 | 42.2% / 70.0% / 79.4% |
| CIFAR-100 linear probe Top-1 | 75.0% |

![Benchmark comparison](benchmark_comparison_chart.png)

![Flickr30K Recall@k](benchmark_recall_curve_chart.png)

![Multi-axis comparison](benchmark_radar_chart.png)

**Context.** Against its own distillation teachers, Keural Mid retains ~45% of SigLIP-SO400M's zero-shot ImageNet accuracy (83.1%) and ~48-56% of its Flickr30K retrieval R@1, after 18K steps on ~20M pairs vs. teachers trained on billions of pairs. Against same-parameter-class peers (CLIP ViT-B/16, SigLIP ViT-B/16 — both smaller than this model), accuracy trails by roughly 2x. Both gaps are consistent with a **training-budget** shortfall rather than an architecture ceiling: loss was still declining at the final step. A compute/data-matched fixed-grid baseline is the recommended next experiment to isolate the ATB architecture's own contribution from this gap.

#### The R@1 gap narrows sharply at higher k

The headline R@1 numbers understate retrieval quality. Broken out by k:

![Flickr30K Recall@5](benchmark_recall_at5_chart.png)

![Flickr30K Recall@10](benchmark_recall_at10_chart.png)

On **Text→Image R@5, Keural Mid (70.0%) outperforms CLIP ViT-B/16 trained on WIT-400M (57.2%)**, and at R@10 it leads 79.4% vs. 68.0% — despite that baseline having seen ~20x more training data. The correct-image is usually retrieved within the top few results even when it isn't ranked first, which suggests the embedding space is broadly well-organized and that the R@1 shortfall is largely a ranking-precision issue rather than a representation-quality one. (Note the CC12M-trained CLIP variant still leads at these k, so this is not a uniform win — it is specific to the WIT-400M baseline on the text→image direction.)

### Adaptive tokenizer — saliency & token placement

![Saliency evolution across training](saliency_evolution_9ckpt.png)

The learned saliency map produces sharp, stable contours that track each image's dominant subject, visible from mid-training through the final checkpoint. Quantitatively, the continuous saliency map stabilizes fast (cosine similarity to the final checkpoint reaches ~0.83-0.90 within the first few thousand steps), but the *discrete* sampled token positions are considerably less stable — only ~50% overlap with the final checkpoint's tokens by step 13-14K, jumping to 73% at step 16K. Exact token placement was still moving late into training even where the saliency field itself looked converged.

---

---

## Comparison with Existing Encoders

| Feature | CLIP ViT-L | SigLIP-SO400M | DINOv2-L | **Keural Mid** |
|---|---|---|---|---|
| Parameters | 307M | 400M | 307M | **183.5M** |
| Tokenization | Fixed patches | Fixed patches | Fixed patches | **Adaptive (ATB)** |
| Token count | Fixed | Fixed | Fixed | **Variable (512–2048)** |
| Semantic token levels | — | — | — | **3-level (HCT)** |
| Position encoding | Learned absolute | Learned absolute | Learned absolute | **Continuous 2D-RoPE** |
| Saliency-aware | No | No | No | **Yes** |

| Same-parameter-class peer | Params | ImageNet Zero-shot Top-1 | Flickr30K I2T R@1 |
|---|---|---|---|
| **Keural Mid** | 183.6M | 37.2% | 42.9% |
| CLIP ViT-B/16 (OpenAI) | ~86M | 68.6% | 88.2% |
| SigLIP ViT-B/16 | ~93M | 76.7% | — |

---

## Roadmap

| Stage | Params | Status |
|---|---|---|
| PoC | 24.7M | Complete |
| **Mid** | **183.6M** | **Stage 1 complete; Stage 2 VLM published — [keural-mid-vlm-bilingual](https://huggingface.co/mkd-hika/keural-mid-vlm-bilingual)** |
| Large | ~1.1B | Planned |

---

## Citation

```bibtex
@misc{keural-mid-2026,
  title  = {Keural Mid-Level Vision Encoder},
  author = {MKD Co., Ltd.},
  year   = {2026},
  url    = {https://huggingface.co/mkd-hika/keural-vision-encoder-mid}
}
```

## License

Copyright © 2026 MKD Co., Ltd. All Rights Reserved. Proprietary — see `LICENSE`.
