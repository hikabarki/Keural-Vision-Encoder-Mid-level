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
---

# Keural Mid-Level Vision Encoder

**183.5M parameters | Built from scratch | Not a CLIP/ViT fine-tune**

> Developed by MKD Co., Ltd.

## Model Description

Keural Mid is a novel vision encoder that introduces three innovations absent from existing encoders (CLIP, SigLIP, DINOv2):

| Innovation | What it does |
|---|---|
| **ATB (Adaptive Token Budget)** | Content-aware token selection — allocates more tokens to complex regions, fewer to simple backgrounds |
| **HCT (Hierarchical Concept Tokenization)** | Tags every token with a semantic level (global/region/detail) — no existing encoder does this |
| **Continuous 2D-RoPE** | Position encoding using actual (cx, cy) coordinates — handles variable resolution natively |

## Model Details

| | |
|---|---|
| Parameters | 183,575,297 (183.5M) |
| Embedding Dimension | 768 |
| Transformer Depth | 24 blocks |
| Attention Heads | 12 (head_dim=64) |
| FFN Type | SwiGLU (hidden=3072) |
| Normalization | RMSNorm |
| Token Budget | 512 default, up to 2048 |
| Precision | bfloat16 |
| Input Resolution | 384×384, 448×448 |

## Architecture

```
RGB Image (B, 3, H, W)
    │
    ▼
┌─────────────────────────┐
│ CNN Stem (4-stage)       │ 3→128→256→512→768, stride 16×
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Saliency Predictor       │ Per-pixel importance map [0,1]
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ ATB Tokenizer            │ Gumbel top-K + NMS + ROI-Align
│ Budget: 5% global        │ Differentiable token selection
│ 25% region, 70% detail   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ HCT                      │ 3 semantic levels + per-level MLPs
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Spatial Transformer (24L)│ RMSNorm, SwiGLU, 2D-RoPE
│ + [POOL] token           │ Scale bias (8 buckets)
└─────────────────────────┘
    │
    ▼
  Output: tokens (B,N,768) + pooled (B,768)
          + level_ids + spatial_metadata + saliency_scores
```

## Training

- **Dataset:** 20M curated pairs from DataComp-1B Medium (Stage 1)
- **Hardware:** 2× NVIDIA H200 141GB
- **Distillation:** Dual-teacher (SigLIP-SO400M + InternViT-300M)
- **Loss:** SigLIP contrastive + HCT + Saliency regularizer + Cosine distillation
- **Optimizer:** AdamW (lr=5e-4, cosine schedule, 2K warmup)
- **Effective batch:** 2304 (576 × 2 accum × 2 GPUs)
- **Precision:** bfloat16
- **Steps:** 18,000 (~2 full epochs)

## Usage

```python
from transformers import AutoModel, AutoConfig
import torch

# Load model
model = AutoModel.from_pretrained(
    "mkd-hika/keural-vision-encoder-mid",
    trust_remote_code=True,
)
model.eval()

# Encode an image
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("example.jpg").convert("RGB")
pixel_values = transform(image).unsqueeze(0)  # (1, 3, 384, 384)

with torch.no_grad():
    output = model(pixel_values=pixel_values)

print(output.tokens.shape)           # (1, 512, 768)
print(output.pooled.shape)           # (1, 768)
print(output.level_ids.shape)        # (1, 512) — 0=global, 1=region, 2=detail
print(output.spatial_metadata.shape) # (1, 512, 4) — cx, cy, scale_w, scale_h
print(output.saliency_scores.shape)  # (1, 512)
```

## Output Fields

| Field | Shape | Description |
|---|---|---|
| `tokens` | (B, N, 768) | Per-token feature vectors |
| `pooled` | (B, 768) | Global image representation |
| `attention_mask` | (B, N) | Valid token positions |
| `level_ids` | (B, N) | 0=global, 1=region, 2=detail |
| `spatial_metadata` | (B, N, 4) | (cx, cy, scale_w, scale_h) normalized to [0,1] |
| `saliency_scores` | (B, N) | Per-token importance score |

## Knowledge Distillation Teachers

| Teacher | Params | Embed Dim | Projection |
|---|---|---|---|
| SigLIP-SO400M | 400M | 1152 | Linear(1152, 768) |
| InternViT-300M | 300M | 1024 | Linear(1024, 768) |

## Comparison with Existing Encoders

| Feature | CLIP ViT-L | SigLIP-SO400M | DINOv2-L | **Keural Mid** |
|---|---|---|---|---|
| Parameters | 307M | 400M | 307M | **183.5M** |
| Tokenization | Fixed patches | Fixed patches | Fixed patches | **Adaptive (ATB)** |
| Token count | Fixed 256 | Fixed 729 | Fixed 729 | **Variable 512-2048** |
| Semantic levels | None | None | None | **3-level HCT** |
| Position encoding | Learned absolute | Learned absolute | Learned absolute | **2D-RoPE** |
| Saliency-aware | No | No | No | **Yes** |

## Roadmap

| Stage | Params | Status |
|---|---|---|
| PoC | 24.7M | Complete |
| **Mid** | **183.5M** | **Training** |
| Large | ~1.1B | Planned 2027 |

## Citation

```bibtex
@misc{keural-mid-2026,
  title={Keural Mid-Level Vision Encoder},
  author={MKD Co., Ltd.},
  year={2026},
  url={https://huggingface.co/mkd-hika/keural-vision-encoder-mid}
}
```

## License

Copyright (c) 2026 MKD Co., Ltd. All Rights Reserved.
