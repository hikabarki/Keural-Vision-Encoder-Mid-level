# Keural Mid-Level Vision Encoder

**183.5M parameters · ATB Tokenizer · HCT · Spatial Transformer · Knowledge Distillation**

> Built from scratch by MKD Co., Ltd. — not a fine-tuned CLIP variant.

---

## Overview

Keural Mid is the second stage of the Keural Vision Encoder roadmap, scaling from the 24.7M PoC to a 183.5M production-grade encoder. It replaces fixed patch tokenization with **Adaptive Token Budget (ATB)** — a content-aware, differentiable token selection mechanism — and introduces **Hierarchical Concept Tokenization (HCT)** to give the transformer explicit semantic-level signals per token.

| | PoC | **Mid (this repo)** | Large (planned) |
|---|---|---|---|
| Parameters | 24.7M | **183.5M** | ~1.1B |
| embed_dim | 512 | **768** | 1280 |
| Transformer depth | 12 | **24** | 40 |
| Token budget | 256 | **512 (max 2048)** | 1024 (max 4096) |
| Hardware | 1× RTX 5090 | **2× H200 141GB** | 64× H100 |
| Text encoder | MiniLM-L6-v2 | **CLIP ViT-L/14** | CLIP ViT-L/14 |
| Status | ✅ Complete | 🔄 Training | 📅 2027 |

---

## Architecture

```
RGB Image (B × 3 × H × W)
        │
        ▼
┌───────────────────────────────┐
│  CNN Stem  (4-stage, stride 2)│  3→128→256→512→768 ch, GELU, GroupNorm
│  Total downsampling: 16×      │  Output: B × 768 × (H/16) × (W/16)
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Saliency Predictor           │  Conv(768→192, k=3) → GELU → Conv(192→1, k=1) → Sigmoid
│  S(i,j) = σ(W₂·GELU(W₁·x))  │  Output: B × 1 × H' × W'
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  ATB Tokenizer                │  Gumbel-softmax top-K · NMS · ROI-align 4×4
│  Budget: 512 default / 2048   │  Split: 5% global / 25% region / 70% detail
│  max. Differentiable at train │  Warmup: fixed grid for first 10K steps
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  HCT  (Hierarchical Concept   │  3 levels: global(0) · region(1) · detail(2)
│  Tokenization)                │  Per-level: learnable embedding + constructor MLP
│                               │  Output: B × N × 768 + level_ids B × N
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Spatial Transformer (24L)    │  RMSNorm · SwiGLU FFN · 2D-RoPE (continuous cx,cy)
│  embed=768, heads=12, d=64    │  Scale bias: 8-bucket log-area embedding on Q
│  Pre-norm, pool token         │  Variable-length attention_mask from ATB
└───────────────────────────────┘
        │
        ▼
  KeuralEncoderOutput
  ├── tokens           B × N × 768
  ├── pooled           B × 768
  ├── attention_mask   B × N
  ├── level_ids        B × N
  ├── spatial_metadata B × N × 4  (cx, cy, scale_w, scale_h)
  └── saliency_scores  B × N
```

---

## Key Design Decisions

### Adaptive Token Budget (ATB)
Unlike CLIP/SigLIP which always produce a fixed N tokens, ATB selects tokens based on saliency at runtime. The budget split (5/25/70) ensures global scene context is always represented while dedicating most capacity to fine-grained detail. Gumbel-softmax makes the selection fully differentiable during training.

### Hierarchical Concept Tokenization (HCT)
Every token carries an explicit semantic level `{0=global, 1=region, 2=detail}`. Each level has its own learnable embedding and token constructor MLP, giving the transformer structural information about what each token represents — something no existing encoder does.

### 2D-RoPE with Continuous Coordinates
Position encoding uses the actual `(cx, cy)` spatial coordinates from the saliency map, not a fixed grid. This means tokens at different resolutions or crop positions are always correctly positioned relative to each other.

### Knowledge Distillation
Two frozen teacher models guide training without requiring proprietary datasets:
- `google/siglip-so400m-patch14-384` (1152-dim, projected to 768)
- `OpenGVLab/InternViT-300M-448px` (1024-dim, projected to 768, bfloat16, auto-resized to 448×448)

This transfers WebLI + LAION knowledge into the student without downloading those datasets.

---

## Repository Structure

```
keural-vlm-mid/
├── architecture/
│   ├── encoder.py              # KeuralMidEncoder — main model
│   └── spatial/
│       ├── block.py            # TransformerBlock (RMSNorm, SwiGLU, 2D-RoPE)
│       └── rope_2d.py          # Continuous 2D rotary position encoding
├── tokenizer/
│   ├── stem.py                 # CNN Stem (4-stage strided conv)
│   ├── saliency.py             # Saliency Predictor (2-conv head)
│   ├── atb_tokenizer.py        # ATB: Gumbel top-K + NMS + ROI-align
│   ├── token_constructor.py    # HCT per-level MLPs + level embeddings
│   └── region_sampler.py       # Region/global token sampling utilities
├── training/
│   ├── losses.py               # SigLIP + HCT + Saliency + Distillation loss
│   └── distillation.py         # Teacher model wrappers (SigLIP, InternViT)
├── alignment/
│   ├── hf_wrapper.py           # HuggingFace PreTrainedModel wrapper
│   └── output_contract.py      # KeuralEncoderOutput dataclass
├── data/
│   └── datasets.py             # DataComp+COYO dataset + BucketedSampler (DDP-ready)
├── configs/
│   ├── keural_mid_2xh200.yaml        # Primary training config (2× H200)
│   ├── keural_mid_4xh100.yaml        # Alternative training config (4× H100)
│   ├── keural_mid_single_gpu.yaml    # Single RTX 5090 config (debug/iteration)
│   ├── keural_mid_smoke.yaml         # Smoke test config (synthetic data)
│   └── keural_mid_architecture.json  # Full architecture spec in JSON
├── scripts/
│   ├── train_mid.py            # Main training script (DDP, NCCL)
│   └── smoke_test_mid.py       # 5-step sanity check (✅ PASSED, 183.5M params)
└── keural_mid_config.py        # Typed config dataclass (YAML → Python)
```

---

## Training

### 2× H200 (Primary)
```bash
torchrun --nproc_per_node=2 \
    scripts/train_mid.py \
    --config configs/keural_mid_2xh200.yaml \
    --output-dir outputs/run \
    --max-samples 20000000
```
Effective batch: `576 × 2 accum × 2 GPUs = 2304` · Est. time: ~6.5 days

### 4× H100 (Alternative)
```bash
torchrun --nproc_per_node=4 \
    scripts/train_mid.py \
    --config configs/keural_mid_4xh100.yaml \
    --output-dir outputs/run
```
Effective batch: `128 × 4 accum × 4 GPUs = 2048` · Est. time: 4–5 days

### Single GPU (Debug / Iteration)
```bash
python3.10 scripts/train_mid.py \
    --config configs/keural_mid_single_gpu.yaml \
    --output-dir outputs/run
```
Effective batch: `8 × 32 accum = 256` · Est. time: ~17 days (50K checkpoint recommended)

### Smoke Test
```bash
cd /home/mkd-s002/vision-encoder/mid
python3.10 scripts/smoke_test_mid.py
# Expected: ALL 5 STEPS PASSED. Parameters: 183.5M
```

---

## Loss Function

```
L = L_SigLIP + λ_hct · L_HCT + λ_sal · L_saliency + λ_distill · L_distill
```

| Component | Formula | Weight |
|---|---|---|
| `L_SigLIP` | Sigmoid contrastive (image ↔ text) | — |
| `L_HCT` | SigLIP on global-level tokens only | λ=1.0 |
| `L_saliency` | `1/(var(S)+ε)` + L2 TV | λ=0.5, β=0.005 |
| `L_distill` | Cosine distance: `mean(1 - cos_sim(student, teacher))` for SigLIP + InternViT | λ=0.5 |

---

## Dataset

**Stage 1** (current): 20M curated pairs from DataComp-1B Medium (`--max-samples 20000000`), 2 full epochs, ~18K steps.
**Stage 2** (planned): Resume from Stage 1 checkpoint, add COYO-700M Filtered (~32M pairs) for combined ~52M pairs.

| Dataset | Downloaded | Used (Stage 1) | Storage | Path |
|---|---|---|---|---|
| DataComp-1B Medium | ~89M | **20M curated** | 3.6TB | `/home/work/keural-model/vision-encoder-dataset/datacomp_medium` |
| COYO-700M Filtered | ~32M | Reserved (Stage 2) | 2.0TB | `/home/work/keural-model/vision-encoder-dataset/coyo_filtered` |

---

## License

Copyright (c) 2026 MKD Co., Ltd. All Rights Reserved.
Contact: sales@mkd.kr
