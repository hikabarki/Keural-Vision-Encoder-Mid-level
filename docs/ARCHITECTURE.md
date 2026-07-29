# Keural Mid-Level Vision Encoder — Architecture Specification

**183.5M parameters | Built from scratch | Not a CLIP/ViT fine-tune**

*MKD Co., Ltd. | 2026*

---

## 1. What Makes This Unique

Keural Mid introduces **three novel components** that no existing vision encoder combines:

| Innovation | What it does | Why it matters |
|---|---|---|
| **ATB (Adaptive Token Budget)** | Content-aware token selection via Gumbel-softmax top-K | Unlike CLIP/SigLIP which always output fixed N tokens, ATB allocates more tokens to complex regions and fewer to simple backgrounds |
| **HCT (Hierarchical Concept Tokenization)** | Tags every token with a semantic level (global/region/detail) | No existing encoder gives the transformer explicit knowledge of *what kind of thing* each token represents |
| **Continuous 2D-RoPE** | Rotary position encoding using actual (cx, cy) coordinates, not grid indices | Handles variable resolution natively without interpolation hacks |

These are combined in a single end-to-end differentiable pipeline — the tokenizer is the innovation, not the transformer blocks.

---

## 2. Model Summary

```
Total Parameters:    183,575,297 (183.5M)
Embedding Dimension: 768
Transformer Depth:   24 blocks
Attention Heads:     12 (head_dim=64)
Token Budget:        512 default, up to 2048
Precision:           bfloat16
Input:               RGB image, any resolution (multiple of 16)
                     Stage 1 pretrained at 384x384 and 448x448
                     Stage 2 (VLM) runs at 512x512
```

---

## 3. Architecture Pipeline

```
RGB Image (B, 3, H, W)
        |
        v
+-------------------------------+
|  Stage 1: CNN Stem            |  4 conv stages, stride 2 each
|  3 -> 128 -> 256 -> 512 -> 768|  GroupNorm + GELU at each stage
|  Total downsampling: 16x      |  Output: (B, 768, H/16, W/16)
+-------------------------------+
        |
        v
+-------------------------------+
|  Stage 2: Saliency Predictor  |  Conv(768->192, k=3) -> GELU
|  S(i,j) = sigmoid(W2*GELU(W1*x))  Conv(192->1, k=1) -> Sigmoid
|  Per-pixel importance [0,1]   |  Output: (B, 1, H/16, W/16)
+-------------------------------+
        |
        v
+-------------------------------+
|  Stage 3: ATB Tokenizer       |  Gumbel-softmax differentiable top-K
|  Budget: 512 tokens           |  NMS for spatial diversity
|  Split: 5% global             |  ROI-Align 4x4 per token position
|         25% region            |  Temperature: 1.0 -> 0.05 (annealed)
|         70% detail            |  Warmup: fixed grid for first 10K steps
+-------------------------------+
        |
        v
+-------------------------------+
|  Stage 4: HCT                 |  3 semantic levels:
|  Level 0 (global): scene ctx  |    - Learnable level embedding [3, 768]
|  Level 1 (region): objects    |    - Per-level constructor MLP:
|  Level 2 (detail): fine-grain |      Linear(16*768, 768)->GELU->Linear(768,768)
|  Output: (B, N, 768) + ids   |
+-------------------------------+
        |
        v
+-------------------------------+
|  Stage 5: Spatial Transformer |  24 Pre-norm blocks:
|  embed=768, heads=12, d=64    |    RMSNorm -> Multi-Head Attention -> Residual
|  FFN: SwiGLU (hidden=3072)    |    RMSNorm -> SwiGLU MLP -> Residual
|  Position: 2D-RoPE (cx, cy)   |  Scale bias: 8-bucket log-area on Q
|  Prepended [POOL] token       |  Variable-length attention mask from ATB
+-------------------------------+
        |
        v
  KeuralEncoderOutput
  +-- tokens           (B, N, 768)     -- per-token features
  +-- pooled           (B, 768)        -- [POOL] token = global image repr
  +-- attention_mask   (B, N)          -- valid token positions
  +-- level_ids        (B, N)          -- 0=global, 1=region, 2=detail
  +-- spatial_metadata (B, N, 4)       -- cx, cy, scale_w, scale_h in [0,1]
  +-- saliency_scores  (B, N)          -- importance score per token
```

---

## 4. Detailed Component Specifications

### 4.1 CNN Stem

| Stage | In Channels | Out Channels | Kernel | Stride | Norm | Activation |
|---|---|---|---|---|---|---|
| 1 | 3 | 128 | 3x3 | 2 | GroupNorm(8) | GELU |
| 2 | 128 | 256 | 3x3 | 2 | GroupNorm(16) | GELU |
| 3 | 256 | 512 | 3x3 | 2 | GroupNorm(32) | GELU |
| 4 | 512 | 768 | 3x3 | 2 | GroupNorm(48) | GELU |

**Design choice:** GroupNorm (not BatchNorm) for resolution-independence. Total stride=16 matches ViT-L patch size for fair comparison.

### 4.2 Saliency Predictor

```
Input:  (B, 768, H', W')     -- stem features
Layer1: Conv2d(768, 192, k=3, pad=1) -> GELU
Layer2: Conv2d(192, 1, k=1)  -> Sigmoid
Output: (B, 1, H', W')       -- per-pixel saliency in [0, 1]
```

**Regularization:**
- Anti-collapse: `1 / (var(S) + eps)` — prevents uniform saliency (all same value)
- Smoothness: `L2 total variation` — prevents noisy spike patterns
- `saliency_beta = 0.005`, `lambda_smooth = 4.0`

### 4.3 ATB Tokenizer (Adaptive Token Budget)

**Budget allocation:**

| Level | Fraction | Floor | Purpose |
|---|---|---|---|
| Global | 5% | 8 tokens min | Scene-level context |
| Region | 25% | 16 tokens min | Object/region level |
| Detail | 70% | 0 | Fine-grained features |

For default budget=512: Global=26, Region=128, Detail=358 tokens.

**Token selection mechanism:**
1. Smooth saliency map per level (kernel=7 for region, kernel=3 for detail)
2. Apply Gumbel noise (temperature annealed from 1.0 to 0.05 over 50K steps)
3. Non-Maximum Suppression (radius=3 for region, radius=1 for detail) for spatial diversity
4. Top-K selection on suppressed scores
5. ROI-Align 4x4 at each selected position -> flatten to 16*768 -> project to 768

**Warmup:** First 10K steps use deterministic uniform grid (not learned selection). ATB activates gradually via temperature annealing.

**Straight-Through Estimator:** Hard picks in forward pass, soft attention weights in backward pass — enables gradient flow through discrete selection.

### 4.4 HCT (Hierarchical Concept Tokenization)

Each token receives:
1. A **level ID** (0, 1, or 2) based on which budget pool selected it
2. A **learnable level embedding** (shape [3, 768]) added to its features
3. Processing through a **level-specific MLP**: `Linear(16*768, 768) -> GELU -> Linear(768, 768)`

This gives the transformer explicit structural information: "this token represents a global scene summary" vs "this token represents a fine detail."

### 4.5 Spatial Transformer

**24 identical blocks, each containing:**
```
x = x + Attention(RMSNorm(x))    -- pre-norm attention with residual
x = x + SwiGLU_MLP(RMSNorm(x))   -- pre-norm FFN with residual
```

**Attention:**
- Multi-head self-attention: 12 heads, head_dim=64
- QKV projection: Linear(768, 3*768)
- Output projection: Linear(768, 768)
- Variable-length mask from ATB token selection

**Position Encoding — 2D-RoPE:**
- Uses continuous (cx, cy) coordinates from the saliency map, not fixed grid indices
- `pos_scale = 100` maps [0,1] normalized coordinates to RoPE frequency regime
- `rope_base = 10000.0`
- Handles variable resolution natively without position interpolation

**Scale Bias:**
- 8 learnable buckets based on `log2(area)` of each token's spatial extent
- Shape: [8, 768], added to Q vectors
- Purpose: Distinguish tokens at same (cx, cy) but covering different scales

**FFN — SwiGLU:**
- Hidden dim: 3072 (4x expansion)
- `SwiGLU(x) = (W1*x * sigmoid(W1*x)) * W2`
- No bias in linear layers

**Pool Token:**
- Learnable [POOL] token prepended to sequence
- After 24 blocks, extracted as the global image representation (B, 768)

---

## 5. Knowledge Distillation

Two frozen teacher models provide supervision:

| Teacher | Parameters | Output Dim | Projection | Weight |
|---|---|---|---|---|
| SigLIP-SO400M (`google/siglip-so400m-patch14-384`) | 400M | 1152 | Linear(1152, 768) | lambda=0.5 |
| InternViT-300M (`OpenGVLab/InternViT-300M-448px`) | 300M | 1024 | Linear(1024, 768), bfloat16, auto-resize to 448x448 | lambda=0.5 |

**Loss:** Cosine distance between student pooled vector and teacher pooled vectors:
```
L_distill = 0.5 * mean(1 - cos_sim(student, siglip_proj))
          + 0.5 * mean(1 - cos_sim(student, internvit_proj))
```

**Why two teachers?**
- SigLIP was trained on Google's WebLI (4B pairs) — strong image-text alignment
- InternViT was trained on 1.3B pairs — different visual representations, better spatial understanding
- Together they provide richer supervision than either alone

---

## 6. Loss Function

```
L = L_SigLIP + lambda_hct * L_HCT + lambda_sal * beta * L_saliency + L_distill
```

| Component | Type | Weight | Purpose |
|---|---|---|---|
| `L_SigLIP` | Sigmoid contrastive (image-text) | 1.0 | Primary alignment objective |
| `L_HCT` | SigLIP on global-level tokens only | lambda=1.0 | Forces global tokens to capture scene meaning |
| `L_saliency` | Anti-collapse + TV smoothness | lambda=0.5, beta=0.005 | Prevents saliency collapse |
| `L_distill` | Cosine distance to frozen teachers | lambda=0.5 | Knowledge transfer from SigLIP + InternViT |

**Learnable parameters in loss:**
- `logit_scale` (init=2.30) — temperature for SigLIP contrastive
- `logit_bias` (init=-10.0) — bias term for SigLIP sigmoid

---

## 7. Output Contract (KeuralEncoderOutput)

| Field | Shape | Type | Description |
|---|---|---|---|
| `tokens` | (B, N, 768) | float | Per-token feature vectors |
| `pooled` | (B, 768) | float | Global image representation ([POOL] token) |
| `attention_mask` | (B, N) | bool | Valid token positions (1=real, 0=pad) |
| `level_ids` | (B, N) | int | Semantic level: 0=global, 1=region, 2=detail |
| `spatial_metadata` | (B, N, 4) | float | (cx, cy, scale_w, scale_h) in [0, 1] |
| `saliency_scores` | (B, N) | float | Importance score per token in [0, 1] |

---

## 8. Comparison with Existing Encoders

| Feature | CLIP ViT-L | SigLIP-SO400M | DINOv2-L | **Keural Mid** |
|---|---|---|---|---|
| Parameters | 307M | 400M | 307M | **183.5M** |
| Tokenization | Fixed 16x16 patches | Fixed 14x14 patches | Fixed 14x14 patches | **Adaptive (ATB)** |
| Token count | Fixed 256 | Fixed 729 | Fixed 729 | **Variable 512-2048** |
| Position encoding | Learned absolute | Learned absolute | Learned absolute | **2D-RoPE (continuous)** |
| Semantic levels | None | None | None | **3-level HCT** |
| Saliency-aware | No | No | No | **Yes** |
| Variable resolution | Interpolation | Interpolation | Interpolation | **Native (RoPE)** |
| Distillation | None | None | Self-distillation | **Dual-teacher KD** |

---

## 9. Code References

| Component | File | Key Class/Function |
|---|---|---|
| CNN Stem | `tokenizer/stem.py` | `ConvStem` |
| Saliency Predictor | `tokenizer/saliency.py` | `SaliencyPredictor` |
| ATB Tokenizer | `tokenizer/atb_tokenizer.py` | `ATBTokenizer` |
| HCT Token Constructor | `tokenizer/token_constructor.py` | `TokenConstructor` |
| Region Sampler | `tokenizer/region_sampler.py` | `RegionSampler` |
| Spatial Transformer | `architecture/encoder.py` | `KeuralVisionEncoder` |
| Transformer Block | `architecture/spatial/block.py` | `SpatialBlock` |
| 2D-RoPE | `architecture/spatial/rope_2d.py` | `RoPE2DFromMetadata` |
| Loss Function | `training/losses.py` | `KeuralMidLoss` |
| Distillation Teachers | `training/distillation.py` | `DistillationTeachers` |
| HF Wrapper | `alignment/hf_wrapper.py` | `KeuralVisionModel` |
| Output Dataclass | `alignment/output_contract.py` | `KeuralEncoderOutput` |
| Config | `keural_mid_config.py` | `KeuralMidConfig` |

---

## 10. Roadmap

| Stage | Parameters | Status | Hardware |
|---|---|---|---|
| PoC | 24.7M | Complete | 1x RTX 5090 |
| **Mid** | **183.5M** | **Training** | **2x H200 141GB** |
| Large | ~1.1B | Planned 2027 | 64x H100 |

---

*Copyright (c) 2026 MKD Co., Ltd. All Rights Reserved.*
