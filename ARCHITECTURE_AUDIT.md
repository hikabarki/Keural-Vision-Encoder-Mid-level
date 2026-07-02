# Keural Mid-Level Vision Encoder — Full Architecture Audit

> **Method:** Reverse-engineered from source (`losses.py`, `train_mid.py`, `datasets.py`, `evaluate.py`, `atb_tokenizer.py`, `encoder.py`, `distillation.py`, config) and cross-checked against `PIPELINE_TECHNICAL_SPEC.md`. Line numbers verified against current files.
> **Output:** Findings only — no code. Severity and a repair plan are in §14–§15.
> **Headline:** The symptom (CIFAR-100 linear probe 61.78% good; ImageNet zero-shot 0.04% ≈ random; Flickr ≈ random) is *fully explained* by a cluster of **embedding-space mismatches** between training and evaluation, compounded by a **distributed-correctness bug** in how the loss/text parameters are (not) synchronized. The vision backbone is learning useful features (probe confirms); the image↔text *alignment* is never correctly evaluated, and is partially mis-trained across GPUs.

---

## 1. Architecture Consistency

| Component | Status | Notes |
|-----------|--------|-------|
| Encoder outputs (`KeuralEncoderOutput`) | ✅ Consistent | `pooled=(B,768)` from `final_norm(x)[:,0]`; fields validated by `validate()`. |
| Tokenizer outputs | ✅ Consistent | `tokens/level_ids/spatial_metadata/saliency_scores` shapes align; concat order `[global|region|detail]` matches `level_ids`. |
| Pooled embedding usage | ✅ Consistent **within training** | Same `encoder_output.pooled` feeds `L_primary` and `L_distill`. ⚠️ Diverges at eval — see §3. |
| Projection heads (image) | ⚠️ None exists | The audit checklist's `pool_token_projection Linear(768→768)` is **not implemented**. Pooling is the `[POOL]` token through 24 blocks. Not a bug, but the mental model in the checklist is wrong. |
| Transformer outputs | ✅ Consistent | Pre-norm residuals, RMSNorm, SDPA. `block_attn_mask` is all-ones inside the encoder (no padding at encoder stage). |
| HCT pipeline | ⚠️ Subtle | `hct_global_text_loss` mean-pools **only LEVEL_GLOBAL** tokens. Correct masking, but see §6 for a magnitude concern. |
| Teacher pipeline | 🔴 Dimension mismatch | InternViT: code reads `last_hidden_state[:,0]` (768-dim) but `internvit_proj` is built for **1024** (yaml/loss default). See §8. |
| Attention masks | ✅ Consistent | Encoder uses all-ones; `GlobalQueryConstructor` builds a valid_mask correctly. |
| Metadata flow | ✅ Consistent | `spatial_metadata (B,N,4)` flows into RoPE (cx,cy) and scale-bias (w·h). Pool token gets `[0.5,0.5,1,1]`. |
| RoPE inputs | ✅ Consistent | cx→first half, cy→second half, ×`pos_scale=100`. `head_dim=64` divisible by 4. ✓ |
| Level embeddings | ✅ Consistent | `level_embeddings[level_ids]` added before transformer; pad=-1 never indexed (no padding at encoder). |

**Verdict:** Architecture is internally coherent *as a forward pass*. The inconsistencies are at the **boundaries** — teacher dim wiring, and (critically) train/eval embedding space — not inside the encoder.

---

## 2. Training Pipeline Audit (image → encoder → loss)

Tracing one sample with `gradient_accumulation=2`, bf16, 2 GPUs:

1. **Image → encoder.** `pixel_values (B,3,H,W)` → `raw_model.encoder(..., step=step, return_saliency_map=True)`. ✅ gradients flow into stem, saliency, constructors, blocks, level_embeddings, pool_token.
2. **Straight-through estimator.** `RegionSampler` detaches the hard top-K/NMS picks but gathers soft saliency scores and builds soft-weight Gaussians that *are* differentiable. ✅ gradient reaches `SaliencyPredictor`. ⚠️ **But** the picked *coordinates* are detached, so the saliency map only learns "be high where I already picked," not "pick better places" — a weak STE. MEDIUM concern, not a bug.
3. **Normalization.** Image `F.normalize(pooled)`, text `F.normalize(text_embeds)` inside `KeuralMidLoss.forward`. ✅ single normalization, no double-norm in the primary loss path. (Teacher path normalizes inside `distillation_loss` — also single.)
4. **Loss + weighting.** `total = primary + λ_hct·hct + λ_sal·β_sal·saliency + distill`. Confirmed `losses.py:196-201`.
5. **Backward / accumulation.** `loss = total / grad_accum` then `.backward()`. Optimizer steps when `accum_count ≥ grad_accum`. ✅ accumulation math correct.
6. **Clipping.** `clip_grad_norm_(model.parameters(), 1.0)` — 🔴 **clips only `model` (the encoder)**. `loss_fn` params (`logit_scale`, `logit_bias`, `siglip_proj`, `internvit_proj`) and `text_encoder.proj` are **never clipped**. Confirmed `train_mid.py:315,319`. `logit_scale` in particular can spike. MEDIUM.
7. **Optimizer.** AdamW over `model + loss_fn` (group 1) and `text_encoder.proj` (group 2). ✅ all trainable params are in the optimizer.
8. **Scheduler.** `cosine_warmup_lr(step, ...)` overwrites every param group's `lr` *only on accumulation boundaries*. ✅ functionally fine.
9. **Mixed precision.** bf16 autocast, no GradScaler (correct for bf16). ✅

### Dead / missing / mis-synchronized gradients
- 🔴 **`loss_fn` and `text_encoder` are NOT wrapped in DDP.** Only `model` is (`train_mid.py:193`). Their gradients are computed locally on each rank but **never all-reduced**. With `find_unused_parameters=False` on `model` this does not crash, but `logit_scale`, `logit_bias`, both teacher projections, and the **text projection** evolve **independently on each GPU**, then get optimizer-stepped with per-rank-divergent gradients. Over 18k steps the two ranks' text/temperature spaces drift apart. This is a genuine multi-GPU correctness bug. **HIGH.**
- ✅ No dead gradients within the encoder (all modules participate; `find_unused_parameters=False` is safe because every encoder param is used).

### Step-counting semantics
- ⚠️ `step` increments **every micro-batch** (`step += 1` at loop end), but an optimizer update happens every `grad_accum` micro-batches. So `max_steps=18000`, `warmup_steps=2000`, `save_every=1000`, and `warmup_steps_fixed_grid=10000` are all counted in **micro-batches, not optimizer steps**. Real optimizer updates ≈ 9000. Warmup ends after 1000 optimizer steps. The tokenizer curriculum flips to adaptive after 5000 optimizer steps. Not wrong per se, but the schedule is half as long in optimizer-step terms than the numbers suggest. LOW–MEDIUM (interacts with §10).

---

## 3. Evaluation Pipeline Audit (checkpoint → model → benchmarks)

This is where the catastrophic zero-shot result comes from.

```
checkpoint  → KeuralVisionModel.from_pretrained()  → ENCODER ONLY (no text proj, no logit_scale/bias)
image path  → Resize384→CenterCrop→Normalize → model.pooled → F.normalize         ✅ matches training image path
text path   → raw CLIPTextModel.pooler_output → F.normalize                       🔴 MISMATCH
similarity  → img @ text.T  (plain dot product, no exp(logit_scale), no bias)     🔴 MISMATCH
```

| Check | Training | Evaluation | Match? |
|-------|----------|------------|--------|
| Image embeddings | `normalize(pooled)` | `normalize(pooled)` | ✅ |
| Text embeddings | `normalize(proj(CLIP_pooler))` | `normalize(CLIP_pooler)` — **no proj** | 🔴 NO |
| Projection layer | trainable `FrozenTextEncoder.proj` | not loaded, not applied | 🔴 NO |
| Temperature | `exp(logit_scale)` | none | 🔴 (irrelevant to argmax ranking, but see below) |
| Bias | `logit_bias` | none | 🔴 (irrelevant to ranking) |
| Normalization | L2 both sides | L2 both sides | ✅ |
| Prompt averaging | n/a (single caption) | 5 templates averaged + renorm | ✅ reasonable |
| Teacher removal | teachers train-only | teachers absent | ✅ |
| Checkpoint loading | — | encoder only | 🔴 (proj/temp/bias missing) |

**Root cause of 0.04% ImageNet / random Flickr:** During training, image embeddings are aligned to `proj(CLIP_pooler)` — a *learned linear transform* of CLIP text space. At eval, they are compared against **raw `CLIP_pooler`**. The learned alignment transform is (a) never saved (§4) and (b) never applied (`evaluate.py:153-155, 262`). The image and text vectors therefore live in **different spaces**, and ranking collapses to chance (0.04% ≈ 1/1000). Temperature/bias don't affect argmax/top-k ranking, so they're harmless for accuracy — but their absence confirms the eval was written against a different mental model than the training loss.

**Why the linear probe is fine:** CIFAR-100 trains a *fresh* `Linear(768→100)` on frozen `pooled` features. It never touches text space. Good features → 61.78%. This cleanly isolates the failure to image↔text alignment, not feature quality.

---

## 4. Checkpoint Audit — trainable parameter coverage

| Parameter | In optimizer? | Saved by `save_pretrained`? | Restored on resume? | Used at inference? | Used at eval? |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Encoder: stem, saliency, constructors | ✅ | ✅ `encoder.*` | ✅ | ✅ | ✅ |
| Encoder: `level_embeddings`,`pool_token` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Encoder: 24 blocks, `final_norm` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FrozenTextEncoder.proj` (text projection) | ✅ (group 2) | 🔴 **NO** | 🔴 re-init | n/a (text-side) | 🔴 **NOT applied** |
| `logit_scale` | ✅ | 🔴 **NO** | 🔴 re-init | n/a | 🔴 not used |
| `logit_bias` | ✅ | 🔴 **NO** | 🔴 re-init | n/a | 🔴 not used |
| `siglip_proj.weight` | ✅ | 🔴 **NO** | 🔴 re-init | n/a (train-only) | n/a |
| `internvit_proj.weight` | ✅ | 🔴 **NO** | 🔴 re-init | n/a (train-only) | n/a |
| Optimizer state | — | ✅ `training_state.pt` | ✅ | — | — |
| CLIP text backbone | frozen | 🔴 NO (re-downloaded) | ✅ (HF cache) | — | ✅ (re-loaded) |

**Findings:**
- 🔴 **CRITICAL: the text projection is the linchpin of image↔text alignment and is never persisted.** Even if eval applied a projection, there's no saved tensor to apply. Confirmed `train_mid.py:382,398` save only `raw_model`.
- 🔴 **Resume silently re-initializes** `text_encoder.proj`, `logit_scale`, `logit_bias`, and both teacher projections (`_resume_from_checkpoint` loads only the encoder + optimizer). After a resume, the optimizer state (Adam moments) refers to freshly-randomized params → training is corrupted on resume. HIGH.
- These are stored only in module instances that live in `train_mid.py` and vanish when the process ends.

---

## 5. Distributed Training Audit

| Aspect | Finding | Severity |
|--------|---------|----------|
| DDP wrap | Only `model` (encoder) is wrapped. `loss_fn`, `text_encoder` are bare modules. | 🔴 HIGH |
| Gradient sync (encoder) | ✅ all-reduced by DDP. | — |
| Gradient sync (loss/text params) | 🔴 **never all-reduced** → per-rank divergence of `logit_scale/bias`, projections, text proj. | HIGH |
| Contrastive negatives | `image_embeds @ text_embeds.T` over **local batch only**; no `all_gather`. | 🔴 HIGH |
| `all_gather` | absent entirely. | — |
| Effective optimizer batch | `576 × 2 × 2 = 2304`. | — |
| Effective contrastive negatives | **576** per term (local micro-batch), not 2304. | quality loss |
| Communication | DDP grad all-reduce + `dist.barrier()` after checkpoints. | ✅ |
| `find_unused_parameters` | `False` — safe for encoder (all used), but masks the un-synced loss/text params (they're simply outside DDP). | note |

**Scaling blockers:** (1) no embedding gather means adding GPUs does **not** increase the contrastive negative pool — the main lever SigLIP/CLIP rely on. (2) The un-synced loss/text params mean more GPUs = more divergence, not better estimates. SigLIP is more batch-robust than softmax-InfoNCE, so this is quality-degrading rather than fatal — but combined with §3 it means the alignment that *is* learned is also weaker than the logs imply.

---

## 6. Loss Function Audit

| Loss | Inputs | Norm | Temp/Bias | Reduction | Weight (effective) | Grad flow | Magnitude risk |
|------|--------|------|-----------|-----------|--------------------|-----------|----------------|
| **Primary SigLIP** | `pooled`, `text_embeds` | L2 both | `exp(logit_scale)`, `logit_bias` | `sum/B` | 1.0 | encoder, text proj, logit_scale/bias | scales with B (sum over B² ÷ B ⇒ ~O(B) magnitude) |
| **HCT** | GLOBAL tokens pooled, `text_embeds` | L2 both | same scale/bias | `sum/B` | `λ_hct=1.0` | encoder, text proj, logit_scale/bias | **same magnitude as primary** → effectively *doubles* the contrastive pressure, and **shares `logit_scale`** with primary |
| **Saliency** | `saliency_map` | — | — | mean | `λ_sal·β_sal = 0.5×0.005 = 0.0025` | saliency predictor, stem (weak STE) | `1/(var+eps)` can be **very large** early when var≈0 (eps=1e-3 ⇒ up to ~1000 before weighting ⇒ ~2.5 after). Could dominate at init. |
| **Distill** | `pooled`, teacher proj | L2 inside | — | mean cos-dist | `λ_distill=0.5` → 0.25 each | encoder, teacher projs | bounded [0,2], well-behaved |

**Findings:**
- ⚠️ **`logit_scale` is shared between Primary and HCT** (both call `sigmoid_siglip_loss(..., self.logit_scale, self.logit_bias)`). Two losses pushing one temperature in possibly different directions. MEDIUM.
- ⚠️ **HCT ≈ Primary in form and magnitude with weight 1.0** → the contrastive objective is effectively counted twice (once on `pooled`, once on GLOBAL-token mean-pool). Since GLOBAL tokens and `pooled` are different tensors, this trains two image representations against text. Could split capacity. MEDIUM.
- ⚠️ **Saliency anti-collapse `1/(var+eps)` with eps=1e-3** is unbounded as var→0. At init saliency≈0.5 everywhere ⇒ var≈0 ⇒ term≈1000, ×0.0025 ≈ 2.5, comparable to other losses. Tapers as variance grows. Monitor early-step `loss/saliency`. MEDIUM.
- ✅ Reduction and normalization are internally consistent across losses.
- 🔴 **All contrastive losses inherit the train/eval space mismatch (§3)** — they optimize alignment to `proj(text)` which eval never uses.

**Dominance/vanishing verdict:** No single loss numerically dominates by orders of magnitude given the weights, but **HCT double-counts contrastive** and **saliency can spike early**. The real problem isn't balance — it's that the contrastive signal is aimed at a space the eval ignores.

---

## 7. Text Pipeline Audit (caption → … → eval)

```
TRAINING:   caption → CLIPTokenizer(77) → CLIPTextModel(frozen) → pooler_output(768)
                    → proj (TRAINABLE 768→768) → [norm in loss] → contrastive
CHECKPOINT: proj NOT saved                                                          🔴
EVAL:       caption → CLIPTokenizer → CLIPTextModel → pooler_output → norm           🔴 no proj
INFERENCE:  (text side not used by the vision encoder at inference)
```

- 🔴 The text embedding space is **not preserved** across the train→checkpoint→eval boundary. Training defines text space as `proj(pooler)`; eval uses `pooler`. The whole alignment hinges on a 768×768 matrix that is discarded. **CRITICAL.**
- ⚠️ `FrozenTextEncoder.forward` returns `proj(raw.clone())` **without** L2 norm; normalization happens in the loss. Consistent for training, but any consumer that forgets to normalize (e.g. a future eval that *does* load proj) must replicate it. Fragile. LOW.
- ✅ Tokenizer (`max_length=77`), backbone, and frozen-ness match between train and eval (both `openai/clip-vit-large-patch14`).

---

## 8. Teacher Distillation Audit

| Check | Finding | Severity |
|-------|---------|----------|
| SigLIP dim | returns 1152; `siglip_proj: Linear(1152→768)`. ✅ matches. | — |
| InternViT dim | `distillation.py` returns `last_hidden_state[:,0]` = **768**, but `internvit_proj` is built for **1024** (yaml `internvit_embed_dim:1024`, also `KeuralMidLoss` default). | 🔴 |
| → Consequence | `internvit_proj(teacher_internvit_pooled)` would be a **shape mismatch (768 vs 1024) → runtime error**, *unless* the InternViT model actually emits 1024-dim hidden states (InternViT-300M hidden size is 1024). The spec's "768 per code" reading of `[:,0]` is the **last_hidden_state channel dim**, which for InternViT-300M is **1024**. So the code likely works, and the dataclass default (768) is the stale value. | needs runtime confirmation |
| Loss | cosine distance, normalized, mean over batch. ✅ | — |
| Checkpointing | `siglip_proj`/`internvit_proj` **not saved** (train-only, acceptable) but **re-init on resume corrupts distillation after resume**. | MEDIUM |
| Mixed precision | InternViT cast to bf16; SigLIP default; outputs cast back. ✅ | — |
| Teacher freezing | `requires_grad_(False)` + eval mode + `@torch.no_grad()`. ✅ solid. | — |
| Projection training | trainable, in optimizer. ✅ | — |

**Action:** confirm InternViT-300M's actual `hidden_size` (almost certainly 1024). If 1024, fix the **dataclass default** (768→1024) for consistency; the active yaml is already correct. If the model truly emits 768, the yaml is wrong. This is a config-truth question, resolvable by one runtime print or checking the HF config.

---

## 9. Configuration Audit

| Field | Status |
|-------|--------|
| `tokenizer.roi_align_size: 4` | 🔴 **Unused.** No ROI-Align in the tokenizer; weighted pooling is used. Dead config. |
| `loss.hct_alpha: 0.5` | 🔴 **Unused.** `KeuralMidLoss` never reads `hct_alpha` (HCT uses the same sigmoid loss, no alpha). Dead config. |
| `loss.primary: "sigmoid_siglip"` | ⚠️ Partially used. Only `sigmoid_siglip` is implemented; `infonce_clip` is a validated choice with no code path. |
| `spatial.use_pool_token: true` | ⚠️ Not branched on — pool token is always added regardless. Dead toggle. |
| `distillation.internvit_embed_dim` | ⚠️ Default (768) ≠ yaml (1024) ≠ possibly model. See §8. |
| `training.optimizer: "adamw"` | ⚠️ Only adamw implemented; no branch. |
| `training.lr_schedule: "cosine_with_warmup"` | ⚠️ `cosine_warmup_lr` is always used; `constant`/`linear_with_warmup` choices have no code path. |
| `loss.saliency_beta` vs `lambda_sal` | ✅ Both used (multiplied). |
| `data.extra_sources: []` | ✅ Used when non-empty. |
| All `stem/saliency/spatial/tokenizer` numeric fields | ✅ Used. |

**Dead config:** `roi_align_size`, `hct_alpha`, `use_pool_token` (as a toggle). **Phantom choices:** `primary`, `optimizer`, `lr_schedule` enums accept values with no implementation.

---

## 10. Hyperparameter / Schedule Audit (can schedules complete?)

Recall (§2) `step` counts **micro-batches**; `grad_accum=2`, so optimizer steps ≈ `step/2`.

| Schedule | Configured | Completes within run? |
|----------|-----------|-----------------------|
| LR warmup | `warmup_steps=2000` (micro) ≈ 1000 opt-steps | ✅ completes early. |
| LR cosine | over `max_steps=18000` (micro) | ✅ reaches `0.05·lr` floor at end. |
| Tokenizer warmup (fixed grid) | `warmup_steps_fixed_grid=10000` (micro) ≈ 5000 opt-steps | ✅ flips to adaptive at micro-step 10000 (= 8000 micro-steps before end). |
| Gumbel temperature anneal | `1.0→0.05` over `temperature_anneal_steps=50000` (micro), starting at step 10000 | 🔴 **Cannot complete.** Only steps 10000–18000 (8000 micro-steps) occur after warmup. Progress = 8000/50000 = 16%. Final Gumbel temp ≈ `1.0 + 0.16·(0.05−1.0) ≈ 0.85`. The sampler stays **noisy/soft** the entire run; it never sharpens toward discrete selection. **HIGH for tokenizer quality.** |

**Key finding:** The adaptive tokenizer gets ~8000 micro-steps of *high-temperature* (≈0.85–1.0) sampling and never anneals to selective behavior. Either `temperature_anneal_steps` must shrink (~6000) or `max_steps` must grow, or warmup must shorten. As-is, the headline feature (adaptive token budget) is barely exercised.

---

## 11. Data Pipeline Audit

| Aspect | Train | Eval | Match? |
|--------|-------|------|--------|
| Resize/crop | `RandomResizedCrop(0.75–1.0, BICUBIC)` + HFlip + ColorJitter | `Resize(384)→CenterCrop(384)` | ⚠️ expected (aug on train) |
| Resolution | buckets `[384,384],[448,448]` | fixed **384** for all evals | ⚠️ 448-trained behavior never eval'd; minor |
| Normalization | ImageNet mean/std in collate | ImageNet mean/std in eval transform | ✅ identical |
| Image range | `ToTensor`→[0,1] then normalize | same | ✅ |
| Caption | raw string → CLIP tokenizer (77) | raw string → CLIP tokenizer | ✅ |
| Bucketing | `BucketedSampler` homogeneous batches, shard-local, windowed shuffle | n/a | ✅ |
| Failure handling | 8-retry then **black image** with original caption | n/a | ⚠️ silent: a corrupt sample becomes a black-image/caption *positive pair* — injects label noise. LOW. |

**Findings:** Train/eval image preprocessing is **consistent** (good — rules out a normalization cause for the zero-shot collapse). Two minor issues: (1) eval only at 384 though training includes 448; (2) decode-failure black-image fallback creates mislabeled positives. Neither explains the headline failure.

---

## 12. Potential Silent Bugs (no exception, hurts accuracy)

1. 🔴 **Wrong text embedding space at eval** (no projection applied). Silent; collapses zero-shot. (§3, §7)
2. 🔴 **Text projection never checkpointed** → alignment unrecoverable even if eval were fixed. (§4)
3. 🔴 **Loss/text params not DDP-synced** → per-rank divergence of temperature/bias/projections. Silent; degrades alignment, worsens with more GPUs. (§5)
4. 🔴 **No `all_gather`** → contrastive negatives capped at local batch. Silent; weaker alignment. (§5)
5. 🔴 **Resume re-initializes loss/text params** while loading their stale Adam moments → silent training corruption after any resume. (§4)
6. 🟠 **Gumbel temperature never anneals** → tokenizer stays soft/noisy the whole run. Silent; suboptimal tokens. (§10)
7. 🟠 **Grad clipping skips loss/text params** → `logit_scale` can spike unclipped. (§2)
8. 🟠 **HCT double-counts contrastive** on a different image tensor with the shared temperature. (§6)
9. 🟠 **InternViT dim default (768) disagrees** with yaml/model (1024) — works only if the active config wins; the dataclass default is a trap for anyone running without the yaml. (§8)
10. 🟡 **Black-image fallback** injects mislabeled positive pairs. (§11)
11. 🟡 **Weak STE**: saliency learns "be confident where picked," limited "pick better." (§2)
12. 🟡 **Eval resolution (384) ≠ part of training (448).** (§11)

---

## 13. Code Smells

- **Dead config fields:** `roi_align_size`, `hct_alpha`, `use_pool_token` toggle; phantom enum choices (`primary`, `optimizer`, `lr_schedule`). (§9)
- **`_rotate_checkpoints` is a no-op** (`pass`) — disk fills unbounded over a long run.
- **Two parallel "save" mental models:** `save_pretrained` (encoder) vs nothing-for-loss/text — fragile; a refactor that "just loads the model" will keep silently dropping alignment params.
- **`FrozenTextEncoder` returns unnormalized embeddings** while every consumer expects to normalize — easy to misuse in a future eval.
- **Hardcoded eval data paths** in `evaluate.py` (ImageNet/Flickr roots) — brittle.
- **`step` semantics** (micro-batch vs optimizer-step) is implicit and leaks into every schedule constant; a future reader will misread the schedule lengths.
- **Teacher dim wiring** spread across three places (dataclass default, yaml, model) with no single source of truth.
- **`primary` sign convention / reduction** is correct but undocumented in-code; the `sum/B` (not `mean`) makes loss magnitude batch-size-dependent — fragile when batch size changes.

---

## 14. Severity Ranking & Expected Impact

| # | Issue | Severity | ImageNet ZS | Retrieval | Lin. probe | Train stability | Reproducibility | Distributed scaling |
|---|-------|:--------:|:-----------:|:---------:|:----------:|:---------------:|:---------------:|:-------------------:|
| 1 | Eval uses raw CLIP text, no projection | **CRITICAL** | **fatal** (0.04%) | **fatal** | none | none | none | none |
| 2 | Text projection not checkpointed | **CRITICAL** | fatal (blocks any fix to #1) | fatal | none | none | **breaks** | none |
| 3 | Loss/text params not DDP-synced | **HIGH** | degrades | degrades | none | mild | breaks (rank-dependent) | **breaks** |
| 4 | No `all_gather` of embeddings | **HIGH** | degrades | degrades | none | none | none | **caps gains** |
| 5 | Resume re-inits loss/text params | **HIGH** | degrades after resume | degrades | none | **corrupts** | **breaks** | none |
| 6 | Gumbel temp never anneals | **HIGH** | mild–moderate | mild–moderate | mild | none | none | none |
| 7 | Grad clip skips loss/text params | MEDIUM | mild | mild | none | mild–moderate | none | none |
| 8 | HCT double-counts contrastive | MEDIUM | mild | mild | mild | none | none | none |
| 9 | InternViT dim default mismatch | MEDIUM | (crash if default used) | — | — | — | trap | — |
| 10 | `1/(var+eps)` saliency spike | MEDIUM | mild | mild | none | mild early | none | none |
| 11 | Black-image fallback | LOW | negligible | negligible | negligible | none | none | none |
| 12 | Eval-only-384 | LOW | mild | mild | mild | none | none | none |
| 13 | Dead config / no-op rotate | LOW | none | none | none | none | (disk) | none |

**The 0.04% ImageNet result is caused almost entirely by #1 (+#2).** Everything else explains why the alignment that *would* show up after fixing #1 is weaker than the training curves suggest (#3, #4, #6).

---

## 15. Repair Plan (ordered; no code — implementation on request)

> Ordering principle: make evaluation *measure the real model* first (#1–#2) so you can see the true baseline, then fix the *training correctness* issues that improve that baseline (#3–#5), then quality (#6–#10).

### Phase 0 — Make evaluation truthful (do first, cheapest, highest information)
**R1. Save & load the full alignment state, and apply the text projection at eval.**
- **Why:** Without this, every zero-shot/retrieval number is meaningless; you cannot tell if any other fix helped.
- **Files:** `train_mid.py` (checkpoint write), `evaluate.py` (text path), optionally a small `alignment_state.pt` next to `pytorch_model.bin`.
- **Expected impact:** ImageNet ZS and Flickr jump from ≈random to a real (possibly modest) number. This is the single most important change.
- **Risk:** Low. Additive.
- **Checkpoint compatibility:** Existing checkpoints **lack the projection/temp/bias** — they cannot be evaluated correctly retroactively. New checkpoints will carry it.
- **Retraining:** Not strictly required to *test* the mechanism, but existing checkpoints are effectively unevaluable for alignment; a fresh (or resumed-then-extended) run is needed to get a trustworthy number.
- **Depends on:** nothing.

### Phase 1 — Training correctness (these change what the weights learn)
**R2. Persist `logit_scale`, `logit_bias`, `siglip_proj`, `internvit_proj`, `text_encoder.proj`; load them on resume.**
- **Why:** Resume currently corrupts these (re-init + stale Adam moments). Also prerequisite for R1's save side.
- **Files:** `train_mid.py` (`_save_training_state`, `_resume_from_checkpoint`).
- **Impact:** Correct resume; reproducibility.
- **Risk:** Low. **Checkpoint compat:** old `training_state.pt` lacks these keys — load defensively. **Retraining:** not required (forward-compatible).
- **Depends on:** none (pairs with R1).

**R3. Synchronize loss/text parameters across ranks.**
- **Why:** They currently diverge per-GPU. Either wrap `loss_fn`+`text_encoder` in DDP, or `all_reduce` their grads manually, or broadcast from rank 0 each step.
- **Files:** `train_mid.py`.
- **Impact:** Correct multi-GPU training; removes silent rank-dependence.
- **Risk:** Medium (DDP wrapping interacts with `find_unused_parameters`). **Checkpoint compat:** none. **Retraining:** **yes** to benefit.
- **Depends on:** independent, but test with R4.

**R4. Add `all_gather` of image & text embeddings before the similarity matrix (with gradient-preserving gather).**
- **Why:** Restores the large negative pool SigLIP relies on; makes GPUs actually help.
- **Files:** `losses.py` (`sigmoid_siglip_loss` / `KeuralMidLoss.forward`), training loop.
- **Impact:** Stronger alignment; better retrieval; real distributed scaling.
- **Risk:** Medium (must use a differentiable all-gather; keep local rows' gradients). **Checkpoint compat:** none. **Retraining:** **yes**.
- **Depends on:** R3 (sync) for coherent cross-rank logits.

**R5. Clip loss/text parameter gradients too.**
- **Why:** `logit_scale` can spike unclipped.
- **Files:** `train_mid.py` (include `loss_fn`/`text_encoder` params in `clip_grad_norm_`).
- **Impact:** Stability. **Risk:** Low. **Retraining:** beneficial, not required. **Depends on:** none.

### Phase 2 — Quality / schedules
**R6. Fix the Gumbel anneal schedule so it completes within `max_steps`.**
- **Why:** Tokenizer never sharpens (anneals only ~16%). Reconcile micro-step vs optimizer-step counting while here.
- **Files:** `keural_mid_config.py` / yaml (`temperature_anneal_steps`, `warmup_steps_fixed_grid`), `atb_tokenizer.py` (decide step unit).
- **Impact:** Better, more selective tokens. **Risk:** Low. **Retraining:** **yes** to benefit. **Depends on:** decide step-unit convention first.

**R7. Resolve InternViT teacher dim as a single source of truth.**
- **Why:** Dataclass default (768) ≠ yaml (1024) ≠ verify model. One runtime check; set all three consistently.
- **Files:** `keural_mid_config.py`, `configs/*.yaml`, possibly `distillation.py`.
- **Impact:** Removes a crash trap for default-config runs. **Risk:** Low. **Retraining:** no (if yaml already correct). **Depends on:** confirm model hidden size.

**R8. Reconsider HCT weight / shared temperature.**
- **Why:** HCT double-counts contrastive at weight 1.0 sharing `logit_scale`. Consider a separate temperature or lower weight.
- **Files:** `losses.py`, config. **Impact:** Cleaner optimization. **Risk:** Medium (changes loss landscape). **Retraining:** **yes**. **Depends on:** evaluate after R1–R4 to see if HCT even helps.

**R9. Saliency anti-collapse stabilization.**
- **Why:** `1/(var+eps)` can spike early. Consider larger eps, a log form, or warmup on `lambda_sal`. **Files:** `losses.py`/config. **Impact:** Early stability. **Risk:** Low. **Retraining:** beneficial. **Depends on:** none.

### Phase 3 — Hygiene
**R10. Remove/implement dead config (`roi_align_size`, `hct_alpha`, `use_pool_token` toggle, phantom enums); implement `_rotate_checkpoints` or document unbounded disk; parameterize eval data paths; normalize-or-document `FrozenTextEncoder` output.**
- **Impact:** Maintainability, fewer future foot-guns. **Risk:** Low. **Retraining:** no. **Depends on:** none.

### Recommended execution order
`R1 + R2` (truthful eval, get real baseline) → `R3 + R4 + R5` (distributed correctness, the big training-quality lever) → re-measure → `R6` (tokenizer), `R7` (teacher dim) → `R8/R9` (loss tuning, only if data says so) → `R10` (hygiene).

**Critical compatibility note:** existing checkpoints (`cc3b10c` run) **cannot be correctly evaluated** for zero-shot/retrieval — the alignment parameters were never saved. To obtain a trustworthy alignment number you will need at least a short retrain (or extend) after R1–R2, and the real quality gains require R3–R4 (retrain).

---

*End of audit. No code modified. Awaiting go-ahead for implementation.*
