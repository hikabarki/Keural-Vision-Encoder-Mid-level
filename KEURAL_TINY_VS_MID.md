# Keural-Tiny vs. Keural-Mid — Head-to-Head

Scope: this file compares **only** the two Keural checkpoints against each other,
using exclusively what's documented in `KEURAL_TINY_BENCHMARK.md` for Tiny and
what's established in `BENCHMARK_COMPARISON.md` / this repo for Mid. No external
baselines (CLIP/SigLIP/DINOv2) are included here.

Generated 2026-07-09.

## 1. Architecture

| Property | Keural-Tiny (POC) | Keural-Mid |
|---|---|---|
| Total params | 24,666,369 (24.7M) | 183,575,297 (183.6M) — **7.4x larger** |
| Embed dim | 384 | 768 — **2x** |
| Depth | 12 | 24 — **2x** |
| Num heads / head dim | 6 / 64 | 12 / 64 |
| MLP | SwiGLU, ratio 4 | SwiGLU, ratio 4 (same) |
| Norm / positional enc. | RMSNorm / 2-D RoPE (base 10000) | RMSNorm / 2-D RoPE (base 10000) — same |
| Tokenizer mechanism | ATB (same design): stem stride-16, saliency-driven region/detail sampling, learned global query pool | ATB, identical mechanism |
| Budget split ratios | global 0.05 / region 0.25 / detail 0.70 | global 0.05 / region 0.25 / detail 0.70 — **identical ratios** |
| Budget floors | global 4 / region 8 / detail 0 | global 8 / region 16 / detail 0 — Mid's floors scale with its 2x larger default budget |
| Default token budget | 256 | 512 |
| Max token budget | 1024 | 2048 |
| Teacher / distillation | **None** — trained from scratch against a frozen `all-MiniLM-L6-v2` (384-d) text tower + trainable projection, SigLIP-style contrastive loss only | **Two frozen vision teachers** — SigLIP-SO400M (1152-d) + InternViT-300M (1024-d), cosine distillation + SigLIP contrastive + HCT |

**Reading**: Mid is a clean 2x width/2x depth scale-up of Tiny (confirmed by the
`keural_mid_config.py` comment "embed_dim: 768, depth: 24 (2× POC)"), using the
identical ATB tokenizer design and budget-split ratios. The one architectural
addition in Mid, not present in Tiny at all, is **vision-teacher distillation** —
Tiny has no vision teacher, only a frozen text encoder.

## 2. Training

| Property | Keural-Tiny (POC) | Keural-Mid |
|---|---|---|
| Dataset | CC3M + CC12M (partial) | Project's own ~20M image-text pair set |
| Dataset size | ~6.9M pairs | ~20M pairs — **~2.9x more data** |
| Steps completed | 75,000 (resumed; 0–30k log lost) | 18,000 |
| Effective batch size | 1,024 (256 × 4 accum × 1 GPU) | 3,072 (768 × 2 accum × 2 GPUs) |
| GPUs | 1× RTX 5090 | 2× H200 |
| Resolution buckets (train) | 224 / 256 / 320 | 224 / 384 / 448 / 512 |
| Token budget used in training | `None` → auto-resolved per bucket (256 @ 224/256, 400 @ 320) | **fixed** `default_token_budget`=512 regardless of resolution (confirmed at [train_mid.py:378](scripts/train_mid.py#L378)) |
| Final loss | total 2.812, primary 1.377 — **plateaued** over the last ~2,000 steps | total 2.866 — **still declining** at step 18,000 |

**Reading**: Tiny trained far longer (75K vs 18K steps) on less data (6.9M vs
20M pairs) and its loss had *plateaued* by the end — Tiny is closer to converged
for its scale/data regime than Mid is for its own. Mid is earlier in its training
trajectory (loss still falling) despite having a larger model and more data per step.
This matters for interpreting any accuracy comparison: Mid's gap to peers (documented
in `BENCHMARK_COMPARISON.md`) is a training-budget story; Tiny's position is closer
to "this specific short run has converged," not "more steps would clearly help" —
though its 6.9M-pair dataset is also small by modern standards.

Note the **opposite** token-budget philosophy: Tiny auto-resolves its budget by
resolution *during training itself* (matching what `evaluate.py`-style inference
would use), while Mid hardcodes a fixed budget in training and only auto-resolves
at inference — which is exactly the mismatch documented and resolved earlier in
`BENCHMARK_COMPARISON.md` §3. Tiny doesn't have this particular mismatch; it has a
different, worse one (see §3 below).

## 3. Evaluation

| Benchmark | Keural-Tiny (POC) | Keural-Mid |
|---|---|---|
| ImageNet zero-shot Top-1 | **N/A** — dataset not on disk, no script | 37.2% |
| ImageNet zero-shot Top-5 | **N/A** | 66.8% |
| Flickr30K I2T R@1 | **N/A** — dataset not on disk, no loader | 42.9% |
| Flickr30K T2I R@1 | **N/A** | 42.2% |
| CIFAR-100 linear probe | **N/A** — no script | 75.0% |

**Tiny has no valid accuracy numbers on any of the three standard benchmarks** —
none of the datasets or eval scripts exist in that repo. The one diagnostic number
that *was* measured (CC12M retrieval, R@1 ≈ 0.5–1.0%) is explicitly flagged in
`KEURAL_TINY_BENCHMARK.md` §5.1 as **invalid**: the trainable text projection
(`nn.Linear(384→384)` on top of frozen MiniLM) was never saved to a sidecar file,
so eval was forced to compare against *raw* MiniLM embeddings — a space the model
was never trained to align with. That number is at random chance and says nothing
about the model's real learned alignment.

This is a materially worse eval-integrity situation than anything found for Mid.
Mid's own train/eval mismatch (§3 of `BENCHMARK_COMPARISON.md`) was a token-budget
discrepancy that, once corrected, moved every metric by <1 point — a real but minor
issue. Tiny's issue is a **permanently lost artifact** (the projection weights
themselves), meaning Tiny's true retrieval/zero-shot performance cannot be
recovered from what's on disk at all, not even by re-running eval correctly.

## 4. Efficiency

| Metric | Keural-Tiny (POC) | Keural-Mid |
|---|---|---|
| Tokens/image @ default budget, native low-res | 256 @ 256×256 | 512 @ ≤256×256 |
| Naive dense stride-16 grid @ that resolution | 256 (16×16) | 256 (16×16) |
| Default budget vs. dense grid | **1.0x** — matches the dense grid exactly | **2.0x** — double the dense grid |
| FLOPs/image (measured, `FlopCounterMode`) | 14.52 GFLOPs @ 256×256, budget 256 | 202.4 GFLOPs @ 224×224, budget 512 |
| Throughput | 39.2 img/s (bs1) / 121.6 img/s (bs32), **bf16 autocast** | 66.2 img/s (bs8), **fp32, no autocast** |
| Precision measured | bf16 | fp32 |

**Not directly comparable as raw numbers** — different resolutions, batch sizes,
and precision (bf16 vs fp32) mean the FLOPs/throughput columns can't be read as
"Mid is Nx the cost of Tiny" without re-measuring both under matched conditions.

**The one clean, comparable, non-numeric finding**: Tiny's default token budget
equals the naive dense stride-16 grid count at its default resolution (1.0x —
genuinely "no more tokens than a plain grid"), while Mid's default budget is
**2x** the dense grid count at every resolution (confirmed during this session's
efficiency profiling). Both models use the identical `_resolve_default_budget()`
formula shape (`pixels/256² × default_token_budget`); the difference comes entirely
from the chosen `default_token_budget` constant (256 for Tiny vs. 512 for Mid)
relative to what a same-stride dense grid would use at the reference resolution.
Whether that's intentional (Mid trades compute for representational capacity) or
worth revisiting is an open question, not evaluated here.

## 5. Summary table (paste-ready)

| Model | Params | Training data | Resolution | ImageNet Zero-shot Top-1 | Flickr30K I2T R@1 | Flickr30K T2I R@1 |
|---|---|---|---|---|---|---|
| Keural-Tiny (POC) | 24.7M | CC3M+CC12M, ~6.9M pairs | 224/256/320px buckets | N/A¹ | N/A¹ | N/A¹ |
| Keural-Mid | 183.6M | ~20M pairs | adaptive 224–512px, budget=512 | 37.2% | 42.9% | 42.2% |

¹ Not a "model is worse" result — the benchmark is simply unmeasurable from what's
on disk for Tiny (§3). Do not read the blank cells as 0% or as evidence Tiny
underperforms Mid on these tasks; no valid measurement exists either way.

## 6. What would actually make this comparison meaningful

Right now this is an architecture/training comparison, not a capability
comparison — Tiny has zero valid accuracy numbers to compare against Mid's real
37.2%/42.9%/42.2%. To fix that captured cheaply:

1. **Recover or retrain Tiny's text projection.** If any intermediate checkpoint
   saved the full training state (`training_state.pt` mentioned in §1 of the Tiny
   doc) before the final save, the projection may be recoverable from there rather
   than requiring a retrain.
2. **Point Tiny's eval at the same benchmark data Mid already uses** — ImageNet
   parquets, Flickr30K images/captions, and CIFAR-100 are all already present on
   this machine (used throughout this session for Mid's eval). Tiny's repo just
   needs a loader pointed at them, or its checkpoint could be evaluated with a
   copy of Mid's `scripts/evaluate.py` if the two model interfaces are compatible.
3. Only after that would params/training-data/resolution differences translate
   into an actual accuracy-vs-scale story.
