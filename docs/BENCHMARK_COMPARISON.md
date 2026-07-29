# Keural Mid Vision Encoder — Benchmark Comparison

Comparison of the Keural Mid vision encoder (checkpoint: `outputs/run/final`, step 18000)
against its distillation teachers and same-parameter-class public vision encoders.

Generated 2026-07-09. Numbers below use `token_budget=512`, matching training exactly
(training always called the encoder with the fixed config default, never the
resolution-based auto-scaling `evaluate.py` uses when no budget is passed —
see §3). Re-running eval at the matched budget moved every metric by <1 point
versus the original auto-scaled (1152-token) run, so the model is not
budget-sensitive in this range; these are simply the more methodologically
correct numbers to report.

## 1. Your model vs. distillation teachers

| Model | Params | Training data | Resolution | ImageNet Zero-shot Top-1 | Flickr30K I2T R@1 | Flickr30K T2I R@1 |
|---|---|---|---|---|---|---|
| **Keural Mid (student)** | 183.6M | ~20M image-text pairs, 18K steps | adaptive, 224–512px (ATB), budget=512 | 37.2% | 42.9% | 42.2% |
| SigLIP-SO400M-384 (teacher) | ~400M | WebLI (~10B image-text pairs seen) | fixed 384px | 83.1% | ~89.1%¹ | ~75.6%¹ |
| InternViT-300M-448px (teacher) | ~300M | distilled from InternViT-6B, no native text tower | fixed 448px | n/a (no text tower; ImageNet linear-probe ~85.1%) | n/a | n/a |

¹ Reference number from a large-scale SigLIP variant (10B samples seen), not confirmed to be the exact `so400m-patch14-384` checkpoint — directional only.

**Retained capacity**: your student keeps ~45% of the SigLIP teacher's zero-shot ImageNet accuracy and ~48–56% of its Flickr30K retrieval R@1, after only 18K steps on ~20M pairs vs. teachers trained on billions of pairs.

## 2. Your model vs. same-parameter-class public encoders

| Model | Params | Training data | Resolution | ImageNet Zero-shot Top-1 | ImageNet Linear Probe | Flickr30K I2T R@1 | Flickr30K T2I R@1 |
|---|---|---|---|---|---|---|---|
| **Keural Mid (student)** | 183.6M | ~20M pairs, 18K steps | adaptive, 224–512px, budget=512 | 37.2% | — (75.0% on CIFAR-100 instead) | 42.9% | 42.2% |
| CLIP ViT-B/16 (OpenAI) | ~86M | WIT-400M pairs | fixed 224px | 68.3–68.6% | — | 88.2%² | 29.9%² |
| CLIP ViT-B/16 (CC12M variant) | ~86M | CC12M (~12M pairs) | fixed 224px | — | — | 62.6%³ | 45.4%³ |
| SigLIP ViT-B/16 (original) | ~93M | WebLI subset | fixed 224/256px | 76.7% | — | — | — |
| DINOv2 ViT-S/14 (self-supervised) | 21M | LVD-142M (curated, no text) | fixed 224/518px | n/a (no text tower) | 79.0% | n/a | n/a |

² OpenAI CLIP zero-shot retrieval numbers from one reported evaluation setup.
³ Same architecture (ViT-B/16) but trained on far less data (CC12M) — shows how strongly retrieval numbers move with training-data scale, useful as a lower-data reference point.

## 3. Notes and caveats

- **Train/eval token-budget mismatch, investigated and resolved.** Training
  ([train_mid.py:378](scripts/train_mid.py#L378)) always called the encoder with
  a fixed `token_budget=config.tokenizer.default_token_budget` (512), regardless
  of image resolution. `evaluate.py` originally called the model with no
  `token_budget` argument, which falls through to `_resolve_default_budget()` —
  resolution-based auto-scaling that yields **1152** tokens at the 384px eval
  resolution, 2.25x more than the model was ever trained on. Re-running the full
  eval suite with `--token-budget 512` (matching training exactly) changed every
  metric by less than 1 point (see `outputs/run/final/eval_results_budget512.json`
  vs. `eval_results.json`) — so the mismatch existed but was not
  materially affecting the numbers. The budget=512 numbers above are the
  methodologically correct ones and are what's reported in this document.
- **Token-budget sensitivity, tested and found negligible.** As a follow-up, eval
  was also run at `token_budget=576` (matching the naive dense stride-16 grid
  token count at 384px — see the efficiency section below) —
  `outputs/run/final/eval_results_budget576.json`. Every metric moved by ≤0.2
  points versus budget=512: ImageNet Top-1 37.18% vs. 37.20%, Flickr I2T R@1
  42.7% vs. 42.9%, T2I R@1 42.3% vs. 42.2%, CIFAR-100 74.87% vs. 74.96%. The
  model's accuracy is effectively flat across the 512–1152 token range tested.
  Note this does **not** by itself demonstrate "ATB beats a dense grid at equal
  compute" — that would require a dense-grid model trained the same way as a
  baseline, which doesn't exist yet (see §5). It only shows Keural Mid's own
  accuracy is insensitive to its own token count.
- **Not an apples-to-apples comparison yet.** CLIP/SigLIP-B/16 were trained on 12M–400M+ pairs over far more optimizer steps than your 18K-step, ~20M-pair run. The gap you see (roughly 2x lower zero-shot accuracy than models that are *smaller* than yours) is more likely a **training-budget gap** than an architecture-capacity gap — your training loss was still declining at step 18000.
- **DINOv2 has no text tower** — it's self-supervised and only comparable on linear-probe / k-NN representation-quality tasks, not zero-shot classification or retrieval.
- **InternViT-300M** also has no native text tower in isolation; its published number is an ImageNet linear-probe (or attention-pooling-probe) result, not zero-shot.
- CIFAR-100 linear probe (74.7%) is your own model's only "linear probe"-style number; none of the public baselines above report CIFAR-100, so it isn't directly comparable to their ImageNet linear-probe scores — included for completeness, not as a matched baseline.
- Your architecture's actual point of novelty — the Adaptive Token Budget (ATB) tokenizer with saliency-driven variable token count — has no established peer benchmark here. A fairer test of that specific contribution would be against other **adaptive/dynamic-token** vision models (e.g. NaViT, FlexiViT), which none of the models above implement.

## 4. Charts

- `figures/benchmark_comparison_chart.png` — ImageNet zero-shot Top-1 and Flickr30K R@1 (single-value view), peer-class models only.
- `figures/benchmark_recall_curve_chart.png` — full Flickr30K Recall@k sweep (R@1/R@5/R@10), both retrieval directions. Shows that Keural Mid's relative gap narrows at higher k (e.g. T2I R@10: 79.4% vs. CLIP-WIT's 68.0% and CLIP-CC12M's 81.5%), which the R@1-only view doesn't capture.
- `figures/saliency_evolution_across_training.png` / `figures/saliency_stability_quantified.png` (n=6 Flickr30K images) and `outputs/saliency_test/saliency_evolution_9ckpt.png` / `saliency_stability_quantified_v2.png` (n=8 established test set, matches prior weekly reports) — see §6.
- `figures/training_loss_curves.png` — full training history, splicing `outputs/training_run1_partial.log` (steps 0–5000) with `outputs/training.log` (steps 5100–17900), 150 logged points total. Training actually ran in two parts: run 1 started 2026-06-30, was SIGKILLed by a kernel cgroup OOM at step 5900 on 2026-07-03 (only checkpointed through step 5000), and run 2 resumed from `checkpoint-5000` on 2026-07-05 — the chart marks this boundary with a dotted vertical line, and run 1's orphaned steps 5700/5800/5900 (never checkpointed, discarded by the resume) are excluded rather than spliced in as a fake continuation. Uncaptured gaps in run 1's reconstructed log (300–1500, 2400–3100, 3300–3800, 4300–5100) are shown dashed, not interpolated silently.
  Four panels: total/primary loss, per-teacher distillation loss, teacher cosine-similarity alignment, and HCT/saliency-regularizer loss. With the full history visible, two things stand out that weren't visible in the run-2-only version: (1) a real instability episode around step 1500–2500 — distillation loss spikes back up to ~0.19 (from ~0.08) and cosine similarity dips from ~0.69 down to ~0.26 before recovering; (2) the dramatic initial drop (loss 21.6→~10 within the first 100 steps) that's expected from random init but worth seeing. Also corroborates §6 quantitatively: the saliency-variance regularizer term sits flat around 4.7–4.9 for nearly the entire run after settling within the first ~1500 steps (step 0's value of 417.9 is an off-scale init spike, omitted from that one panel and noted in its caption) — the same "reaches equilibrium fast, doesn't trend anywhere" behavior visible spatially in `figures/saliency_evolution_across_training.png`. Distillation loss flattens by ~step 11000 while teacher cosine similarity (final ~0.85) is still gently rising at step 18000 — late-training gains are coming mostly from the primary contrastive loss.

## 5. Suggested next comparison

Before drawing conclusions about the ATB architecture itself, the more informative next experiment is a **compute/data-matched baseline**: train a fixed-grid ViT (no ATB) of similar size on the same ~20M pairs for the same 18K steps, and compare directly. That isolates the architecture's contribution from the training-budget gap visible in the tables above.

## 6. Qualitative: saliency map and token placement

`saliency_token_placement.png` overlays the predicted saliency map and the sampled
region/detail token locations on a few real images (checkpoint step 18000,
token_budget=512). Left panel: saliency heatmap, a **stylized constant-alpha blend**
(same 50% opacity everywhere) — this is not a strict value-proportional readout, since
constant-alpha blending interacts with how light or dark the underlying photo pixel
already is, independent of the actual saliency value. It was chosen over a strict
value-proportional rendering because it reads more clearly at a glance; treat it as
illustrative, not quantitative. Right panel: the actual sampled region (aqua) and
detail (orange) token boxes — this part *is* the literal model output.

**Revised finding — image choice matters a lot here.** An earlier draft of this
section, based on 3 ad-hoc "in the wild" Flickr30K photos (candid, cluttered,
no single obvious subject), concluded the saliency map was purely
regularizer-driven with no semantic behavior. That conclusion doesn't survive
testing on `outputs/saliency_test/imgs` + `imgs2` — an established 8-image set
(clean stock photos, one dominant foreground subject each: a dog in a blanket,
a waterfall, a pug, a walrus, a woman, a coffee cup, produce, a vintage car)
used consistently in prior weekly reports. On that set
(`outputs/saliency_test/saliency_evolution_9ckpt.png`, checkpoints
3000/4000/7000/8000/10000/11000/14000/16000/final), the saliency map produces
**sharp, stable contours that visibly track each image's object silhouette**,
not smooth undifferentiated blobs. `saliency_variance_regularizer`
([losses.py:90-101](training/losses.py#L90-L101)) is still the only *direct*
supervision on the map (anti-collapse + smoothness, neither rewards a specific
subject) — but empirically, the indirect task-driven gradient (through the
sampler, back from the primary/distillation loss) is evidently strong enough to
produce a real, consistent figure-ground signal, at least on images with one
visually distinct dominant subject. Whether it holds up on cluttered,
multi-subject, or low-contrast scenes (like the original Flickr30K sample) is
still an open question — call this "works well on clean single-subject images,
untested/weaker on cluttered ones," not "no semantic behavior."

**What the quantitative measurement still holds, now cross-validated on both
image sets** (`figures/saliency_stability_quantified.png`, n=6 Flickr30K images;
`outputs/saliency_test/saliency_stability_quantified_v2.png`, n=8 established
images) — cosine similarity of the raw saliency map + Jaccard overlap of the
actual sampled region/detail grid cells, each checkpoint vs. the previous one
and vs. final:
- The **continuous saliency map** stabilizes fast on both sets — cos-sim to
  final reaches ~0.83-0.90 by the earliest checkpoints tested (step 1000-3000)
  and creeps slowly to 1.0.
- The **discrete sampled tokens** are considerably less stable on both sets:
  only 34-41% overlap (Jaccard) with the final checkpoint by step ~4000, still
  only 49-52% by step ~13000-14000, then a real jump to 73% at step 16000 on
  both sets independently. Small shifts in the underlying saliency values are
  enough to flip which cells win the top-k/NMS selection, so exact token
  placement keeps changing well past the point where the saliency *field*
  itself looks converged — this part of the original finding reproduces
  cleanly across two independent image sets and is the more reliable claim of
  the two.

Net: the saliency map's broad shape is both fast-converging **and** more
semantically meaningful than initially concluded — the earlier "no object
detection" claim was an artifact of testing on 3 visually cluttered images with
no single obvious subject. The discrete-token-instability finding, by contrast,
holds up on both image sets and is the one to keep: exact token placement was
still shifting meaningfully right up to the end of the run, even where the
saliency field itself looked converged.
