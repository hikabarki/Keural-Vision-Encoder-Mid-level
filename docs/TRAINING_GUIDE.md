# Keural Mid — Step-by-Step Training Guide

End-to-end runbook for training the Keural Mid vision encoder and aligning it
into a VLM. Every command here has been verified against this repository.

**Current status:** Stage 1 complete (18,000 steps). Stage 2 code and data are
staged and verified; awaiting GPU availability.

---

## 0. Prerequisites

### 0.1 Environment

```bash
cd /home/work/Hika/vision-encoder/mid
source venv_mid/bin/activate
```

If the venv does not exist yet:

```bash
bash setup_venv.sh
```

### 0.2 Check the GPUs are free

**Do not launch onto busy GPUs.** This machine is shared, and long jobs from
other projects run here.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Both Stage 2 launch scripts abort automatically if >20 GB is already allocated,
but check first so you know *why* something is running before killing it.

### 0.3 Verify the trained encoder is present

```bash
ls outputs/run/final/
# expected: config.json  model.safetensors  alignment_state.pt
```

`alignment_state.pt` is required for evaluation — it holds the trained text
projection. Without it, evaluation silently falls back to raw CLIP text
embeddings and every retrieval number becomes meaningless.

---

## 1. Stage 1 — Contrastive Pre-training (complete)

Recorded here for reproduction. **You do not need to re-run this.**

```bash
torchrun --nproc_per_node=2 scripts/train_mid.py \
  --config configs/keural_mid_2xh200.yaml \
  --output-dir outputs/run \
  --max-samples 20000000
```

| Setting | Value |
|---|---|
| Data | ~20M image–text pairs (DataComp-Medium) |
| Steps | 18,000 |
| Effective batch | 3,072 (768/GPU × 2 grad-accum × 2 GPUs) |
| Optimizer | AdamW, lr 5e-4 cosine, 2,000 warmup |
| Precision | bfloat16 |
| Final loss | 2.87 (from 21.56) |

> **Note:** the training data directory
> (`/home/work/keural-model/vision-encoder-dataset/`) has since been deleted.
> Re-running Stage 1 requires re-acquiring DataComp/COYO first.

### 1.1 Evaluate a Stage 1 checkpoint

```bash
python3 scripts/evaluate.py \
  --checkpoint outputs/run/final \
  --benchmarks imagenet flickr30k cifar100 \
  --token-budget 512 \
  --device cuda
```

**Always pass `--token-budget 512`.** Training used a fixed budget of 512
tokens; omitting the flag triggers resolution-based auto-scaling (1,152 tokens
at 384px), which the model was never trained on. See
[BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) §3.

Final results: 37.2% ImageNet zero-shot Top-1, 42.9%/42.2% Flickr30K R@1,
75.0% CIFAR-100 linear probe.

---

## 2. Stage 2 — VLM Alignment

Attaches the frozen encoder to an LLM through a trainable projector. Two
sub-stages: the projector learns the LLM's embedding geometry first (2a), then
learns to follow instructions (2b).

```
Image → [FROZEN encoder 183.5M] → 512×768 tokens
      → [TRAINABLE LevelAwareProjector ~8.9M] → 512×3584
      → [FROZEN Qwen2.5-7B + trainable LoRA] → text
```

### 2.0 Data locations

All Stage 2 data lives on the KT Cloud NFS share (7.7 TB free), **not** the
local disk:

| Dataset | Path | Records |
|---|---|---|
| LLaVA-Pretrain (2a) | `$DATA/llava-pretrain/pretrain_558k.jsonl` | 558,128 |
| LLaVA-Instruct (2b EN) | `$DATA/llava-instruct/train_150k.jsonl` | 156,712 |
| KoLLaVA (2b KO) | `$DATA/kollava-instruct/train_ko_coco.jsonl` | 361,953 |
| Mixed EN+KO (2b bilingual) | `$DATA/mixed_en_ko_instruct.jsonl` | 518,665 |
| COCO images | `$DATA/coco/train2017/` | 118,287 |
| LLaVA-Pretrain images | `$DATA/llava-pretrain/images/` | 558,128 |

where `DATA=/home/work/keural-model/vlm-datasets`.

Verified: **0 records dropped** for missing images in any set.

### 2.1 Stage 2a — Projector warm-up (captions)

Teaches the projector to map 768-dim visual tokens into Qwen's 3584-dim
embedding space. Run this **first**.

```bash
bash scripts/run_stage2a_caption.sh
tail -f outputs/stage2a.log
```

| Setting | Value | Why |
|---|---|---|
| LR | 1e-3 | High — the projector is training from random init |
| Steps | 5,000 | Warm-up only; convergence is fast |
| Batch / accum | 16 / 2 (eff. 32) | ~36% of one H200 — see §4.1 |
| Precision | bf16 + LoRA | Not 4-bit; see §4.3 |

**What healthy looks like:** loss drops steeply in the first few hundred steps
(the projector finding the embedding space), then flattens. Checkpoints land in
`outputs/projector_2a/`.

### 2.2 Stage 2b — Instruction tuning

Pick **one**:

**English only** — smaller, faster, good for a first validation run:

```bash
bash scripts/run_stage2b_instruct.sh
tail -f outputs/stage2b.log
```

**Bilingual EN + Korean** — 518,665 mixed records, both using the same COCO
images:

```bash
bash scripts/run_stage2b_bilingual.sh
tail -f outputs/stage2b_bilingual.log
```

| Setting | Value | Why |
|---|---|---|
| LR | 2e-5 | **Much lower than 2a** — a high LR here destroys 2a's alignment |
| Steps | 8,000 (EN) / 12,000 (bilingual) | Bilingual has 3.3× the data |
| Max length | 512 | Multi-turn conversations are longer than captions |
| Batch / accum | 8 / 4 (eff. 32) | ~34% of one H200 — see §4.1 |

Labels are masked to **assistant turns only** — the model learns to answer, not
to echo the question. This is handled in `data/vlm_dataset.build_llava_batch`
and is unit-tested against the Qwen chat template.

> **Language mix note:** the bilingual set is ~70% Korean / 30% English, which
> is simply the natural size ratio of the two datasets. If you want balanced
> behavior rather than Korean-leaning, downsample the Korean side when building
> the mixture.

---

## 3. Monitoring

```bash
# live log
tail -f outputs/stage2a.log

# structured metrics (one JSON object per logged step)
tail -f outputs/projector_2a/training_log.jsonl

# GPU utilization
watch -n5 nvidia-smi
```

**Signals to watch:**

| Signal | Healthy | Problem |
|---|---|---|
| `loss` | decreasing, then flattening | rising, or `nan` |
| `grad_norm` | stable, order 1–100 | exploding, or exactly 0 |
| GPU util | 80–100% | <40% → dataloader-bound (see §4.2) |

---

## 4. Troubleshooting

### 4.1 CUDA out of memory

VRAM was profiled before launch (encoder side measured under a hard memory cap;
LLM weights are exact; activations estimated):

| Component | GiB | Source |
|---|---|---|
| Qwen2.5-7B weights (bf16) | 14.2 | exact |
| Encoder + projector @ batch 8 | 0.62 | **measured** |
| Activations | 4.18 MiB / token | estimated |

Projected peak on one 141 GiB H200:

| Stage | Batch | Seq | Total | Utilization |
|---|---|---|---|---|
| 2a | 16 | 544 | ~50 GiB | 36% |
| 2b | 8 | 1024 | ~48 GiB | 34% |

Both configs sit near a third of one card, so there is roughly 2× headroom even
if the activation estimate is off by a factor of two.

If a run OOMs anyway, halve the micro-batch and double accumulation to keep the
effective batch (32) constant:

```bash
--batch-size 8 --grad-accum 4     # 2a fallback
--batch-size 4 --grad-accum 8     # 2b fallback
```

### 4.2 Low GPU utilization

The datasets are on **NFS** (network storage) and consist of hundreds of
thousands of loose JPEGs. NFS random-read latency can starve the GPU.

Diagnose: if `nvidia-smi` shows GPU util well below 50% while the loss is still
progressing, it is I/O, not compute. Mitigations: increase dataloader workers,
or repack the images into WebDataset `.tar` shards for sequential reads.

### 4.3 bitsandbytes / 4-bit errors

`bitsandbytes` in this environment has **no compiled binary for the installed
CUDA version**, so the 4-bit QLoRA path fails at import.

You do not need it. A 7B model in bf16 is ~15 GB on a 141 GB H200 — quantizing
would only cost quality and speed. All launch scripts already pass `--no-4bit`,
and the bitsandbytes import is lazy so the bf16 path never touches it.

### 4.4 Resuming after a crash

```bash
python3 scripts/train_projector.py ... --resume
```

Loads the latest checkpoint's projector weights and optimizer state from
`--output-dir`.

### 4.5 Host RAM OOM (the Stage 1 failure mode)

Stage 1 run 1 was SIGKILLed by the kernel cgroup OOM killer after 3.3 days:
24 dataloader workers each held a ~9.7 GB copy of the sample index, exceeding
the 352 GiB container cap. Fixed by reducing `num_workers` 12 → 6.

If a long run dies with exit code -9 and no Python traceback, check:

```bash
cat /sys/fs/cgroup/memory.events   # oom_kill counter
```

---

## 5. Evaluating the VLM

### 5.1 First — look at the output (do this before any benchmark)

A loss curve does not tell you whether the model emits coherent text. As soon
as Stage 2a or 2b writes a checkpoint:

```bash
python3 scripts/demo_vlm.py \
  --projector-ckpt outputs/projector_2a/checkpoint-5000/projector.pt \
  --lora-ckpt      outputs/projector_2a/checkpoint-5000/lora_adapter \
  --image /path/to/photo.jpg \
  --question "What is in this image?"
```

Interactive mode (`--chat`) keeps asking about the same image. Korean works
too — just ask in Korean after the bilingual run.

**Expectations:** after **2a** the model captions but does not follow
instructions — that is correct, instruction-following comes from 2b. After
**2b** it should answer questions directly. Output that ignores the image
entirely means the projector is not aligned; check that `--hidden-dim` matches
what training used (default 2048).

### 5.2 Benchmarks

```bash
P=outputs/projector_2b_bilingual/checkpoint-12000

python3 scripts/eval_vlm.py --benchmark pope \
  --projector-ckpt $P/projector.pt --lora-ckpt $P/lora_adapter --limit 500
```

| Benchmark | Measures |
|---|---|
| `pope` | Object hallucination (yes/no) — accuracy, F1, **yes-ratio** |
| `vqav2` | Open-ended VQA, standard soft accuracy |
| `mme` | 14 perception/cognition subtasks, per-category + MME score |
| `mmbench` | Multiple choice (A/B/C/D) — robust to answer parsing |
| `heldout` | **Loss/perplexity on our own held-out split, EN and KO separately** |

The first four stream from the HuggingFace hub — no manual preparation. Use
`--limit` for a quick signal, drop it for a full run. Results land in
`outputs/eval_<benchmark>.json`.

#### Run the blind baseline — this is not optional

VQAv2 and MME are substantially answerable from language priors alone. A model
that never sees the image can score ~40% on VQAv2 by exploiting question
statistics. So run each benchmark **twice**:

```bash
python3 scripts/eval_vlm.py --benchmark vqav2 --projector-ckpt $P/projector.pt --limit 1000
python3 scripts/eval_vlm.py --benchmark vqav2 --projector-ckpt $P/projector.pt --limit 1000 --no-vision
```

```
vision_score  -  blind_score  =  what the vision pipeline actually contributes
```

If that gap is near zero, the encoder and projector are not helping no matter
how respectable the headline number looks. Given this encoder is undertrained
(37.2% zero-shot ImageNet), the gap is the number that matters. Blind runs are
saved separately as `outputs/eval_<benchmark>_blind.json`.

#### Korean evaluation

No public Korean VLM benchmark exists, and the bilingual run is ~70% Korean —
so without `heldout` the majority of training would be unmeasured. It scores
teacher-forced loss and perplexity on the 1,000 held-out EN and 1,000 held-out
KO records reserved when the JSONLs were built, reported per language:

```bash
python3 scripts/eval_vlm.py --benchmark heldout --projector-ckpt $P/projector.pt --limit 500
```

Expect Korean perplexity to be higher than English — Korean is the harder
language for a primarily-English-pretrained LLM. Watch the *trend* across
checkpoints rather than the absolute value.

> **Read the POPE yes-ratio, not just accuracy.** A model that answers "yes" to
> every question scores ~50% accuracy with a yes-ratio near 1.0. The script
> reports both, plus how many answers were unparseable.

### 5.3 Not yet ported

`eval_mme.py` and the GUI tools remain in the PoC repo
(`../tiny-reference/VISION-ENCODER_tiny-main/scripts/`) and still target the
384-dim encoder and Mistral decoder.

---

## 6. Roadmap — what is not built yet

### 6.1 Korean — ready
Data staged and verified. Just run `run_stage2b_bilingual.sh`.

### 6.2 OCR / documents — see [OCR_ROADMAP.md](OCR_ROADMAP.md)

Empirically, the encoder **already accepts 768px and 1024px input with no code
changes** — the continuous 2D-RoPE and ATB tokenizer are genuinely
resolution-elastic. The remaining gaps are (a) the encoder was pretrained only
at 384/448px so high-res behavior is unvalidated, (b) it never saw text-heavy
images, and (c) 2048 vision tokens is a long sequence for the LLM. Details and
options in the linked document.

### 6.3 Continued encoder pretraining
The highest-leverage remaining work. Accuracy trails same-size peers (37.2% vs
CLIP-B/16's 68.6%) because of training budget, not architecture — loss was
still declining at step 18,000. Requires re-acquiring DataComp/COYO first.
