# OCR / Document Understanding — Scoping

Assessment of what it would take to make Keural Mid work on documents and
text-heavy images. All measurements here were taken against the trained
checkpoint (`outputs/run/final`), not estimated.

**Summary:** the resolution barrier is smaller than expected — the encoder
already accepts high-resolution input unchanged. The real constraints are token
*coverage* under ATB's sampling, and the fact that the encoder has never seen
text.

---

## 1. What already works (verified)

The encoder accepts high-resolution input **with no code changes**:

| Input | Auto budget | Output | Result |
|---|---|---|---|
| 384px | 1,152 | (1, 1152, 768) | OK |
| 768px | 2,048 (capped) | (1, 2048, 768) | OK |
| 1024px | 2,048 (capped) | (1, 2048, 768) | OK |

This is a genuine architectural advantage. The continuous 2D-RoPE encodes
position from real normalized `(cx, cy)` coordinates rather than a fixed grid
index, and the ATB tokenizer allocates a runtime budget rather than a fixed
patch count — so neither component is tied to the training resolution. A
standard fixed-grid ViT would need interpolated position embeddings or an
AnyRes/tiling wrapper to do this.

**Consequence: image tiling is not required.** That removes what looked like
the largest piece of work.

---

## 2. The actual constraints

### 2.1 Token coverage vs. grid size — the real limit

The conv stem has stride 16, so each stem cell always covers a 16×16 px region
of the input. Raising resolution therefore increases the *number* of cells, not
the detail per cell:

| Input | Stem grid | Cells | Max budget | Coverage |
|---|---|---|---|---|
| 384px | 24×24 | 576 | 2,048 | 100% (budget exceeds cells) |
| 768px | 48×48 | 2,304 | 2,048 | **89%** |
| 1024px | 64×64 | 4,096 | 2,048 | **50%** |
| 1536px | 96×96 | 9,216 | 2,048 | **22%** |

ATB *samples* the grid — that is the entire design intent, and it is correct
for natural images where most of the frame is uninformative background. For
documents it is a problem: text is distributed across the whole page, so any
sparse sample risks skipping text regions entirely.

Two ways out, with a real trade-off:

**(a) Raise `max_token_budget`.** A config change, but attention cost is
quadratic in sequence length — 4,096 tokens is ~4× the attention FLOPs of
2,048, on top of already being 16× the 512-token training setting. It also
consumes LLM context (see §2.3).

**(b) Teach the saliency predictor to prioritize text.** Keeps the budget
small and is the architecturally interesting option — it is what ATB is *for*.
Requires training on document images (§2.2), and the saliency map currently has
no supervision that would encourage it (see §2.2).

### 2.2 The encoder has never seen text

Stage 1 trained on ~20M natural image–caption pairs. Essentially no documents,
receipts, screenshots, or dense text. Two knock-on effects:

- The convolutional features were never optimized to encode glyph shapes.
- The saliency predictor has never been rewarded for finding text. Its only
  direct supervision is an anti-collapse term plus a total-variation smoothness
  penalty (`training/losses.py`) — neither of which favors text regions. Any
  text-seeking behavior would have to emerge indirectly through the task
  gradient.

This is the deepest of the gaps and cannot be fixed at the projector/LLM layer.

### 2.3 Sequence-length cost at the LLM

| Config | Vision tokens | Share of Qwen2.5-7B's 32k context |
|---|---|---|
| Training-matched (384px, 512) | 512 | 1.6% |
| 768px, budget 2,048 | 2,048 | 6.2% |
| Hypothetical 4,096 budget | 4,096 | 12.5% |

Workable, but it multiplies VRAM and step time in Stage 2, and leaves less room
for multi-turn document Q&A.

### 2.4 High-resolution behavior is unvalidated

The encoder *runs* at 768/1024px but was trained only at 384/448px with a fixed
512-token budget. Running at 2,048 tokens is far outside that distribution.

A prior, much milder version of this was measured: evaluating at 1,152 tokens
instead of the trained 512 changed every benchmark by <1 point
([BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) §3). That is reassuring but
it is a 2.25× stretch — 768px/2,048 tokens is a 4× stretch, and no measurement
exists for it.

---

## 3. Options, in increasing order of cost

### Option A — Projector-only OCR attempt (cheap, low expected quality)
Train Stage 2b on document data (DocVQA, TextVQA) at 768px with budget 2,048,
changing nothing in the encoder.

- Cost: one training run.
- Expected outcome: weak. The encoder cannot resolve or represent text well, so
  the LLM receives visual tokens that do not encode the glyphs.
- Worth doing only as a **baseline measurement** to quantify the gap.

### Option B — Continued encoder pretraining on text-heavy data (recommended)
Resume Stage 1 from `checkpoint-18000` with document images mixed into the
corpus, at mixed resolutions including 768px.

- Data: **SynthDoG** (synthetic documents, has a **Korean** generator — serves
  the bilingual goal simultaneously), plus real documents.
- Fixes both the feature gap and the high-resolution distribution gap.
- Cost: a multi-day encoder training run, plus data acquisition.

### Option C — Add explicit saliency supervision for text
Extend the loss with a term that rewards placing detail tokens on text regions
(supervisable from synthetic documents, where glyph bounding boxes are known
for free).

- Directly addresses §2.1(b) and makes ATB genuinely useful for documents at a
  small token budget — the strongest version of the architecture's own thesis.
- Cost: loss-function work plus an encoder run. Highest risk, highest payoff.

---

## 4. Recommended sequence

1. **Finish the English → bilingual VLM first.** It is ready to launch and
   validates the whole Stage 2 stack on data where the encoder is competent.
2. **Run Option A as a measurement**, not as a product — get a real DocVQA/
   TextVQA number so the gap is quantified rather than assumed.
3. **Then Option B**, with SynthDoG-Korean pulling double duty for the
   bilingual and document goals.
4. **Option C** only if the architecture's adaptive-token thesis is something
   you want to prove out on documents specifically.

## 5. Datasets to acquire when starting this

| Dataset | Purpose | Note |
|---|---|---|
| SynthDoG (`naver-clova-ix/synthdog-*`) | Synthetic documents, incl. Korean | Free glyph boxes for Option C |
| DocVQA | Document QA benchmark | Standard eval |
| TextVQA | Scene-text QA | Standard eval |
| OCR-VQA | Book-cover OCR QA | Large, cheap |

None are downloaded yet — deliberately. Fetching them before the resolution and
coverage decisions above are made would mean training the model to read text it
cannot physically resolve.
