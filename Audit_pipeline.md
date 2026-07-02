# Keural Mid-Level Vision Encoder – Pipeline Health Audit Checklist

## Background

The current training appears healthy based on optimization logs:

- Contrastive loss decreases smoothly (~10.0 → ~3.0)
- Distillation loss decreases steadily
- SigLIP cosine similarity reaches ~0.71
- InternViT cosine similarity reaches ~0.76
- Learnable temperature increases normally (2.3 → 3.3)
- CIFAR-100 Linear Probe = 61.78%

However,

- ImageNet Zero-Shot = 0.04%
- Flickr30K Retrieval ≈ Random

These evaluation results are significantly lower than expected.

The goal of this audit is to determine whether there is a subtle implementation bug in the multimodal training pipeline.

---

# Highest Priority

## 1. Verify KeuralMidLoss implementation

Please inspect the implementation of:

```
training/losses.py
KeuralMidLoss
```

Specifically verify:

### Image embeddings

- Are the embeddings used for contrastive learning exactly the same embeddings returned during evaluation (`encoder_out.pooled`)?
- Are they taken after the final projection layer?

---

### Text embeddings

Verify that the frozen CLIP text encoder output is projected correctly.

Questions:

- Is the projection trainable?
- Is the projected embedding actually used in the loss?
- Is the projection output normalized?

---

### L2 normalization

Before similarity computation verify:

```python
image = F.normalize(image, dim=-1)
text = F.normalize(text, dim=-1)
```

Both image and text embeddings should be normalized.

---

### SigLIP loss

Please verify the sigmoid contrastive implementation.

Things to check:

- positive labels
- negative labels
- bias term
- learnable temperature
- broadcasting
- masking

Especially ensure positive image-caption pairs are assigned correctly.

---

### Temperature

Verify that evaluation uses the same learned temperature.

Training:

```
logit_scale
```

Evaluation should also use it.

---

### Bias

SigLIP includes a learnable bias.

Confirm:

- bias participates in training
- bias is correctly applied

---

# Distributed Training

## 2. Verify all_gather()

The training runs on multiple GPUs.

Question:

Does KeuralMidLoss perform

```python
torch.distributed.all_gather(...)
```

(or equivalent)

for BOTH

- image embeddings
- text embeddings

before computing similarities?

Without gathering,

GPU 0 only compares against GPU 0 captions.

GPU 1 only compares against GPU 1 captions.

This greatly reduces the number of negatives.

---

# Encoder Output

## 3. Verify pooled representation

Training uses

```
encoder_out
```

Evaluation uses

```
encoder_out.pooled
```

Please verify these are identical to the embeddings used inside the contrastive loss.

If contrastive training uses another tensor but evaluation uses `pooled`, zero-shot accuracy will collapse.

---

# Projection Head

## 4. Verify projection location

Architecture contains

```
pool_token_projection
Linear(768 → 768)
```

Questions:

- Is this projection applied before contrastive loss?
- Is evaluation using the projected feature?
- Is teacher distillation using the same feature?

All three should operate on the same embedding space.

---

# FrozenTextEncoder

Please inspect

```
FrozenTextEncoder
```

Verify:

- OpenAI CLIP text encoder is frozen
- Only projection layer is trainable
- Projection participates in optimizer
- Projection output is returned
- Projection output is normalized

---

# Distillation

Verify:

Student embedding used for

- SigLIP distillation
- InternViT distillation

is exactly the same embedding used for contrastive learning.

Otherwise multiple embedding spaces may compete.

---

# Loss Weighting

Current total loss:

```
L =
L_siglip
+ L_hct
+ L_saliency
+ L_distill
```

Please verify:

- relative magnitudes of each loss
- one loss is not dominating optimization

Please log:

```
loss_siglip
loss_hct
loss_saliency
loss_distill
```

over training.

---

# Evaluation Consistency

Training:

```
OpenAI CLIP text encoder
```

Evaluation:

```
OpenAI CLIP text encoder
```

This appears correct.

However verify:

- tokenizer identical
- preprocessing identical
- image normalization identical
- projection identical
- pooled embedding identical

---

# Tokenizer

Architecture:

```
warmup_steps_fixed_grid = 10000
```

Training ends at

```
18000
```

Questions:

- Is adaptive tokenization actually active after 10k?
- Is token budget changing?
- Is Gumbel temperature annealing working?

Please log:

- token count
- temperature
- saliency variance

---

# Saliency

Verify saliency map is not collapsing.

Log:

```
mean
variance
min
max
```

throughout training.

---

# Gradient Flow

Please verify gradients reach:

- CNN stem
- Saliency predictor
- Token constructor MLP
- Transformer
- Pool projection
- Text projection
- Learnable temperature
- Learnable bias

Look for any module with zero gradients.

---

# Checkpoint Verification

Verify checkpoint contains:

- student weights
- pool projection
- text projection
- temperature
- bias

Evaluation should load every trainable parameter.

---

# Logging

Please add diagnostics for every checkpoint:

Image embedding norm

```
||image||
```

Text embedding norm

```
||text||
```

Cosine similarity

Positive pairs

Negative pairs

Temperature

Bias

Average logits

Positive logits

Negative logits

---

# Main Concern

The architecture, optimizer, scheduler, training loop, and learning curves all appear healthy.

The remaining likely sources of the poor zero-shot performance are:

1. Bug inside `KeuralMidLoss`
2. Incorrect distributed contrastive implementation (`all_gather`)
3. Evaluation using a different embedding than the one optimized during training
4. Projection head mismatch
5. Incorrect normalization
6. Incorrect SigLIP implementation
7. Loss balancing causing insufficient image-text alignment

These components should be carefully audited before concluding that the model itself is underperforming.