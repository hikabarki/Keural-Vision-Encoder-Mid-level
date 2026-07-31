# Stage 2 training figures

Regenerate any time with:

```bash
python3 scripts/make_stage2_figures.py
```

All values are parsed from the run logs — nothing is hand-entered — so the
figures stay correct as training continues.

| File | Shows |
|---|---|
| `stage2a_val_perplexity.png` | Held-out perplexity, stabilized vs unstable 2a run. The headline chart: 17.3 vs 53.0. |
| `stage2a_grad_norm.png` | Gradient norm before clipping, log scale. Makes the steps 450–650 blow-up visible and shows the stabilized run sitting below the clip threshold throughout. |
| `stage2a_train_loss.png` | Training loss, both 2a runs, 20-step moving average over the raw per-step values. |
| `stage2b_training.png` | Stage 2b loss and held-out perplexity as two stacked panels (never a dual axis), with the convergence point marked. |

Sources: `outputs/projector_2a/training_log.jsonl`,
`outputs/stage2a_<timestamp>.log` (the unstable run),
`outputs/projector_2b/training_log.jsonl`.

## Caveats to keep with the figures

- **Stage 2b perplexity (~2.90) is not comparable to Stage 2a (~17.3)** — different
  data and assistant-only label masking. Never plot them on shared axes.
- The `earlier run's best 23.9` reference line in the 2a chart is a single
  recorded value; that run's log was deleted, so it is drawn as a reference
  line rather than a reconstructed curve.
- **None of these measure factual accuracy.** Stage 2b's perplexity improved
  while caption hallucination increased substantially. Use POPE and the
  `--no-vision` blind baseline for that.

Colors are the validated categorical slots (blue / orange / aqua); every series
carries a direct end-label, which is the required relief for the low-contrast
aqua slot rather than decoration.
