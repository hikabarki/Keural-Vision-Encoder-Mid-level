"""
Generate Stage 2a / 2b training figures into docs/figures/.

Data comes from the run logs themselves, never from hand-entered numbers, so
regenerating after more training just works:

  outputs/projector_2a/training_log.jsonl   stabilized 2a (seed 42, lr 5e-4)
  outputs/stage2a_<ts>.log                  the unstable 2a run (lr 1e-3, no seed)
  outputs/projector_2b/training_log.jsonl   2b instruction tuning

Colors are the validated categorical slots 1-3 (blue/orange/aqua). Aqua sits
below 3:1 on the light surface, so every series carries a direct end-label --
that is the required relief, not decoration.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# -- design tokens (validated palette, light surface) --------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "figure.dpi": 200,
})


def style(ax, title, xlabel, ylabel, subtitle=None):
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="bottom")
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.5)


def headroom(ax, frac=0.14):
    """Reserve right margin so end labels sit inside the figure, not clipped."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * frac)


def plain_log_y(ax, ticks):
    """Log y-axis with plain numbers. Minor ticks default to 6x10^1 style, which
    collides with the plain major labels -- suppress them."""
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


def end_label(ax, x, y, text, color, dy=0):
    """Direct label at the series end -- required relief for low-contrast hues."""
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(6, dy), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")


# -- parsers -------------------------------------------------------------------
def jsonl(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tr = [(r["step"], r["loss"], r.get("grad_norm"))
          for r in rows if r.get("type") != "val" and "loss" in r]
    va = [(r["step"], r["val_en_ppl"]) for r in rows
          if r.get("type") == "val" and "val_en_ppl" in r]
    return tr, va


def textlog(path):
    tr, va = [], []
    for line in open(path, errors="replace"):
        m = re.search(r"\[train\] step=\s*(\d+)/\d+.*loss=([\d.]+)\s+grad_norm=([\d.e+-]+)", line)
        if m:
            tr.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m = re.search(r"\[val\]\s+step=\s*(\d+)\s+en loss=[\d.]+ ppl=([\d.]+)", line)
        if m:
            va.append((int(m.group(1)), float(m.group(2))))
    return tr, va


stab_tr, stab_va = jsonl(ROOT / "outputs/projector_2a/training_log.jsonl")
b_tr, b_va = jsonl(ROOT / "outputs/projector_2b/training_log.jsonl")
unstable = sorted(ROOT.glob("outputs/stage2a_2*.log"))
uns_tr, uns_va = textlog(unstable[-1]) if unstable else ([], [])

# The earlier run's log was deleted; only its best validation value survives.
# Drawn as a reference line, never as a fabricated curve.
EARLIER_BEST = 23.90


def unzip(rows, i):
    return [r[0] for r in rows], [r[i] for r in rows]


# =============================================================================
# 1. Stage 2a validation perplexity -- stabilized vs unstable
# =============================================================================
fig, ax = plt.subplots(figsize=(7.6, 4.4))
if uns_va:
    x, y = unzip(uns_va, 1)
    ax.plot(x, y, color=ORANGE, marker="o", markersize=4)
    end_label(ax, x, y, f"unstable  {y[-1]:.1f}", ORANGE)
x, y = unzip(stab_va, 1)
ax.plot(x, y, color=BLUE, marker="o", markersize=4)
end_label(ax, x, y, f"stabilized  {y[-1]:.1f}", BLUE)

ax.axhline(EARLIER_BEST, color=INK_2, lw=1.1, ls=(0, (4, 3)), alpha=0.75)
headroom(ax)
ax.text(ax.get_xlim()[1], EARLIER_BEST, f"earlier run's best {EARLIER_BEST} ", color=INK_2,
        fontsize=8.5, va="bottom", ha="right")
plain_log_y(ax, [20, 30, 50, 80, 120])
style(ax, "Stage 2a — held-out validation perplexity",
      "training step", "perplexity (log scale, lower is better)",
      "Seed 42 + peak LR 5e-4 + warmup 500 vs. unseeded LR 1e-3")
ax.legend(["unstable run (LR 1e-3, unseeded)", "stabilized run (LR 5e-4, seed 42)"],
          frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "stage2a_val_perplexity.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 2. Stage 2a gradient norms -- the instability, made visible
# =============================================================================
fig, ax = plt.subplots(figsize=(7.6, 4.4))
if uns_tr:
    x, y = unzip(uns_tr, 2)
    ax.plot(x, y, color=ORANGE, lw=1.3, alpha=0.9)
x, y = unzip(stab_tr, 2)
ax.plot(x, y, color=BLUE, lw=1.3, alpha=0.9)
ax.axhline(1.0, color=INK_2, lw=1.1, ls=(0, (4, 3)), alpha=0.8)
ax.text(ax.get_xlim()[1] * 0.99, 1.05, "clip threshold 1.0", color=INK_2,
        fontsize=8.5, ha="right", va="bottom")
ax.axvspan(450, 650, color=ORANGE, alpha=0.08, lw=0)
ax.text(550, ax.get_ylim()[1] * 0.35, "blow-up\nsteps 450–650", color=ORANGE,
        fontsize=8.5, ha="center", va="top", fontweight="bold")
plain_log_y(ax, [0.1, 1, 10, 100])
headroom(ax, 0.04)
style(ax, "Stage 2a — gradient norm before clipping",
      "training step", "grad norm (log scale)",
      "Values above the line were clipped; height shows how violent the raw gradient was")
ax.legend(["unstable run (LR 1e-3, unseeded)", "stabilized run (LR 5e-4, seed 42)"],
          frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "stage2a_grad_norm.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 3. Stage 2b -- loss and perplexity as separate panels (never a dual axis)
# =============================================================================
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
x, y = unzip(b_tr, 1)
a1.plot(x, y, color=BLUE, lw=1.2, alpha=0.55)
w = 20
sm = [sum(y[max(0, i - w):i + 1]) / len(y[max(0, i - w):i + 1]) for i in range(len(y))]
a1.plot(x, sm, color=BLUE, lw=2)
end_label(a1, x, sm, f"{sm[-1]:.2f}", BLUE)
style(a1, "Stage 2b — instruction tuning (LLaVA-Instruct 150K)",
      "", "training loss",
      "Warm-started from Stage 2a checkpoint-5000 · 8,000 steps · effective batch 64")

x, y = unzip(b_va, 1)
a2.plot(x, y, color=AQUA, marker="o", markersize=4)
end_label(a2, x, y, f"{y[-1]:.2f}", AQUA)
flat = next((s for s, v in b_va if abs(v - y[-1]) < 0.01), None)
if flat:
    a2.axvline(flat, color=INK_2, lw=1.1, ls=(0, (4, 3)), alpha=0.75)
    a2.text(flat, max(y), f" converged ~step {flat}", color=INK_2, fontsize=8.5,
            va="top", ha="left")
headroom(a1); headroom(a2)
style(a2, "", "training step", "held-out perplexity")
a2.text(0, 1.03, "Not comparable to Stage 2a — different data and assistant-only "
        "label masking", transform=a2.transAxes, fontsize=9, color=INK_2, va="bottom")
fig.tight_layout()
fig.savefig(OUT / "stage2b_training.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 4. Stage 2a training loss -- stabilized vs unstable
# =============================================================================
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for rows, color, name in ((uns_tr, ORANGE, "unstable"), (stab_tr, BLUE, "stabilized")):
    if not rows:
        continue
    x, y = unzip(rows, 1)
    sm = [sum(y[max(0, i - w):i + 1]) / len(y[max(0, i - w):i + 1]) for i in range(len(y))]
    ax.plot(x, y, color=color, lw=1, alpha=0.28)
    ax.plot(x, sm, color=color, lw=2,
            label=("unstable run (LR 1e-3, unseeded)" if name == "unstable"
                   else "stabilized run (LR 5e-4, seed 42)"))
    end_label(ax, x, sm, f"{name}  {sm[-1]:.2f}", color)
headroom(ax, 0.10)
style(ax, "Stage 2a — training loss",
      "training step", "loss (20-step moving average)",
      "Faint line is per-step loss; bold line is the moving average")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "stage2a_train_loss.png", bbox_inches="tight")
plt.close(fig)

print(f"stabilized 2a : {len(stab_tr)} train / {len(stab_va)} val points")
print(f"unstable 2a   : {len(uns_tr)} train / {len(uns_va)} val points"
      f"  (from {unstable[-1].name if unstable else 'MISSING'})")
print(f"stage 2b      : {len(b_tr)} train / {len(b_va)} val points")
for f in sorted(OUT.glob("stage2*.png")):
    print(f"  wrote {f.relative_to(ROOT)}  ({f.stat().st_size/1024:.0f} KB)")
