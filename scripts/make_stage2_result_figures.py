"""
Stage 2 result figures for the technical documentation report.

Two charts:
  stage2_vision_contribution.png  — sighted vs blind across benchmarks. This is
      the headline evidence: the GAP is what the vision encoder contributes, and
      it is the one number that cannot be faked by a strong language model.
  stage2_val_perplexity.png       — held-out perplexity for both 2b runs.

All numbers are read from the result JSON, never hand-entered, so the figures
cannot drift from the measurements.

Colors are the validated categorical slots 1-2 (blue / orange), which pass the
CVD and normal-vision separation checks. Every bar is directly value-labeled, so
identity never rests on color alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def load(*rel):
    for r in rel:
        p = ROOT / r
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# 1. Vision contribution — sighted vs blind
# ---------------------------------------------------------------------------
pope_s = load("outputs/eval_2b/pope_trained_outputs_projector_2b_bilingual_checkpoint-12000.json")
pope_b = load("outputs/queue/pope_blind.json")
vqa_s = load("outputs/eval_2b/vqav2_sighted.json")
vqa_b = load("outputs/eval_2b/vqav2_blind.json")

rows = []
if pope_s and pope_b:
    rows.append(("POPE\naccuracy", pope_s["accuracy"] * 100, pope_b["accuracy"] * 100))
if pope_s and pope_b:
    rows.append(("POPE\nF1", pope_s["f1"] * 100, pope_b["f1"] * 100))
if vqa_s and vqa_b:
    rows.append(("VQAv2\nstrict", vqa_s["accuracy"] * 100, vqa_b["accuracy"] * 100))
    if "lenient_accuracy" in vqa_s:
        rows.append(("VQAv2\nlenient", vqa_s["lenient_accuracy"] * 100,
                     vqa_b["lenient_accuracy"] * 100))

if rows:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = range(len(rows))
    w = 0.36
    s_vals = [r[1] for r in rows]
    b_vals = [r[2] for r in rows]
    bs = ax.bar([i - w / 2 for i in x], s_vals, w, color=BLUE,
                label="With image (sighted)", zorder=3)
    bb = ax.bar([i + w / 2 for i in x], b_vals, w, color=ORANGE,
                label="Blind — visual tokens removed", zorder=3)
    for rect, v in list(zip(bs, s_vals)) + list(zip(bb, b_vals)):
        ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    # The gap is the actual result; label it explicitly.
    for i, (_, s, b) in enumerate(rows):
        top = max(s, b)
        ax.annotate(f"+{s - b:.1f}", (i, top + 7), ha="center", fontsize=10,
                    fontweight="bold", color=BLUE)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("score (%)", fontsize=9.5)
    ax.set_ylim(0, max(max(s_vals), max(b_vals)) * 1.30)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0)
    ax.set_title("Vision contribution — Keural Mid VLM (bilingual, checkpoint-12000)",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=20)
    ax.text(0, 1.02, "The bold figure is the sighted-minus-blind gap: what the vision "
                     "encoder contributes beyond the language model's priors",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_vision_contribution.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote stage2_vision_contribution.png")
    for n, s, b in rows:
        print(f"   {n.replace(chr(10),' '):16s} sighted {s:6.2f}  blind {b:6.2f}  gap {s-b:+6.2f}")


# ---------------------------------------------------------------------------
# 2. Stage 2b held-out perplexity — English run vs bilingual run
# ---------------------------------------------------------------------------
def val_series(path, key):
    p = ROOT / path
    if not p.exists():
        return [], []
    xs, ys = [], []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "val" and key in r:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


en_x, en_y = val_series("outputs/projector_2b/training_log.jsonl", "val_en_ppl")
ko_x, ko_y = val_series("outputs/projector_2b_bilingual/training_log.jsonl", "val_ko_ppl")

if en_y or ko_y:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    if en_y:
        ax.plot(en_x, en_y, color=BLUE, lw=2, marker="o", markersize=3)
        ax.annotate(f"English 2b  {en_y[-1]:.2f}", (en_x[-1], en_y[-1]),
                    xytext=(6, 0), textcoords="offset points", color=BLUE,
                    fontsize=9, fontweight="bold", va="center")
    if ko_y:
        ax.plot(ko_x, ko_y, color=ORANGE, lw=2, marker="o", markersize=3)
        ax.annotate(f"Korean (bilingual)  {ko_y[-1]:.2f}", (ko_x[-1], ko_y[-1]),
                    xytext=(6, 0), textcoords="offset points", color=ORANGE,
                    fontsize=9, fontweight="bold", va="center")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.26)
    ax.set_xlabel("training step", fontsize=9.5)
    ax.set_ylabel("held-out perplexity (lower is better)", fontsize=9.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.5)
    ax.set_title("Stage 2b — held-out perplexity", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left", pad=20)
    ax.text(0, 1.02, "Separate runs on different data and label masking — each curve is "
                     "comparable only with itself",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    ax.legend(["English instruction tuning (8,000 steps)",
               "Bilingual EN+KO, Korean split (12,000 steps)"],
              frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_val_perplexity.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote stage2_val_perplexity.png")
    if en_y:
        print(f"   English 2b : {en_y[0]:.2f} -> {en_y[-1]:.2f} over {en_x[-1]} steps")
    if ko_y:
        print(f"   Korean     : {ko_y[0]:.2f} -> {ko_y[-1]:.2f} over {ko_x[-1]} steps")
