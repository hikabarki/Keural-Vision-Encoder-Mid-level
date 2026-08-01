"""
Model-selection figure for the technical report.

Four checkpoints were trained after Stage 2a. The point of this chart is that
they do NOT form a ranking - each later stage traded one capability for another,
and the calibration panel is what makes that visible. POPE accuracy alone would
hide it, because a model answering "no" to everything still scores ~50%.

Values are read from the evaluation JSON, never typed.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200})

def L(p):
    q = ROOT / p
    return json.loads(q.read_text()) if q.exists() else None

def mme_total(p):
    d = L(p)
    return sum(v["score"] for v in d["categories"].values()) if d else None

MODELS = [
    ("Stage 2b\nbilingual\n(SFT)",      "outputs/queue/pope_n2000.json",      "outputs/queue/mme_full.json"),
    ("Stage 2c v1\nOCR",                "outputs/queue/pope_2c_n2000.json",   "outputs/queue/mme_2c.json"),
    ("Stage 2c v2\nOCR + rebalanced",   "outputs/queue/pope_2cv2_n2000.json", "outputs/queue/mme_2cv2.json"),
    ("Stage 3\nDPO",                    "outputs/queue/pope_dpo_n2000.json",  None),
]
names, acc, yr, mme = [], [], [], []
for n, pp, mm in MODELS:
    p = L(pp)
    if not p:
        continue
    names.append(n); acc.append(p["accuracy"] * 100); yr.append(p["yes_ratio"])
    mme.append(mme_total(mm) if mm else None)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True)
x = range(len(names))
a1.bar(x, acc, 0.5, color=BLUE, zorder=3)
for i, v in enumerate(acc):
    a1.annotate(f"{v:.1f}%", (i, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
for i, v in enumerate(mme):
    if v:
        a1.annotate(f"MME {v:.0f}", (i, 2), ha="center", fontsize=8.5,
                    color=INK_2, va="bottom")
a1.set_ylabel("POPE accuracy (%)", fontsize=9.5)
a1.set_ylim(0, max(acc) * 1.22)
a1.set_axisbelow(True); a1.grid(axis="x", alpha=0)
a1.set_title("Model selection — every stage after SFT traded one capability for another",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=20)
a1.text(0, 1.02, "MME total (all 14 categories) annotated inside each bar",
        transform=a1.transAxes, fontsize=9, color=INK_2, va="bottom")

cols = [AQUA if abs(v - 0.5) < 0.1 else ORANGE for v in yr]
a2.bar(x, yr, 0.5, color=cols, zorder=3)
a2.axhline(0.5, color=INK_2, lw=1.2, ls=(0, (4, 3)))
a2.text(len(names) - 0.5, 0.515, "gold ratio 0.500 — a calibrated model sits here",
        fontsize=8.5, color=INK_2, ha="right", va="bottom")
for i, v in enumerate(yr):
    a2.annotate(f"{v:.3f}", (i, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
a2.set_ylabel("POPE yes-ratio", fontsize=9.5)
a2.set_ylim(0, 1.0)
a2.set_xticks(list(x)); a2.set_xticklabels(names, fontsize=8.5)
a2.set_axisbelow(True); a2.grid(axis="x", alpha=0)
a2.text(0, 1.02, "Distance from 0.500 is the calibration failure. Only Stage 2b is "
        "calibrated; 2c drifts toward 'yes', DPO collapses to 'no'.",
        transform=a2.transAxes, fontsize=9, color=INK_2, va="bottom")
fig.tight_layout()
fig.savefig(OUT / "model_selection.png", bbox_inches="tight")
print("wrote model_selection.png")
for n, a, y, m in zip(names, acc, yr, mme):
    print(f"  {n.replace(chr(10),' '):34s} POPE {a:5.2f}%  yes {y:.3f}  "
          f"MME {m if m else '—'}")
