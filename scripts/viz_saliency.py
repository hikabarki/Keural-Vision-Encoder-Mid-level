"""
Visualize the SaliencyPredictor output for a trained checkpoint.

Runs stem -> SaliencyPredictor on CPU (does not touch training GPUs) and saves,
per image, a panel: original | saliency heatmap | overlay.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/viz_saliency.py \
        --checkpoint outputs/run/checkpoint-3000 \
        --images outputs/saliency_test/imgs \
        --out outputs/saliency_test --size 384
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from alignment.hf_wrapper import KeuralVisionModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="dir of images")
    ap.add_argument("--out", default="outputs/saliency_test")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[viz] loading {args.checkpoint} on CPU ...")
    model = KeuralVisionModel.from_pretrained(args.checkpoint).to(device).eval()
    stem = model.encoder.tokenizer.stem
    saliency = model.encoder.tokenizer.saliency

    tfm = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
    ])
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    img_paths = sorted([p for p in Path(args.images).iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] {len(img_paths)} images, size {args.size}")

    panels = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x01 = tfm(img)                       # (3,S,S) in [0,1] for display
        x = norm(x01).unsqueeze(0)           # (1,3,S,S) normalized for model
        with torch.no_grad():
            feat = stem(x)                   # (1,768,h,w)
            sal = saliency(feat)             # (1,1,h,w) in (0,1)
            sal_up = F.interpolate(sal, size=(args.size, args.size),
                                   mode="bilinear", align_corners=False)[0, 0]
        sal_np = sal_up.numpy()
        disp = x01.permute(1, 2, 0).numpy()

        # normalize saliency to [0,1] for display contrast
        smin, smax = float(sal_np.min()), float(sal_np.max())
        sal_disp = (sal_np - smin) / (smax - smin + 1e-8)

        stats = f"grid={tuple(sal.shape[-2:])}  min={smin:.3f} max={smax:.3f} mean={sal_np.mean():.3f} var={sal_np.var():.4f}"
        print(f"[viz] {p.name}: {stats}")

        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
        ax[0].imshow(disp); ax[0].set_title(p.name); ax[0].axis("off")
        im = ax[1].imshow(sal_disp, cmap="jet"); ax[1].set_title("saliency map"); ax[1].axis("off")
        ax[2].imshow(disp); ax[2].imshow(sal_disp, cmap="jet", alpha=0.5)
        ax[2].set_title("overlay"); ax[2].axis("off")
        fig.suptitle(stats, fontsize=9, y=0.02)
        fig.tight_layout()
        outp = out_dir / f"saliency_{p.stem}.png"
        fig.savefig(outp, dpi=90, bbox_inches="tight"); plt.close(fig)
        panels.append(outp)
        print(f"[viz] wrote {outp}")

    # also a combined contact sheet
    n = len(panels)
    if n:
        fig, axs = plt.subplots(n, 1, figsize=(13, 4.6 * n))
        if n == 1: axs = [axs]
        for a, pth in zip(axs, panels):
            a.imshow(plt.imread(pth)); a.axis("off")
        fig.tight_layout()
        sheet = out_dir / "saliency_contact_sheet.png"
        fig.savefig(sheet, dpi=80, bbox_inches="tight"); plt.close(fig)
        print(f"[viz] contact sheet -> {sheet}")


if __name__ == "__main__":
    main()
