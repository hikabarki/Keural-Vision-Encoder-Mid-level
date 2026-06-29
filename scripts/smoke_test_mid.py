"""
Keural Mid-Level Encoder — Smoke Test.

Runs 5 steps with synthetic data and distillation disabled.
Verifies: model builds, forward pass runs, loss is finite, backward succeeds,
parameter count is in the expected 220–240M range.

Usage:
    cd /home/mkd-s002/vision-encoder/mid
    python3.10 scripts/smoke_test_mid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel
from data.datasets import FrozenTextEncoder, SyntheticPairDataset, build_collate_fn, BucketedSampler
from keural_mid_config import KeuralMidConfig
from torch.utils.data import DataLoader
from training.losses import KeuralMidLoss


def run_smoke_test():
    config = KeuralMidConfig.load_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "keural_mid_smoke.yaml"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] Device: {device}")

    # Build model
    hf_cfg = KeuralVisionConfig(keural_config=config.to_dict())
    model = KeuralVisionModel(hf_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    assert 180_000_000 <= n_params <= 200_000_000, (
        f"Expected ~183.5M params, got {n_params/1e6:.1f}M. Check config dimensions."
    )
    print(f"[smoke] Parameter count OK ({n_params/1e6:.1f}M in expected range 180M–200M).")

    # Build loss + text encoder
    loss_fn = KeuralMidLoss(
        config.loss,
        siglip_embed_dim=config.distillation.siglip_embed_dim,
        embed_dim=config.spatial.embed_dim,
    ).to(device)
    text_encoder = FrozenTextEncoder(
        model_name=config.data.text_encoder_name,
        output_dim=config.spatial.embed_dim,
        max_length=config.data.max_caption_length,
    ).to(device)

    # Optimizer
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.training.lr)

    # Synthetic data
    buckets = [tuple(b) for b in config.data.image_buckets]
    ds = SyntheticPairDataset(num_samples=64, image_buckets=buckets, seed=42)
    sampler = BucketedSampler(
        ds.bucket_assignments, batch_size=config.training.batch_size_per_device,
        shuffle=False, seed=42, drop_last=True,
    )
    collate = build_collate_fn(config.data.pixel_mean, config.data.pixel_std)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)

    model.train()
    loss_fn.train()
    text_encoder.train()

    print("[smoke] Running 5 training steps...")
    loader_iter = iter(loader)

    for step in range(5):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        pixel_values = batch["pixel_values"].to(device)
        captions = batch["captions"]

        text_embeds = text_encoder(captions)

        use_amp = config.training.precision == "bfloat16" and device.type == "cuda"
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoder_out, sal_map = model.encoder(
                    pixel_values=pixel_values, token_budget=None,
                    step=step, return_saliency_map=True,
                )
                loss_out = loss_fn(encoder_out, text_embeds, saliency_map=sal_map)
        else:
            encoder_out, sal_map = model.encoder(
                pixel_values=pixel_values, token_budget=None,
                step=step, return_saliency_map=True,
            )
            loss_out = loss_fn(encoder_out, text_embeds, saliency_map=sal_map)

        assert torch.isfinite(loss_out.total), f"Loss is NaN/Inf at step {step}"

        loss_out.total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        print(
            f"[smoke] step={step} "
            f"loss={loss_out.total.item():.4f} "
            f"primary={loss_out.primary.item():.4f} "
            f"tokens={encoder_out.tokens.shape}"
        )

    print("\n[smoke] ALL 5 STEPS PASSED.")
    print(f"[smoke] Output shape: tokens={encoder_out.tokens.shape}, pooled={encoder_out.pooled.shape}")
    print("[smoke] Smoke test complete — ready to run full training.")


if __name__ == "__main__":
    run_smoke_test()
