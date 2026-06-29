"""
Keural Mid-Level Vision Encoder — Training Script.

Adds knowledge distillation on top of the POC training loop:
  - Loads two frozen teacher models (SigLIP-SO400M, InternViT-300M)
  - Passes their pooled CLS vectors to KeuralMidLoss as distillation targets
  - Everything else (AMP, grad accum, checkpointing) is the same as POC

Usage:
    cd /home/mkd-s002/vision-encoder/mid
    python3.10 scripts/train_mid.py --config configs/keural_mid_single_gpu.yaml --output-dir outputs/run

Smoke test (no data, no teachers):
    python3.10 scripts/train_mid.py --config configs/keural_mid_smoke.yaml --synthetic --max-steps 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

# Ensure the mid/ folder is on the Python path when called from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel
from data.datasets import (
    BucketedSampler,
    CC3MDataset,
    CombinedImageTextDataset,
    FrozenTextEncoder,
    SyntheticPairDataset,
    build_collate_fn,
)
from keural_mid_config import KeuralMidConfig
from training.losses import KeuralMidLoss
from training.distillation import DistillationTeachers


# --------- Helpers ---------

def cosine_warmup_lr(step, peak_lr, warmup_steps, max_steps, min_lr_ratio=0.05):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * min_lr_ratio + peak_lr * (1 - min_lr_ratio) * cosine


def _rotate_checkpoints(output_dir, best_step=None):
    # Auto-deletion disabled — user manages checkpoints manually
    pass


def _save_training_state(ckpt_dir, step, optimizer, scaler):
    state = {"step": step, "optimizer": optimizer.state_dict()}
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, ckpt_dir / "training_state.pt")


def _resume_from_checkpoint(output_dir, model, optimizer, scaler):
    ckpts = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if not ckpts:
        return 0
    latest = ckpts[-1]
    resume_step = int(latest.name.split("-")[-1])
    print(f"[train] Resuming from {latest.name} (step {resume_step})")
    try:
        loaded = KeuralVisionModel.from_pretrained(latest)
        model.load_state_dict(loaded.state_dict())
        print(f"[train] Model weights loaded.")
    except Exception as e:
        print(f"[train] WARNING: Could not load weights: {e}")
        return 0
    state_path = latest / "training_state.pt"
    if state_path.exists():
        try:
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            if scaler is not None and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"[train] WARNING: Optimizer state failed: {e}. Starting optimizer fresh.")
    return resume_step + 1


def build_dataloader(config, synthetic=False, synthetic_size=1000, max_samples=None, rank=0, world_size=1):
    buckets = [tuple(b) for b in config.data.image_buckets]
    if synthetic:
        ds = SyntheticPairDataset(num_samples=synthetic_size, image_buckets=buckets, seed=config.training.seed)
        num_workers = 0
    else:
        primary = CC3MDataset(
            captions_path=config.data.captions_path,
            images_root=config.data.images_root,
            buckets=buckets, augment=True, max_samples=max_samples,
        )
        if config.data.extra_sources:
            extra_datasets = [
                CC3MDataset(
                    captions_path=src["captions_path"], images_root=src["images_root"],
                    buckets=buckets, augment=True, max_samples=max_samples,
                )
                for src in config.data.extra_sources
            ]
            ds = CombinedImageTextDataset([primary] + extra_datasets)
        else:
            ds = primary
        num_workers = config.data.num_workers

    sampler = BucketedSampler(
        ds.bucket_assignments, batch_size=config.training.batch_size_per_device,
        shuffle=True, seed=config.training.seed, drop_last=True,
        shard_keys=getattr(ds, "shard_keys", None),
        rank=rank, world_size=world_size,
    )
    collate = build_collate_fn(config.data.pixel_mean, config.data.pixel_std)
    return DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


# --------- Training loop ---------

def setup_ddp():
    """Initialize distributed training if launched with torchrun."""
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(config, output_dir, synthetic=False, synthetic_size=1000, max_steps_override=None, max_samples=None):
    output_dir = Path(output_dir)

    rank, world_size, device = setup_ddp()
    is_main = rank == 0

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    torch.manual_seed(config.training.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed + rank)

    if is_main:
        print(f"[train] World size: {world_size} | Device: {device}")

    # Build student (the model we're training)
    hf_cfg = KeuralVisionConfig(keural_config=config.to_dict())
    model = KeuralVisionModel(hf_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[train] Student params: {n_params:,} ({n_params/1e6:.1f}M)")

    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    raw_model = model.module if dist.is_initialized() else model

    # Build loss (owns SigLIP projection + learnable temperature/bias)
    loss_fn = KeuralMidLoss(
        config.loss,
        siglip_embed_dim=config.distillation.siglip_embed_dim,
        internvit_embed_dim=config.distillation.internvit_embed_dim,
        embed_dim=config.spatial.embed_dim,
    ).to(device)

    # Text encoder (frozen body, trainable projection to 768-dim)
    text_encoder = FrozenTextEncoder(
        model_name=config.data.text_encoder_name,
        output_dim=config.spatial.embed_dim,
        max_length=config.data.max_caption_length,
    ).to(device)

    # Frozen teacher models for KD (skipped if distillation.enabled=false)
    teachers = DistillationTeachers(config.distillation, device)

    # Optimizer covers student + loss (temp/bias + SigLIP projection) + text projection
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=config.training.lr,
        betas=tuple(config.training.betas), weight_decay=config.training.weight_decay,
    )
    optimizer.add_param_group({"params": [p for p in text_encoder.parameters() if p.requires_grad]})

    # Mixed precision
    use_amp = config.training.precision in ("bfloat16", "float16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.training.precision == "bfloat16" else torch.float16
    needs_scaler = config.training.precision == "float16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if needs_scaler else None

    loader = build_dataloader(
        config, synthetic=synthetic, synthetic_size=synthetic_size,
        max_samples=max_samples, rank=rank, world_size=world_size,
    )

    max_steps = max_steps_override if max_steps_override is not None else config.training.max_steps

    if is_main:
        with (output_dir / "training_config.json").open("w") as f:
            json.dump(asdict(config), f, indent=2)

    step = _resume_from_checkpoint(output_dir, raw_model, optimizer, scaler)

    best_loss = float("inf")
    best_step = None
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(loader)
    log_lines = []
    t0 = time.time()
    last_log_t = t0

    model.train()
    loss_fn.train()
    text_encoder.train()
    # teachers stay in eval() — DistillationTeachers.__init__ sets them eval

    eff_batch = config.training.batch_size_per_device * config.training.gradient_accumulation * world_size
    if is_main:
        print(f"[train] Starting. max_steps={max_steps}, "
              f"batch_per_gpu={config.training.batch_size_per_device}, "
              f"accum={config.training.gradient_accumulation}, "
              f"gpus={world_size}, "
              f"effective_batch={eff_batch}")

    try:
        while step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            captions = batch["captions"]

            text_embeds = text_encoder(captions)

            # Get frozen teacher pooled vectors (no_grad inside DistillationTeachers.forward)
            t_siglip, t_internvit = teachers(pixel_values)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    encoder_out, sal_map = raw_model.encoder(
                        pixel_values=pixel_values,
                        token_budget=config.tokenizer.default_token_budget,
                        step=step, return_saliency_map=True,
                    )
                    loss_out = loss_fn(
                        encoder_out, text_embeds,
                        saliency_map=sal_map,
                        teacher_siglip_pooled=t_siglip,
                        teacher_internvit_pooled=t_internvit,
                    )
                    loss = loss_out.total / config.training.gradient_accumulation
            else:
                encoder_out, sal_map = raw_model.encoder(
                    pixel_values=pixel_values, token_budget=None,
                    step=step, return_saliency_map=True,
                )
                loss_out = loss_fn(
                    encoder_out, text_embeds,
                    saliency_map=sal_map,
                    teacher_siglip_pooled=t_siglip,
                    teacher_internvit_pooled=t_internvit,
                )
                loss = loss_out.total / config.training.gradient_accumulation

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            if accum_count >= config.training.gradient_accumulation:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

                current_lr = cosine_warmup_lr(
                    step, config.training.lr, config.training.warmup_steps, max_steps
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                if is_main and step % config.training.log_every == 0:
                    elapsed = time.time() - last_log_t
                    last_log_t = time.time()
                    elapsed_total = time.time() - t0
                    secs_per_step = elapsed_total / max(step, 1)
                    eta_secs = int(secs_per_step * (max_steps - step))
                    eta_h, eta_m = divmod(eta_secs // 60, 60)
                    pct = step / max_steps * 100

                    imgs_per_sec = config.training.batch_size_per_device * config.training.gradient_accumulation / max(elapsed, 1e-6)
                    gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

                    log = {
                        "step": step, "lr": current_lr,
                        "grad_norm": float(grad_norm),
                        "elapsed_s": elapsed, "eta_secs": eta_secs,
                        "pct_done": round(pct, 2),
                        "imgs_per_sec": round(imgs_per_sec, 1),
                        "gpu_mem_gb": round(gpu_mem_gb, 1),
                        **loss_out.components,
                    }
                    log_lines.append(log)
                    sal_val = log.get('loss/saliency', 0.0)
                    cos_sig = log.get('cos_sim_siglip', 0.0)
                    cos_int = log.get('cos_sim_internvit', 0.0)
                    d_sig = log.get('distill_siglip', 0.0)
                    d_int = log.get('distill_internvit', 0.0)
                    budget = log.get('token_budget_used', 0.0)
                    logit_s = log.get('loss/logit_scale', 0.0)
                    print(
                        f"[{now_str}] step={step:6d}/{max_steps} ({pct:.1f}%) "
                        f"ETA={eta_h}h{eta_m:02d}m "
                        f"lr={current_lr:.2e} "
                        f"loss={log['loss/total']:.4f} "
                        f"primary={log['loss/primary']:.4f} "
                        f"distill={log['loss/distill']:.4f} "
                        f"(sig={d_sig:.4f} int={d_int:.4f}) "
                        f"hct={log['loss/hct']:.4f} "
                        f"sal={sal_val:.4f} "
                        f"cos_sim=[S:{cos_sig:.3f} I:{cos_int:.3f}] "
                        f"budget={budget:.0f} "
                        f"gnorm={float(grad_norm):.2f} "
                        f"temp={logit_s:.2f} "
                        f"img/s={imgs_per_sec:.0f} "
                        f"gpu={gpu_mem_gb:.1f}GB "
                        f"({elapsed:.1f}s)"
                    )

                if is_main and step > 0 and step % config.training.save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    raw_model.save_pretrained(ckpt_dir)
                    _save_training_state(ckpt_dir, step, optimizer, scaler)
                    current_loss = loss_out.total.item()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_step = step
                        print(f"[train] New best at step {step} (loss={best_loss:.4f})")
                    print(f"[train] Saved checkpoint to {ckpt_dir}")
                    _rotate_checkpoints(output_dir, best_step=best_step)
                if dist.is_initialized():
                    dist.barrier()

                step += 1

        if is_main:
            final_dir = output_dir / "final"
            raw_model.save_pretrained(final_dir)
            print(f"[train] Final model saved to {final_dir}")

    except KeyboardInterrupt:
        if is_main:
            print("\n[train] Stopped by user.")
    except Exception as exc:
        print(f"\n[train] FATAL (rank {rank}): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        if is_main:
            with (output_dir / "training_log.jsonl").open("w") as f:
                for entry in log_lines:
                    f.write(json.dumps(entry) + "\n")
        cleanup_ddp()

    return {"steps_completed": step, "final_loss": log_lines[-1]["loss/total"] if log_lines else None}


# --------- CLI ---------

def main():
    parser = argparse.ArgumentParser(description="Keural Mid-Level Encoder training")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output-dir", default="./outputs/run")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-size", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--precision", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--log-every", type=int, default=None)
    args = parser.parse_args()

    config = KeuralMidConfig.load_yaml(args.config)
    if args.batch_size is not None:
        config.training.batch_size_per_device = args.batch_size
    if args.grad_accum is not None:
        config.training.gradient_accumulation = args.grad_accum
    if args.precision is not None:
        config.training.precision = args.precision
    if args.log_every is not None:
        config.training.log_every = args.log_every

    summary = train(
        config=config, output_dir=args.output_dir,
        synthetic=args.synthetic, synthetic_size=args.synthetic_size,
        max_steps_override=args.max_steps, max_samples=args.max_samples,
    )
    print("[train] Summary:", summary)


if __name__ == "__main__":
    main()
