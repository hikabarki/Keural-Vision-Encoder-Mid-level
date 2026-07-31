"""
Keural Mid — Stage 2 Projector Alignment Training.

Trains a LevelAwareProjector that bridges the frozen Keural Mid vision encoder
(768-dim) and an instruction-tuned LLM. The LLM is loaded in 4-bit NF4 (QLoRA)
and remains frozen except for lightweight LoRA adapters. Only the projector and
LoRA weights train.

Ported from the PoC (24.7M / 384-dim) train_projector.py, with the PoC's two
dimension footguns removed:
  - encoder_dim is read from the loaded encoder (768 for Mid), not hardcoded 384.
  - llm_dim is read from llm.config.hidden_size, not hardcoded 4096 — so ANY
    HF causal LM works without editing the script (Gemma-2-2B=2304,
    Qwen2-0.5B=896, Llama-3-8B=4096, Mistral-7B=4096, ...).

Architecture during Stage 2:
    Image (384x384)
        v  Vision Encoder (frozen, 183.5M params)
    512 tokens x 768-dim  (KeuralEncoderOutput)
        v  LevelAwareProjector (trains here)
    512 visual tokens x <llm_dim>
        v  <any HF causal LM> (4-bit NF4 + LoRA)
    "Caption: <text>"

Usage:
    cd /home/work/Hika/vision-encoder/mid
    python3 scripts/train_projector.py \\
        --vision-ckpt outputs/run/final \\
        --llm-name google/gemma-2-2b-it \\
        --output-dir outputs/projector \\
        --tar-dir data/cc3m

    # Quick sanity check (5 steps, small batch):
    python3 scripts/train_projector.py ... --max-steps 5 --batch-size 2 --grad-accum 1
"""

from __future__ import annotations

import sys, os, io, tarfile, glob, argparse, json, math, time, random, shutil
from pathlib import Path

# Repo root = parent of this script's directory (scripts/ -> repo root).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

import alignment.hf_wrapper  # registers KeuralVisionConfig/Model with AutoModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
# BitsAndBytesConfig / prepare_model_for_kbit_training are imported lazily inside
# the loader only when 4-bit is requested — so the bf16 (--no-4bit) path works
# even where bitsandbytes has no working CUDA binary.

from alignment.projectors import LevelAwareProjector
from data.vlm_dataset import (
    caption_tar_stream, llava_json_stream, pull_batch,
    build_caption_batch, build_llava_batch, _normalize_conversation,
    PrefetchLoader,
)
from PIL import Image


# -- args ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Stage 2 projector alignment training (Keural Mid)")
parser.add_argument("--vision-ckpt", default="outputs/run/final",
                    help="Path to trained Stage 1 vision encoder checkpoint")
parser.add_argument("--llm-name", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF repo ID or local path to the LLM (any causal LM)")
parser.add_argument("--output-dir", default="outputs/projector")
parser.add_argument("--data-mode", choices=["caption-tar", "llava-json"], default="caption-tar",
                    help="Stage 2a = caption-tar (jpg/txt shards); Stage 2b = llava-json (instruction)")
parser.add_argument("--tar-dir", nargs="+", default=["data/cc3m"],
                    help="[caption-tar] One or more directories containing tar shards (*.tar)")
parser.add_argument("--jsonl-path", default=None,
                    help="[llava-json] Path to LLaVA-Instruct JSONL")
parser.add_argument("--image-root", default=None,
                    help="[llava-json] Root dir the JSONL 'image' paths are relative to")
parser.add_argument("--no-4bit", action="store_true",
                    help="Load the LLM in bf16 instead of 4-bit NF4 (use when bitsandbytes "
                         "is unavailable; fine on H200-class GPUs with LoRA)")
parser.add_argument("--max-steps", type=int, default=10000)
parser.add_argument("--batch-size", type=int, default=4,
                    help="Images per micro-batch (before grad accumulation)")
parser.add_argument("--grad-accum", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = batch_size x grad_accum)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42,
                    help="Seeds projector + LoRA init. Same on every rank; "
                         "data order is seeded per-rank in vlm_dataset.")
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--save-every", type=int, default=1000)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--val-jsonl", nargs="+", default=None,
                    help="Held-out JSONL(s) for periodic validation loss. Pass as\n                         lang:path pairs (e.g. en:/path/en_val.jsonl ko:/path/ko_val.jsonl)\n                         to get per-language numbers.")
parser.add_argument("--val-image-root", default=None, help="Image root for --val-jsonl")
parser.add_argument("--val-every", type=int, default=250, help="Run validation every N steps")
parser.add_argument("--val-samples", type=int, default=64, help="Records per validation pass")
parser.add_argument("--loader-workers", type=int, default=16,
                    help="Threads for image read+decode. The datasets live on NFS, so the\n                         naive serial path leaves the GPU I/O-starved.")
parser.add_argument("--token-budget", type=int, default=512,
                    help="Vision token budget passed to the encoder (Mid default 512)")
parser.add_argument("--image-size", type=int, default=384,
                    help="Square resize resolution for input images (Mid trains at 384)")
parser.add_argument("--max-caption-length", type=int, default=128,
                    help="Max tokenized caption length (in tokens)")
parser.add_argument("--hidden-dim", type=int, default=2048,
                    help="Projector MLP hidden dimension")
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"],
                    help="LoRA target module names in the LLM")
parser.add_argument("--init-from", default=None, metavar="CKPT_DIR",
                    help="Warm-start projector + LoRA weights from another "
                         "stage's checkpoint dir (e.g. Stage 2a's "
                         "checkpoint-5000), with a FRESH optimizer and LR "
                         "schedule. Required for Stage 2b -- --resume only "
                         "looks inside --output-dir, so 2b would otherwise "
                         "start from random weights.")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in --output-dir")
args = parser.parse_args()

# -- distributed (DDP) setup ---------------------------------------------------
# device_map="auto" splits the LLM ACROSS GPUs (pipeline parallel), so only one
# card computes at a time — measured at ~30-55% utilisation each. Under DDP each
# rank holds a FULL model copy and they train simultaneously on disjoint data.
import torch.distributed as dist
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
DDP = LOCAL_RANK != -1
if DDP:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{LOCAL_RANK}")
else:
    RANK, WORLD = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN = RANK == 0


def rprint(*a, **k):
    """Print on rank 0 only — otherwise every log line is duplicated per GPU."""
    if IS_MAIN:
        print(*a, **k)


rprint(f"Device: {device} | DDP: {DDP} (rank {RANK}/{WORLD})")
if device.type != "cuda":
    print("WARNING: training on CPU will be extremely slow. Use CUDA for Stage 2.")

# -- seeding -------------------------------------------------------------------
# Projector and LoRA weights are freshly initialised every launch. Without a
# fixed seed that init is different each run, so whether training survives the
# peak-LR phase is luck: the 2026-07-30 runs saw one init reach val ppl 23.9 by
# step 750 while another blew up to 124.5 on IDENTICAL data and hyperparameters.
# Seeding makes runs comparable and failures reproducible.
#
# The SAME seed on every rank is intentional: ranks must start from identical
# weights (DDP broadcasts rank 0's parameters at construction, so a mismatch
# would be silently overwritten rather than error). Data order is seeded
# separately per rank in vlm_dataset (1234 + rank) so ranks see disjoint shards.
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rprint(f"Seed: {args.seed} (all ranks; data order seeded per-rank separately)")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# -- LR schedule (cosine with linear warmup) -----------------------------------
def cosine_warmup_lr(step: int, peak_lr: float, warmup_steps: int,
                     max_steps: int, min_lr_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * min_lr_ratio + peak_lr * (1.0 - min_lr_ratio) * cosine


# -- load frozen vision encoder ------------------------------------------------
print(f"\nLoading vision encoder: {args.vision_ckpt}")
vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=torch.bfloat16)
vision = vision.to(device).eval()
# Non-persistent buffers get cast to bf16 (losing fp32 precision) by
# from_pretrained(dtype=bfloat16) — reinitialize them at fp32. pool_metadata is
# persistent (restored correctly) but we reset it defensively too.
try:
    vision.encoder.pool_metadata.data[:] = torch.tensor(
        [[0.5, 0.5, 1.0, 1.0]], device=vision.encoder.pool_metadata.device
    )
    for block in vision.encoder.blocks:
        rope = block.attn.rope
        i = torch.arange(0, rope.half_dim, 2, dtype=torch.float32)
        rope.inv_freq.data = (1.0 / (rope.base ** (i / rope.half_dim))).to(rope.inv_freq.device)
    print("  Reinitialized RoPE inv_freq / pool_metadata buffers at fp32")
except AttributeError as e:
    print(f"  WARNING: buffer reinit skipped ({e}) — verify encoder structure if results look wrong")
for p in vision.parameters():
    p.requires_grad_(False)
n_vis = sum(p.numel() for p in vision.parameters())
# Encoder embedding dim — read from the model, not hardcoded.
encoder_dim = vision.encoder.embed_dim
print(f"  {n_vis:,} params - all frozen | encoder_dim = {encoder_dim}")


# -- load tokenizer ------------------------------------------------------------
print(f"\nLoading tokenizer: {args.llm_name}")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# -- load LLM (4-bit NF4 by default, or bf16 with --no-4bit) --------------------
print(f"\nLoading LLM: {args.llm_name}")
if args.no_4bit:
    print("  Precision: bf16 (no quantization) — LoRA on top")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name, torch_dtype=torch.bfloat16,
        device_map={"": LOCAL_RANK} if DDP else "auto",
    )
else:
    print("  Quantization: 4-bit NF4 double-quant (bfloat16 compute)")
    from transformers import BitsAndBytesConfig  # lazy: only when 4-bit requested
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": LOCAL_RANK} if DDP else "auto",
    )
# LLM hidden dim — read from config, NOT hardcoded. This is what makes the
# script work with any HF causal LM without edits.
llm_dim = llm.config.hidden_size
print(f"  llm_dim (hidden_size) = {llm_dim}")

if args.no_4bit:
    for p in llm.parameters():
        p.requires_grad_(False)  # freeze base; LoRA adds trainable adapters
else:
    from peft import prepare_model_for_kbit_training  # lazy: 4-bit path only
    llm = prepare_model_for_kbit_training(llm)
lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=args.lora_targets,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
llm = get_peft_model(llm, lora_cfg)
llm.print_trainable_parameters()


# -- build projector -----------------------------------------------------------
print(f"\nBuilding LevelAwareProjector: {encoder_dim} -> {args.hidden_dim} -> {llm_dim}")
projector = LevelAwareProjector(
    encoder_dim=encoder_dim,
    llm_dim=llm_dim,
    hidden_dim=args.hidden_dim,
).to(device).to(torch.bfloat16)
n_proj = sum(p.numel() for p in projector.parameters())
print(f"  {n_proj:,} params ({n_proj/1e6:.1f}M) - all trainable")


# -- optimizer over projector + LoRA params ------------------------------------
lora_params = [p for p in llm.parameters() if p.requires_grad]
all_trainable = list(projector.parameters()) + lora_params

# -- DDP wrapping --------------------------------------------------------------
# Two trainable groups need gradient all-reduce: the projector and the LoRA
# adapters inside the LLM. find_unused_parameters=True is required on the LLM
# because LoRA only touches a subset of modules, so many params legitimately
# receive no gradient and DDP would otherwise error.
# projector_module / llm_module are the UNWRAPPED handles, used for saving and
# for validation (calling .module avoids DDP's forward-hook bookkeeping).
projector_module, llm_module = projector, llm
if DDP:
    from torch.nn.parallel import DistributedDataParallel as DDPWrap
    projector = DDPWrap(projector, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    llm = DDPWrap(llm, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                  find_unused_parameters=True)
    rprint(f"  DDP: projector + LoRA wrapped across {WORLD} ranks")
optimizer = torch.optim.AdamW(
    all_trainable,
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
)
print(f"\nOptimizer: AdamW, lr={args.lr}, wd={args.weight_decay}")
print(f"  Trainable params: {sum(p.numel() for p in all_trainable):,}")


# -- checkpoint discovery ------------------------------------------------------
def numbered_checkpoints(d: Path):
    """checkpoint-<int> dirs, ascending. Anything with a non-numeric suffix
    (checkpoint-1000.bak, a manual copy) is IGNORED rather than crashing
    int() — a stray directory used to take down checkpoint rotation, and
    therefore the whole run, at the first save."""
    out = []
    for p in d.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            out.append((int(p.name.split("-")[-1]), p))
        except ValueError:
            print(f"[ckpt] ignoring non-numeric checkpoint dir: {p.name}")
    return [p for _, p in sorted(out)]


# -- weight loading ------------------------------------------------------------
def load_trained_weights(ckpt: Path) -> None:
    """Load projector + LoRA weights from a checkpoint dir. Does NOT touch the
    optimizer or the step counter — shared by --resume (which restores those
    separately) and --init-from (which deliberately starts a fresh schedule)."""
    proj_path = ckpt / "projector.pt"
    if not proj_path.exists():
        raise SystemExit(f"ERROR: {proj_path} missing")
    # Load into the UNWRAPPED module: under DDP `projector` is a
    # DistributedDataParallel whose keys are all "module."-prefixed, so loading
    # the saved (unprefixed) state_dict into it raises a key-mismatch error.
    projector_module.load_state_dict(torch.load(proj_path, map_location=device))
    print("  Projector weights loaded")

    # LoRA adapters are ~5M of the ~14M trainable params. Skipping them silently
    # reinitialises a third of the learned weights while the loss curve looks
    # merely "a bit worse" rather than obviously broken.
    adapter_dir = ckpt / "lora_adapter"
    adapter_file = next(
        (adapter_dir / n for n in ("adapter_model.safetensors",
                                   "adapter_model.bin")
         if (adapter_dir / n).exists()), None)
    if adapter_file is None:
        raise SystemExit(
            f"ERROR: no adapter weights in {adapter_dir}. Continuing without "
            "them would discard the trained LoRA.")
    if adapter_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        lora_sd = load_file(str(adapter_file))
    else:
        lora_sd = torch.load(adapter_file, map_location="cpu")

    # peft's set_peft_model_state_dict() is unusable in this environment: peft
    # 0.19.1 unconditionally calls _maybe_shard_state_dict_for_tp(), which does
    #   from transformers.integrations.tensor_parallel import EmbeddingParallel
    # and that symbol does not exist in transformers 4.57.6 -> ImportError.
    #
    # Do the remap ourselves. save_pretrained() writes "...lora_A.weight", while
    # the live model names the same parameter "...lora_A.<adapter_name>.weight".
    adapter_name = "default"
    remapped = {}
    for k, v in lora_sd.items():
        for tag in ("lora_A.", "lora_B.", "lora_embedding_A.", "lora_embedding_B."):
            if tag in k:
                k = k.replace(tag, f"{tag}{adapter_name}.")
                break
        remapped[k] = v

    model_keys = set(llm_module.state_dict().keys())
    matched = {k: v for k, v in remapped.items() if k in model_keys}
    if len(matched) != len(remapped):
        sample = sorted(set(remapped) - model_keys)[:3]
        raise SystemExit(
            f"ERROR: {len(remapped) - len(matched)}/{len(remapped)} LoRA tensors "
            f"did not match model parameters, e.g. {sample}. Refusing to "
            "continue with a partially loaded adapter.")
    # strict=False because `matched` intentionally covers only LoRA params;
    # every base-model weight is legitimately absent from it.
    llm_module.load_state_dict(matched, strict=False)
    print(f"  LoRA adapter loaded ({len(matched)} tensors)")


# -- warm start (Stage 2b picks up where Stage 2a left off) --------------------
# Stage 2b writes to a DIFFERENT --output-dir than 2a, and --resume only ever
# looks inside its own output dir. Without --init-from, Stage 2b would start
# from a randomly initialised projector at lr 2e-5 -- a fine-tuning LR far too
# low to train a projector from scratch -- silently discarding all of Stage 2a
# and making the two-stage design pointless.
if args.init_from:
    src = Path(args.init_from)
    if not src.is_dir():
        raise SystemExit(f"ERROR: --init-from {src} is not a directory")
    print(f"\nWarm-starting from {src}")
    load_trained_weights(src)
    print("  Fresh optimizer + LR schedule (weights only, step stays 0)")

# -- resume from checkpoint ----------------------------------------------------
start_step = 0
if args.resume:
    ckpts = numbered_checkpoints(output_dir)
    if ckpts:
        latest = ckpts[-1]
        print(f"\nResuming from {latest.name}")
        load_trained_weights(latest)
        state_path = latest / "training_state.pt"
        if not state_path.exists():
            raise SystemExit(
                f"ERROR: {state_path} missing - refusing to resume without "
                "optimizer state (Adam moments would restart cold).")
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"] + 1
        print(f"  Optimizer state loaded. Resuming at step {start_step}")
    else:
        print("  No checkpoint found - starting from scratch")


# -- data stream (caption-tar for 2a, llava-json for 2b) -----------------------
# caption_tar_stream / llava_json_stream / pull_batch imported from
# data.vlm_dataset (unit-tested separately).
if args.data_mode == "caption-tar":
    data_stream = caption_tar_stream(args.tar_dir)
else:  # llava-json
    if not args.jsonl_path or not args.image_root:
        raise SystemExit("--data-mode llava-json requires --jsonl-path and --image-root")
    data_stream = llava_json_stream(args.jsonl_path, args.image_root, yield_path=True,
                                    rank=RANK, world_size=WORLD)


# -- checkpoint save / rotate --------------------------------------------------
# Most recent validation result, and the best checkpoint seen so far. Populated
# by the training loop; validation runs BEFORE the save at any shared step, so
# LATEST_VAL is current when save_checkpoint reads it.
LATEST_VAL: dict = {}
BEST = {"ppl": float("inf"), "step": None}


def save_checkpoint(step: int) -> None:
    if not IS_MAIN:
        return  # rank 0 owns checkpointing; other ranks hold identical weights
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(projector_module.state_dict(), ckpt_dir / "projector.pt")
    llm_module.save_pretrained(ckpt_dir / "lora_adapter")

    # Inference-critical config. Weights alone do NOT describe how to run the
    # model: token_budget, image_size and hidden_dim must match training exactly
    # or output degrades silently. demo_vlm.py defaulted to image_size 384 while
    # Stage 2 trained at 512 precisely because nothing in the checkpoint said
    # otherwise. Anyone loading this dir months from now reads it from here.
    config = {
        "step": step,
        "stage": Path(args.output_dir).name,
        # -- must match at inference --
        "token_budget": args.token_budget,
        "image_size": args.image_size,
        "hidden_dim": args.hidden_dim,
        "encoder_dim": encoder_dim,
        "llm_dim": llm_dim,
        "llm_name": args.llm_name,
        "vision_ckpt": args.vision_ckpt,
        # -- provenance --
        "seed": args.seed,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "world_size": WORLD,
        "eff_batch": args.batch_size * args.grad_accum * WORLD,
        "max_caption_length": args.max_caption_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "init_from": args.init_from,
        "data": args.jsonl_path or args.tar_dir,
        # -- quality at this checkpoint --
        "val": {lang: {"loss": L, "ppl": ppl}
                for lang, (L, ppl) in LATEST_VAL.items()},
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2))

    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "encoder_dim": encoder_dim, "llm_dim": llm_dim,
                "llm_name": args.llm_name, "seed": args.seed,
                # RNG state so a resumed run continues the same stochastic
                # stream (dropout, shuffling) instead of silently diverging.
                "rng": {"python": random.getstate(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None}},
               ckpt_dir / "training_state.pt")

    val_note = ""
    if LATEST_VAL:
        val_note = "  val: " + " | ".join(f"{lang} ppl={ppl:.2f}"
                                          for lang, (_, ppl) in LATEST_VAL.items())
        primary = next(iter(LATEST_VAL.values()))[1]
        if primary < BEST["ppl"]:
            BEST.update(ppl=primary, step=step)
            (output_dir / "best_checkpoint.json").write_text(json.dumps(
                {"step": step, "ppl": primary,
                 "checkpoint": f"checkpoint-{step}"}, indent=2))
            val_note += "  <- BEST so far"
    print(f"[train] Saved checkpoint-{step} to {ckpt_dir}{val_note}")

    # Rotation keeps the last 3, but NEVER deletes the best-scoring checkpoint.
    # Validation can worsen late in a run; unconditional rotation would delete
    # the best model and leave only worse ones.
    for old in numbered_checkpoints(output_dir)[:-3]:
        if BEST["step"] is not None and old.name == f"checkpoint-{BEST['step']}":
            print(f"[train] Keeping {old.name} (best val ppl={BEST['ppl']:.2f})")
            continue
        shutil.rmtree(old, ignore_errors=True)
        print(f"[train] Removed old checkpoint {old.name}")


# -- training loop -------------------------------------------------------------
eff_batch = args.batch_size * args.grad_accum * WORLD
rprint(f"\nStarting training:")
rprint(f"  Steps: {args.max_steps} | Micro-batch: {args.batch_size} | Grad-accum: {args.grad_accum} | Eff-batch: {eff_batch}")
rprint(f"  LR: {args.lr} | Warmup: {args.warmup_steps} | Token budget: {args.token_budget} | Image size: {args.image_size}")

embed_fn = llm_module.get_input_embeddings()

# Threaded prefetch: overlaps NFS image reads/decode with GPU compute.
loader = PrefetchLoader(data_stream, args.batch_size, transform,
                        workers=args.loader_workers,
                        from_paths=(args.data_mode == 'llava-json'))
rprint(f'[data] prefetch: {args.loader_workers} loader threads')

# -- held-out validation sets -------------------------------------------------
# Periodic teacher-forced loss on data the model never trains on. Without this
# the only signal is training loss, which cannot distinguish learning from
# memorisation. Reported per language so Korean quality is visible separately
# (no public Korean VLM benchmark exists to check it against).
val_sets = {}
if args.val_jsonl:
    for spec in args.val_jsonl:
        lang, _, path = spec.partition(":")
        if not path:
            lang, path = "val", spec
        recs = [json.loads(l) for l in open(path)][: args.val_samples]
        val_sets[lang] = recs
        print(f"[val] {lang}: {len(recs)} held-out records from {path}")


@torch.no_grad()
def run_validation():
    """Teacher-forced loss on the held-out sets. Returns {lang: (loss, ppl)}."""
    was_training = projector_module.training
    projector_module.eval(); llm_module.eval()
    out = {}
    for lang, recs in val_sets.items():
        tot_loss, tot_tok = 0.0, 0
        for r in recs:
            msgs = _normalize_conversation(r)
            img_p = Path(args.val_image_root) / r["image"]
            if not msgs or not img_p.exists():
                continue
            try:
                pv = transform(Image.open(img_p).convert("RGB")).unsqueeze(0).to(
                    device, dtype=torch.bfloat16)
            except Exception:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                vis = projector_module(vision(pv, token_budget=args.token_budget))
                ed = next(embed_fn.parameters()).device
                ids, attn, labels = build_llava_batch(
                    [msgs], tokenizer, args.max_caption_length, ed)
                inp = torch.cat([vis.to(ed), embed_fn(ids).to(torch.bfloat16)], dim=1)
                full_attn = torch.cat(
                    [torch.ones(vis.shape[:2], dtype=torch.long, device=ed), attn], dim=1)
                vis_lab = torch.full(vis.shape[:2], -100, dtype=torch.long, device=ed)
                o = llm_module(inputs_embeds=inp, attention_mask=full_attn,
                               labels=torch.cat([vis_lab, labels], dim=1))
            ntok = int((labels != -100).sum())
            if ntok:
                tot_loss += o.loss.item() * ntok; tot_tok += ntok
        if tot_tok:
            L = tot_loss / tot_tok
            out[lang] = (L, float(torch.exp(torch.tensor(L))))
        else:
            # Every record was skipped -- almost always because `image` paths in
            # this val file do not resolve under --val-image-root. That used to
            # drop the language from the report with no message at all, so a
            # bilingual run could silently monitor only one of its two languages.
            print(f"[val] WARNING: '{lang}' produced NO usable records "
                  f"({len(recs)} loaded). Check that its image paths resolve "
                  f"under --val-image-root={args.val_image_root}", flush=True)
    if was_training:
        projector_module.train(); llm_module.train()
    return out
log_path = output_dir / "training_log.jsonl"

projector.train()
llm.train()
optimizer.zero_grad(set_to_none=True)

accum_count = 0
accum_loss = 0.0
step = start_step
start_time = time.time()

try:
    while step < args.max_steps:
        pixel_values, targets = next(loader)
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            with torch.no_grad():
                vis_out = vision(pixel_values, token_budget=args.token_budget)

            visual_embeds = projector(vis_out)  # (B, N_vis, llm_dim)
            embed_device = next(embed_fn.parameters()).device

            # Text tokenization + label building. caption-tar: label = caption;
            # llava-json: label = assistant turns only (masking done in module).
            if args.data_mode == "caption-tar":
                input_ids, attn_mask, text_labels = build_caption_batch(
                    targets, tokenizer, args.max_caption_length, embed_device)
            else:
                input_ids, attn_mask, text_labels = build_llava_batch(
                    targets, tokenizer, args.max_caption_length, embed_device)

            B, N_vis = visual_embeds.shape[:2]

            with torch.no_grad():
                text_embeds = embed_fn(input_ids).to(torch.bfloat16)

            inputs_embeds = torch.cat(
                [visual_embeds.to(embed_device), text_embeds], dim=1
            )

            vis_attn = torch.ones(B, N_vis, dtype=torch.long, device=embed_device)
            full_attn = torch.cat([vis_attn, attn_mask], dim=1)

            # text_labels already has padding (2a) / user turns (2b) masked to
            # -100 by the batch builder. Vision positions are never a target.
            vis_labels = torch.full((B, N_vis), -100, dtype=torch.long, device=embed_device)
            labels = torch.cat([vis_labels, text_labels], dim=1)

            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum

        # Under DDP, all-reduce only on the FINAL micro-step of the accumulation
        # window; syncing every micro-step wastes interconnect bandwidth.
        is_last_micro = (accum_count + 1) >= args.grad_accum
        if DDP and not is_last_micro:
            with projector.no_sync(), llm.no_sync():
                loss.backward()
        else:
            loss.backward()
        accum_loss += loss.item()
        accum_count += 1

        if accum_count >= args.grad_accum:
            grad_norm = nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            current_lr = cosine_warmup_lr(step, args.lr, args.warmup_steps, args.max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                steps_done = max(step - start_step + 1, 1)
                secs_per_step = elapsed / steps_done
                eta_secs = int(secs_per_step * (args.max_steps - step))
                eta_h, eta_m = divmod(eta_secs // 60, 60)
                pct = step / args.max_steps * 100
                line = (
                    f"[train] step={step:5d}/{args.max_steps} ({pct:.1f}%) "
                    f"ETA={eta_h}h{eta_m:02d}m  lr={current_lr:.2e}  "
                    f"loss={accum_loss:.4f}  grad_norm={float(grad_norm):.2e}"
                )
                rprint(line)
                if IS_MAIN:
                    with log_path.open("a") as f:
                        f.write(json.dumps({
                            "step": step, "loss": accum_loss,
                            "lr": current_lr, "grad_norm": float(grad_norm),
                        }) + "\n")

            accum_loss = 0.0

            # -- periodic held-out validation --------------------------------
            if IS_MAIN and val_sets and step > 0 and step % args.val_every == 0:
                vres = run_validation()
                if vres:
                    LATEST_VAL.clear()
                    LATEST_VAL.update(vres)
                    parts = " | ".join(
                        f"{lang} loss={L:.4f} ppl={ppl:.2f}" for lang, (L, ppl) in vres.items())
                    rprint(f"[val]   step={step:5d}  {parts}", flush=True)
                    with log_path.open("a") as f:
                        f.write(json.dumps({
                            "step": step, "type": "val",
                            **{f"val_{lang}_loss": L for lang, (L, _) in vres.items()},
                            **{f"val_{lang}_ppl": ppl for lang, (_, ppl) in vres.items()},
                        }) + "\n")

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(step)

            step += 1

except KeyboardInterrupt:
    print(f"\n[train] Interrupted at step {step} - saving checkpoint...")
    save_checkpoint(step)
    print("[train] Checkpoint saved. Exiting.")
    sys.exit(0)

rprint(f"\n[train] Training complete - {step} steps")
save_checkpoint(step)
print(f"\nCheckpoints saved to: {output_dir}")
print(f"  projector weights:  checkpoint-{step}/projector.pt")
print(f"  LoRA adapter:       checkpoint-{step}/lora_adapter/")


if DDP:
    dist.barrier()
    dist.destroy_process_group()
