"""
Keural Mid-Level Vision Encoder — Evaluation Script.

Evaluates a trained checkpoint on standard vision benchmarks:
  1. Zero-shot ImageNet classification (top-1 / top-5)
  2. Image-text retrieval (Flickr30k — R@1, R@5, R@10)
  3. Linear probe on CIFAR-100

Usage:
    cd /home/work/Hika/vision-encoder/mid
    python scripts/evaluate.py \
        --checkpoint outputs/run/checkpoint-10000 \
        --benchmarks imagenet flickr30k cifar100 \
        --batch-size 128

    # Single benchmark:
    python scripts/evaluate.py \
        --checkpoint outputs/run/checkpoint-10000 \
        --benchmarks imagenet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from alignment.hf_wrapper import KeuralVisionConfig, KeuralVisionModel

# Must match train_mid.ALIGNMENT_STATE_FILE — the sidecar holding the trained
# text projection (and logit_scale/bias, teacher projections).
ALIGNMENT_STATE_FILE = "alignment_state.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> KeuralVisionModel:
    """Load a trained KeuralVisionModel from a checkpoint directory."""
    print(f"[eval] Loading model from {checkpoint_path}...")
    model = KeuralVisionModel.from_pretrained(checkpoint_path)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] Loaded {n_params:,} params ({n_params/1e6:.1f}M)")
    # Single GPU eval — DataParallel incompatible with KeuralEncoderOutput dataclass
    return model


class _EvalTextEncoder:
    """Text encoder for evaluation that reproduces the TRAINING text space.

    Training aligned image embeddings against ``proj(CLIP_pooler_output)`` where
    ``proj`` is a trained Linear(768->768). Evaluation MUST apply the same
    projection, otherwise image and text embeddings live in different spaces and
    zero-shot/retrieval collapses to chance. The projection is loaded from the
    ``alignment_state.pt`` sidecar written next to the checkpoint.

    If the sidecar is missing (legacy checkpoint trained before this fix), we
    emit a loud warning and fall back to the raw CLIP pooler output — which will
    NOT match the trained space; numbers from that path are not meaningful.
    """

    def __init__(self, checkpoint_path: str, device: torch.device,
                 model_name: str = "openai/clip-vit-large-patch14",
                 embed_dim: int = 768):
        from transformers import CLIPTokenizer, CLIPTextModel
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
        raw_dim = self.text_model.config.hidden_size

        self.proj = torch.nn.Linear(raw_dim, embed_dim).to(device).eval()
        self.has_proj = False
        sidecar = Path(checkpoint_path) / ALIGNMENT_STATE_FILE
        if sidecar.exists():
            state = torch.load(sidecar, map_location="cpu")
            # text_proj keys are prefixed "proj." (from FrozenTextEncoder).
            proj_sd = {k[len("proj."):]: v for k, v in state["text_proj"].items()
                       if k.startswith("proj.")}
            self.proj.load_state_dict(proj_sd)
            self.proj.to(device)
            self.has_proj = True
            print(f"[eval] Loaded trained text projection from {ALIGNMENT_STATE_FILE} "
                  "(text embeddings now match the training space).")
        else:
            print(f"[eval] WARNING: {ALIGNMENT_STATE_FILE} not found in checkpoint. "
                  "This is a LEGACY checkpoint trained before alignment params were "
                  "persisted. Falling back to RAW CLIP text embeddings, which do NOT "
                  "match the trained image space — zero-shot/retrieval numbers will be "
                  "meaningless. Retrain (or resume) to produce a valid checkpoint.")

    @torch.no_grad()
    def __call__(self, captions: list[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings in the training space, (B, embed_dim)."""
        inputs = self.tokenizer(
            captions, padding=True, truncation=True, max_length=77, return_tensors="pt",
        ).to(self.device)
        pooled = self.text_model(**inputs).pooler_output.float()
        if self.has_proj:
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)


@torch.no_grad()
def encode_images(model, dataloader, device, token_budget=None) -> torch.Tensor:
    """Encode all images through the model, return pooled embeddings."""
    all_embeds = []
    for batch in tqdm(dataloader, desc="Encoding images"):
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        elif isinstance(batch, dict):
            images = batch["pixel_values"].to(device)
        else:
            images = batch.to(device)
        out = model(pixel_values=images, token_budget=token_budget)
        pooled = F.normalize(out.pooled, dim=-1)
        all_embeds.append(pooled.cpu())
    return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# Benchmark 1: Zero-shot ImageNet Classification
# ---------------------------------------------------------------------------

def eval_imagenet(model, device, text_encoder, batch_size=128, data_root=None, token_budget=None):
    """Zero-shot ImageNet classification using CLIP-style text prompts.

    ``text_encoder`` is an _EvalTextEncoder that applies the trained text
    projection so text embeddings match the training image space.
    """
    print("\n" + "=" * 60)
    print("  Benchmark: Zero-shot ImageNet Classification")
    print("=" * 60)

    try:
        from torchvision import transforms
        from transformers import CLIPTokenizer, CLIPTextModel
        from PIL import Image
        import pandas as pd
        import io
    except ImportError:
        print("[eval] ERROR: required packages not installed.")
        return None

    # Check for ImageNet validation set (parquet format)
    if data_root is None:
        data_root = "/home/work/keural-model/vision-encoder-dataset/imagenet"

    parquet_dir = Path(data_root) / "data"
    parquet_files = sorted(parquet_dir.glob("validation-*.parquet")) if parquet_dir.exists() else []

    if not parquet_files:
        print(f"[eval] ImageNet validation parquets not found at {parquet_dir}")
        return None

    # Load all validation data from parquets
    print(f"[eval] Loading {len(parquet_files)} parquet files...")
    all_images, all_labels = [], []
    for pf in tqdm(parquet_files, desc="Loading parquets"):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            all_images.append(img)
            all_labels.append(row["label"])

    print(f"[eval] ImageNet val: {len(all_images)} images, {max(all_labels)+1} classes")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build text embeddings for each class using ImageNet class names.
    # Uses the trained text projection (via text_encoder) so the embeddings live
    # in the same space as the trained image embeddings.
    print("[eval] Building class text embeddings...")

    # Standard ImageNet class names (load from HF datasets)
    try:
        from datasets import load_dataset
        ds_info = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        class_names = ds_info.features["label"].names
    except Exception:
        # Fallback: use label indices
        num_classes = max(all_labels) + 1
        class_names = [str(i) for i in range(num_classes)]

    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of the large {}.",
        "a photo of the small {}.",
        "a photo of many {}.",
    ]

    class_embeds = []
    for cls_name in tqdm(class_names, desc="Text embeddings"):
        cls_name_clean = cls_name.replace("_", " ")
        prompts = [t.format(cls_name_clean) for t in templates]
        # text_encoder applies trained proj + L2-normalizes each prompt.
        embeds = text_encoder(prompts)
        # Average the 5 templates, then renormalize.
        mean_embed = F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)
        class_embeds.append(mean_embed.cpu())

    class_embeds = torch.cat(class_embeds, dim=0).to(device)  # (1000, 768)
    print(f"[eval] Class embeddings: {class_embeds.shape}")

    # Encode images and compute accuracy
    correct_1, correct_5, total = 0, 0, 0
    for i in tqdm(range(0, len(all_images), batch_size), desc="Evaluating"):
        batch_imgs = all_images[i:i + batch_size]
        batch_labels = torch.tensor(all_labels[i:i + batch_size], device=device)
        batch_tensor = torch.stack([transform(img) for img in batch_imgs]).to(device)

        with torch.no_grad():
            out = model(pixel_values=batch_tensor, token_budget=token_budget)
            img_embeds = F.normalize(out.pooled, dim=-1)
            logits = img_embeds @ class_embeds.T

        _, top5_preds = logits.topk(5, dim=-1)
        correct_1 += (top5_preds[:, 0] == batch_labels).sum().item()
        correct_5 += (top5_preds == batch_labels.unsqueeze(1)).any(dim=1).sum().item()
        total += batch_labels.size(0)

    top1 = 100.0 * correct_1 / total
    top5 = 100.0 * correct_5 / total
    print(f"\n[eval] ImageNet Zero-shot Results:")
    print(f"       Top-1 Accuracy: {top1:.2f}%")
    print(f"       Top-5 Accuracy: {top5:.2f}%")
    return {"imagenet_top1": top1, "imagenet_top5": top5}


# ---------------------------------------------------------------------------
# Benchmark 2: Image-Text Retrieval (Flickr30k)
# ---------------------------------------------------------------------------

def eval_flickr30k(model, device, text_encoder, batch_size=128, data_root=None, token_budget=None):
    """Image-text retrieval on Flickr30k test split.

    ``text_encoder`` applies the trained text projection (training space).
    """
    print("\n" + "=" * 60)
    print("  Benchmark: Flickr30k Image-Text Retrieval")
    print("=" * 60)

    try:
        from torchvision import transforms
        from transformers import CLIPTokenizer, CLIPTextModel
        from PIL import Image
    except ImportError:
        print("[eval] ERROR: required packages not installed.")
        return None

    # Try loading Flickr30k
    flickr_path = data_root or "/home/work/keural-model/vision-encoder-dataset/flickr30k"
    images_dir = Path(flickr_path) / "flickr30k-images"
    captions_file = Path(flickr_path) / "flickr_annotations_30k.csv"

    if not images_dir.exists() or not captions_file.exists():
        print(f"[eval] Flickr30k not found at {flickr_path}")
        print(f"       Expected: {images_dir} and {captions_file}")
        return None

    # Parse captions from CSV (format: raw, sentids, split, filename, img_id)
    import csv, ast
    image_captions = {}
    with open(captions_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test":
                filename = row["filename"]
                captions = ast.literal_eval(row["raw"])
                image_captions[filename] = captions

    # Use test split (or first 1000 if test is larger)
    image_names = sorted(image_captions.keys())[:1000]
    captions_list = [image_captions[name][0] for name in image_names]  # first caption
    print(f"[eval] Flickr30k: {len(image_names)} test images")

    transform = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Encode images
    print(f"[eval] Encoding {len(image_names)} images...")
    all_img_embeds = []
    for i in tqdm(range(0, len(image_names), batch_size), desc="Images"):
        batch_names = image_names[i:i + batch_size]
        batch_imgs = []
        for name in batch_names:
            img = Image.open(images_dir / name).convert("RGB")
            batch_imgs.append(transform(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = model(pixel_values=batch_tensor, token_budget=token_budget)
            embeds = F.normalize(out.pooled, dim=-1)
        all_img_embeds.append(embeds.cpu())
    img_embeds = torch.cat(all_img_embeds, dim=0)

    # Encode captions through the trained text projection (training space).
    print(f"[eval] Encoding {len(captions_list)} captions...")
    all_text_embeds = []
    for i in tqdm(range(0, len(captions_list), batch_size), desc="Captions"):
        batch_caps = captions_list[i:i + batch_size]
        embeds = text_encoder(batch_caps)  # trained proj + L2-normalized
        all_text_embeds.append(embeds.cpu())
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # Compute retrieval metrics
    sim = img_embeds @ text_embeds.T  # (N, N)

    # Image -> Text retrieval
    i2t_ranks = []
    for i in range(len(image_names)):
        sorted_indices = sim[i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)

    # Text -> Image retrieval
    t2i_ranks = []
    for i in range(len(captions_list)):
        sorted_indices = sim[:, i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)

    i2t_ranks = torch.tensor(i2t_ranks, dtype=torch.float)
    t2i_ranks = torch.tensor(t2i_ranks, dtype=torch.float)

    results = {
        "i2t_R@1": (i2t_ranks <= 1).float().mean().item() * 100,
        "i2t_R@5": (i2t_ranks <= 5).float().mean().item() * 100,
        "i2t_R@10": (i2t_ranks <= 10).float().mean().item() * 100,
        "t2i_R@1": (t2i_ranks <= 1).float().mean().item() * 100,
        "t2i_R@5": (t2i_ranks <= 5).float().mean().item() * 100,
        "t2i_R@10": (t2i_ranks <= 10).float().mean().item() * 100,
    }

    print(f"\n[eval] Flickr30k Retrieval Results:")
    print(f"       Image→Text  R@1={results['i2t_R@1']:.1f}  R@5={results['i2t_R@5']:.1f}  R@10={results['i2t_R@10']:.1f}")
    print(f"       Text→Image  R@1={results['t2i_R@1']:.1f}  R@5={results['t2i_R@5']:.1f}  R@10={results['t2i_R@10']:.1f}")
    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Linear Probe on CIFAR-100
# ---------------------------------------------------------------------------

def eval_cifar100(model, device, batch_size=256, token_budget=None):
    """Linear probe classification on CIFAR-100."""
    print("\n" + "=" * 60)
    print("  Benchmark: CIFAR-100 Linear Probe")
    print("=" * 60)

    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("[eval] ERROR: torchvision not installed.")
        return None

    transform = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("[eval] Downloading CIFAR-100...")
    train_ds = datasets.CIFAR100(root="/tmp/cifar100", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100(root="/tmp/cifar100", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features
    print(f"[eval] Extracting train features ({len(train_ds)} images)...")
    train_features, train_labels = [], []
    for images, labels in tqdm(train_loader, desc="Train"):
        images = images.to(device)
        with torch.no_grad():
            out = model(pixel_values=images, token_budget=token_budget)
            pooled = out.pooled.cpu()
        train_features.append(pooled)
        train_labels.append(labels)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print(f"[eval] Extracting test features ({len(test_ds)} images)...")
    test_features, test_labels = [], []
    for images, labels in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        with torch.no_grad():
            out = model(pixel_values=images, token_budget=token_budget)
            pooled = out.pooled.cpu()
        test_features.append(pooled)
        test_labels.append(labels)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Train linear classifier
    print("[eval] Training linear probe (100 epochs)...")
    embed_dim = train_features.shape[1]
    linear = torch.nn.Linear(embed_dim, 100).to(device)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = torch.nn.CrossEntropyLoss()

    train_features_gpu = train_features.to(device)
    train_labels_gpu = train_labels.to(device)

    for epoch in range(100):
        # Mini-batch training
        perm = torch.randperm(len(train_features_gpu))
        total_loss = 0
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            logits = linear(train_features_gpu[idx])
            loss = criterion(logits, train_labels_gpu[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    # Evaluate
    test_features_gpu = test_features.to(device)
    with torch.no_grad():
        logits = linear(test_features_gpu)
        preds = logits.argmax(dim=-1).cpu()
        accuracy = (preds == test_labels).float().mean().item() * 100

    print(f"\n[eval] CIFAR-100 Linear Probe Results:")
    print(f"       Top-1 Accuracy: {accuracy:.2f}%")
    return {"cifar100_linear_top1": accuracy}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Keural Vision Encoder Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--benchmarks", nargs="+", default=["cifar100"],
                        choices=["imagenet", "flickr30k", "cifar100"],
                        help="Benchmarks to run (default: cifar100)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--imagenet-path", type=str, default=None, help="Path to ImageNet val/")
    parser.add_argument("--flickr-path", type=str, default=None, help="Path to Flickr30k/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Fixed token budget to use (e.g. 512 to match training, which "
                             "always used config.tokenizer.default_token_budget rather than "
                             "resolution-based auto-scaling). Default: auto-scale by resolution "
                             "(model's default inference behavior).")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    # Build the text encoder ONCE (loads CLIP + trained projection from sidecar).
    # CIFAR-100 is a linear probe on image features and needs no text encoder.
    needs_text = any(b in args.benchmarks for b in ("imagenet", "flickr30k"))
    text_encoder = _EvalTextEncoder(args.checkpoint, device) if needs_text else None

    all_results = {"checkpoint": args.checkpoint, "token_budget": args.token_budget}
    t0 = time.time()

    if "imagenet" in args.benchmarks:
        result = eval_imagenet(model, device, text_encoder, args.batch_size, args.imagenet_path,
                                token_budget=args.token_budget)
        if result:
            all_results.update(result)

    if "flickr30k" in args.benchmarks:
        result = eval_flickr30k(model, device, text_encoder, args.batch_size, args.flickr_path,
                                 token_budget=args.token_budget)
        if result:
            all_results.update(result)

    if "cifar100" in args.benchmarks:
        result = eval_cifar100(model, device, args.batch_size, token_budget=args.token_budget)
        if result:
            all_results.update(result)

    elapsed = time.time() - t0
    all_results["eval_time_s"] = round(elapsed, 1)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in all_results.items():
        if k not in ("checkpoint", "eval_time_s", "token_budget"):
            print(f"  {k:25s} : {v:.2f}%")
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Token budget: {args.token_budget if args.token_budget is not None else 'auto (resolution-scaled)'}")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 60)

    # Save results. Suffix the filename by token budget so a fixed-budget rerun
    # never clobbers the default auto-scaled eval_results.json already on disk.
    suffix = f"_budget{args.token_budget}" if args.token_budget is not None else ""
    results_path = Path(args.checkpoint) / f"eval_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results saved to {results_path}")


if __name__ == "__main__":
    main()
