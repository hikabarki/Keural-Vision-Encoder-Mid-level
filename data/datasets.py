"""
Keural Vision Encoder — Data Pipeline.

Reads WebDataset tar shards (DataComp / COYO img2dataset format) via a pre-built
index TSV, serves bucketed image-text pairs to the training loop.

Classes
-------
ImageTextPair           One sample returned by every dataset.
SyntheticPairDataset    Random tensors for smoke-testing (no data needed).
CC3MDataset             WebDataset tar shards + index TSV → random-access dataset.
CombinedImageTextDataset  Concatenates multiple CC3MDataset instances.
BucketedSampler         Groups indices by resolution bucket for efficient batching.
FrozenTextEncoder       Frozen pretrained text encoder + trainable linear projection.

Functions
---------
build_index_from_shards   Scans tar shards and writes a fresh index TSV.
build_collate_fn          Returns a collate function for DataLoader.
"""

from __future__ import annotations

import csv
import hashlib
import pickle
from collections import OrderedDict
import io
import logging
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None  # type: ignore
    UnidentifiedImageError = Exception  # type: ignore

try:
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
except ImportError:
    T = None  # type: ignore
    InterpolationMode = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ImageTextPair
# ---------------------------------------------------------------------------

@dataclass
class ImageTextPair:
    """One training sample."""
    pixel_values: torch.Tensor  # (3, H, W), float32, values in [0, 1]
    caption: str
    bucket_idx: int


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

def _assign_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> int:
    """Assign to the bucket whose aspect ratio is closest to the image's."""
    if not buckets:
        return 0
    if width <= 0 or height <= 0:
        return 0
    img_aspect = width / height
    best, best_diff = 0, float("inf")
    for i, (bh, bw) in enumerate(buckets):
        diff = abs(img_aspect - bw / bh)
        if diff < best_diff:
            best_diff = diff
            best = i
    return best


def _make_transforms(
    bucket_h: int, bucket_w: int, augment: bool
) -> "T.Compose":
    """Torchvision transform pipeline for one bucket size."""
    if augment:
        return T.Compose([
            T.RandomResizedCrop(
                (bucket_h, bucket_w),
                scale=(0.75, 1.0),
                ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            T.ToTensor(),
        ])
    return T.Compose([
        T.Resize(
            (bucket_h, bucket_w),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.CenterCrop((bucket_h, bucket_w)),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# SyntheticPairDataset
# ---------------------------------------------------------------------------

class SyntheticPairDataset(Dataset):
    """Random image-text pairs for smoke-testing. No files needed."""

    _CAPTIONS = [
        "a photo of a cat sitting on a chair",
        "a beautiful landscape with mountains",
        "a close-up of a red flower",
        "people walking in a busy city street",
        "a dog playing in the park",
        "sunset over the ocean with orange sky",
        "a bowl of fresh fruit on a table",
        "children playing in a playground",
        "a vintage car parked on the street",
        "a snowy forest in winter",
    ]

    def __init__(
        self,
        num_samples: int,
        image_buckets: list[tuple[int, int]],
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if not image_buckets:
            raise ValueError("image_buckets must not be empty")

        self.num_samples = num_samples
        self.image_buckets = image_buckets
        self.seed = seed

        rng = random.Random(seed)
        self.bucket_assignments: list[int] = [
            rng.randrange(len(image_buckets)) for _ in range(num_samples)
        ]
        self._caption_indices = [
            rng.randrange(len(self._CAPTIONS)) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> ImageTextPair:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        bucket_idx = self.bucket_assignments[idx]
        H, W = self.image_buckets[bucket_idx]
        pixel_values = torch.rand(3, H, W, generator=rng)
        caption = self._CAPTIONS[self._caption_indices[idx]]
        return ImageTextPair(pixel_values=pixel_values, caption=caption, bucket_idx=bucket_idx)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index_from_shards(
    shards_root: str | Path,
    output_path: str | Path,
    shard_pattern: str = "*.tar",
) -> int:
    """Scan tar shards and write a 5-column index TSV.

    Columns: caption  shard_filename  sample_id  width  height

    Returns number of valid samples written.
    """
    shards_root = Path(shards_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(shards_root.glob(shard_pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching '{shard_pattern}' in {shards_root}")

    total = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for shard in shards:
            try:
                with tarfile.open(shard, "r") as tf:
                    members = tf.getmembers()
                    # Group by stem (sample_id)
                    by_stem: dict[str, dict[str, tarfile.TarInfo]] = {}
                    for m in members:
                        stem, ext = os.path.splitext(m.name)
                        by_stem.setdefault(stem, {})[ext.lstrip(".")] = m

                    for stem, parts in sorted(by_stem.items()):
                        if "jpg" not in parts and "jpeg" not in parts and "png" not in parts:
                            continue
                        img_ext = next(e for e in ("jpg", "jpeg", "png") if e in parts)
                        img_member = parts[img_ext]
                        caption = ""
                        if "txt" in parts:
                            try:
                                cap_bytes = tf.extractfile(parts["txt"])
                                if cap_bytes:
                                    caption = cap_bytes.read().decode("utf-8", errors="replace").strip()
                            except Exception:
                                pass

                        # Get image dimensions without full decode
                        try:
                            img_bytes = tf.extractfile(img_member)
                            if img_bytes is None:
                                continue
                            img = Image.open(io.BytesIO(img_bytes.read()))
                            w, h = img.size
                        except Exception:
                            continue

                        words = caption.split()
                        if len(words) < 2:
                            continue

                        writer.writerow([caption, shard.name, stem, w, h])
                        total += 1
            except Exception as exc:
                logger.warning("Skipping corrupt shard %s: %s", shard.name, exc)

    logger.info("Index written: %d samples → %s", total, output_path)
    return total


# ---------------------------------------------------------------------------
# Index cache helpers
# ---------------------------------------------------------------------------

def _cache_path_for(index_path: Path, buckets: list, max_samples) -> Path:
    key = f"{index_path.stat().st_mtime:.0f}-{index_path.stat().st_size}-{buckets}-{max_samples}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return index_path.with_suffix(f".{h}.cache.pkl")


def _load_index(
    index_path: Path,
    shards_root: Path,
    buckets: list,
    min_caption_words: int,
    max_samples,
) -> tuple[list, list]:
    """Parse TSV and return (samples, bucket_assignments). Uses disk cache."""
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB — handles very long captions
    cache = _cache_path_for(index_path, buckets, max_samples)

    if cache.exists():
        try:
            with cache.open("rb") as f:
                data = pickle.load(f)
            logger.info("Index loaded from cache (%d samples) — %s", len(data["samples"]), cache.name)
            return data["samples"], data["bucket_assignments"], data.get("shard_keys", [""] * len(data["samples"]))
        except Exception:
            cache.unlink(missing_ok=True)

    logger.info("Building index cache for %s (first run only)...", index_path.name)
    samples, bucket_assignments, shard_keys = [], [], []
    with index_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if max_samples is not None and len(samples) >= max_samples:
                break
            if len(row) < 5:
                continue
            caption, shard_name, sample_id, w_str, h_str = row[:5]
            if len(caption.split()) < min_caption_words:
                continue
            try:
                w, h = int(w_str), int(h_str)
            except ValueError:
                continue
            if not (shards_root / shard_name).exists():
                continue
            samples.append((caption, shard_name, sample_id))
            bucket_assignments.append(_assign_bucket(w, h, buckets))
            shard_keys.append(shard_name)

    with cache.open("wb") as f:
        pickle.dump({"samples": samples, "bucket_assignments": bucket_assignments, "shard_keys": shard_keys}, f)
    logger.info("Cache saved → %s (%d samples)", cache.name, len(samples))
    return samples, bucket_assignments, shard_keys


# ---------------------------------------------------------------------------
# CC3MDataset  (WebDataset tar shards + index TSV)
# ---------------------------------------------------------------------------

class CC3MDataset(Dataset):
    """Random-access dataset over WebDataset tar shards.

    Parameters
    ----------
    captions_path : path to the 5-column index TSV
                    (caption, shard_filename, sample_id, width, height)
    images_root   : directory containing the .tar shard files
    buckets       : list of (H, W) resolution buckets
    augment       : apply random crop / flip / colour jitter if True
    min_caption_words : skip samples with fewer words in the caption
    """

    def __init__(
        self,
        captions_path: str | Path,
        images_root: str | Path,
        buckets: list[tuple[int, int]],
        augment: bool = True,
        min_caption_words: int = 2,
        max_samples: int | None = None,
    ) -> None:
        self.shards_root = Path(images_root)
        self.buckets = buckets
        self.augment = augment

        # Pre-build per-bucket transforms
        self._transforms = [
            _make_transforms(H, W, augment) for H, W in buckets
        ]

        # Load index
        index_path = Path(captions_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index TSV not found: {index_path}\n"
                f"Run build_index_from_shards() to create it."
            )

        self._samples, self.bucket_assignments, self.shard_keys = _load_index(
            index_path, self.shards_root, buckets, min_caption_words, max_samples
        )

        if not self._samples:
            raise RuntimeError(
                f"No valid samples loaded from {index_path}. "
                f"Check that shards_root={self.shards_root} contains the .tar files."
            )

        # Per-worker tar handle cache — LRU-capped to avoid holding hundreds of open fd's
        self._tar_cache: "OrderedDict[str, tarfile.TarFile]" = OrderedDict()
        self._tar_cache_max = 16

        logger.info(
            "CC3MDataset: %d samples from %s (shards_root=%s)",
            len(self._samples), index_path.name, self.shards_root,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _open_tar(self, shard_name: str) -> tarfile.TarFile:
        """Open and cache a tar file handle (LRU, max self._tar_cache_max per worker)."""
        if shard_name in self._tar_cache:
            self._tar_cache.move_to_end(shard_name)
        else:
            if len(self._tar_cache) >= self._tar_cache_max:
                _, evicted = self._tar_cache.popitem(last=False)
                evicted.close()
            path = self.shards_root / shard_name
            self._tar_cache[shard_name] = tarfile.open(path, "r")
        return self._tar_cache[shard_name]

    def _load_image(self, shard_name: str, sample_id: str) -> Optional["Image.Image"]:
        """Extract and decode one image from a shard."""
        tf = self._open_tar(shard_name)
        for ext in ("jpg", "jpeg", "png", "webp"):
            member_name = f"{sample_id}.{ext}"
            try:
                member = tf.getmember(member_name)
                raw = tf.extractfile(member)
                if raw is None:
                    continue
                img = Image.open(io.BytesIO(raw.read())).convert("RGB")
                return img
            except KeyError:
                continue
            except Exception:
                continue
        return None

    def __getitem__(self, idx: int) -> ImageTextPair:
        caption, shard_name, sample_id = self._samples[idx]
        bucket_idx = self.bucket_assignments[idx]
        transform = self._transforms[bucket_idx]

        # Try current index, then a few random fallbacks on failure
        for attempt in range(8):
            try:
                if attempt > 0:
                    fb_idx = (idx + attempt * 7919) % len(self._samples)
                    caption, shard_name, sample_id = self._samples[fb_idx]
                    bucket_idx = self.bucket_assignments[fb_idx]
                    transform = self._transforms[bucket_idx]

                img = self._load_image(shard_name, sample_id)
                if img is None:
                    continue
                pixel_values = transform(img)
                return ImageTextPair(
                    pixel_values=pixel_values,
                    caption=caption,
                    bucket_idx=bucket_idx,
                )
            except Exception as exc:
                logger.debug("Sample %d attempt %d failed: %s", idx, attempt, exc)

        # Last resort: black image in the first bucket
        H, W = self.buckets[0]
        return ImageTextPair(
            pixel_values=torch.zeros(3, H, W),
            caption=caption,
            bucket_idx=0,
        )


# ---------------------------------------------------------------------------
# CombinedImageTextDataset
# ---------------------------------------------------------------------------

class CombinedImageTextDataset(Dataset):
    """Concatenates multiple CC3MDataset instances into one."""

    def __init__(self, datasets: list[CC3MDataset]) -> None:
        if not datasets:
            raise ValueError("datasets list must not be empty")
        self._datasets = datasets
        self._offsets: list[int] = []
        total = 0
        for ds in datasets:
            self._offsets.append(total)
            total += len(ds)
        self._total = total

        # Merge bucket_assignments and shard_keys
        self.bucket_assignments: list[int] = []
        self.shard_keys: list[str] = []
        for ds in datasets:
            self.bucket_assignments.extend(ds.bucket_assignments)
            self.shard_keys.extend(getattr(ds, "shard_keys", [""] * len(ds)))

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> ImageTextPair:
        # Find which sub-dataset owns this index
        ds_idx = len(self._datasets) - 1
        for i, offset in enumerate(self._offsets):
            if i + 1 < len(self._offsets) and idx < self._offsets[i + 1]:
                ds_idx = i
                break
        local_idx = idx - self._offsets[ds_idx]
        return self._datasets[ds_idx][local_idx]


# ---------------------------------------------------------------------------
# BucketedSampler
# ---------------------------------------------------------------------------

class BucketedSampler(Sampler):
    """Yields index batches where every item in a batch shares the same bucket.

    Guarantees that the DataLoader collate function always receives images of
    the same spatial size, so torch.stack() never fails on a shape mismatch.
    """

    def __init__(
        self,
        bucket_assignments: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        shard_keys: list[str] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.bucket_assignments = bucket_assignments
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0
        self._rank = rank
        self._world_size = world_size
        # shard key per sample index — used for locality-aware batching
        self._shard_keys = shard_keys if shard_keys is not None else [""] * len(bucket_assignments)

        # Group indices by bucket
        buckets: dict[int, list[int]] = {}
        for i, b in enumerate(bucket_assignments):
            buckets.setdefault(b, []).append(i)
        self._buckets = buckets
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        batches: list[list[int]] = []
        for indices in self._buckets.values():
            # Sort by shard so consecutive samples hit the same tar file —
            # avoids opening hundreds of cold shards per batch.
            idxs = sorted(indices, key=lambda i: self._shard_keys[i])
            # Shuffle within shard-sized windows to keep locality while mixing.
            window = max(self.batch_size * 16, 1024)
            for w in range(0, len(idxs), window):
                chunk = idxs[w: w + window]
                if self.shuffle:
                    rng.shuffle(chunk)
                idxs[w: w + window] = chunk
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start: start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        # Shard batches across ranks — each rank gets every world_size-th batch
        if self._world_size > 1:
            batches = batches[self._rank :: self._world_size]
        return batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._batches = self._build_batches()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# FrozenTextEncoder
# ---------------------------------------------------------------------------

class FrozenTextEncoder(nn.Module):
    """Frozen pretrained text encoder + trainable linear projection.

    Supports two backends:
    - HuggingFace CLIP text model (e.g. openai/clip-vit-large-patch14)
    - sentence-transformers (e.g. all-MiniLM-L6-v2)
    Falls back to a hash-based encoder for smoke tests when loading fails.
    """

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        self._fallback = False
        self._backend = None

        # Try CLIP text model first (for openai/clip-vit-large-patch14)
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self._model = CLIPTextModel.from_pretrained(model_name)
            for p in self._model.parameters():
                p.requires_grad_(False)
            raw_dim = self._model.config.hidden_size
            self._backend = "clip"
            logger.info("[FrozenTextEncoder] Loaded CLIP '%s' (dim=%d)", model_name, raw_dim)
        except Exception:
            # Fall back to sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                for p in self._model.parameters():
                    p.requires_grad_(False)
                raw_dim = self._model.get_embedding_dimension()
                self._backend = "st"
                logger.info("[FrozenTextEncoder] Loaded ST '%s' (dim=%d)", model_name, raw_dim)
            except Exception as exc:
                logger.warning(
                    "[FrozenTextEncoder] Using fallback hash encoder "
                    "(could not load '%s': %s)", model_name, exc
                )
                self._model = None
                self._fallback = True
                raw_dim = 128

        self.proj = nn.Linear(raw_dim, output_dim)

    def _encode_fallback(self, captions: list[str]) -> torch.Tensor:
        vecs = []
        for cap in captions:
            digest = hashlib.sha256(cap.encode()).digest()
            floats = [(b / 127.5) - 1.0 for b in digest]
            floats = (floats * 4)[:128]
            vecs.append(torch.tensor(floats, dtype=torch.float32))
        return torch.stack(vecs)

    @torch.no_grad()
    def _encode_clip(self, captions: list[str], device: torch.device) -> torch.Tensor:
        inputs = self._tokenizer(
            captions, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        out = self._model(**inputs)
        # pooled_output is the EOS token embedding (CLS equivalent for CLIP)
        return out.pooler_output.float()

    @torch.no_grad()
    def _encode_st(self, captions: list[str], device: torch.device) -> torch.Tensor:
        embeddings = self._model.encode(
            captions, convert_to_tensor=True, device=str(device),
            show_progress_bar=False, batch_size=len(captions),
        )
        return embeddings.float()

    def forward(self, captions: list[str]) -> torch.Tensor:
        device = self.proj.weight.device
        if self._fallback or self._model is None:
            raw = self._encode_fallback(captions).to(device)
        elif self._backend == "clip":
            raw = self._encode_clip(captions, device)
        else:
            raw = self._encode_st(captions, device)
        return self.proj(raw.clone())  # (B, output_dim)


# ---------------------------------------------------------------------------
# build_collate_fn
# ---------------------------------------------------------------------------

def build_collate_fn(
    pixel_mean: list[float],
    pixel_std: list[float],
) -> Callable:
    """Return a collate function that stacks ImageTextPair items into a batch.

    All items in a batch must share the same bucket_idx (enforced by
    BucketedSampler). The collate applies ImageNet-style normalisation.
    """
    mean = torch.tensor(pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(pixel_std,  dtype=torch.float32).view(3, 1, 1)

    def collate(items: list[ImageTextPair]) -> dict:
        bucket_ids = {item.bucket_idx for item in items}
        if len(bucket_ids) > 1:
            raise ValueError(
                f"Bucket mismatch in batch: found bucket ids {bucket_ids}. "
                "BucketedSampler must group by bucket before collating."
            )
        pixel_values = torch.stack([item.pixel_values for item in items])  # (B, 3, H, W)
        pixel_values = (pixel_values - mean) / std
        return {
            "pixel_values": pixel_values,
            "captions": [item.caption for item in items],
            "bucket_idx": items[0].bucket_idx,
        }

    return collate
