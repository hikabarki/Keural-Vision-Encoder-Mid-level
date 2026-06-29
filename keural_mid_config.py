"""
Keural Mid-Level Vision Encoder — Configuration system.

Identical structure to poc/keural_config.py but with mid-level defaults:
  - embed_dim: 768, depth: 24 (2× POC)
  - token_budget: 512 default, 2048 max
  - text_encoder: CLIP ViT-L/14 (768-dim)
  - knowledge_distillation: enabled (SigLIP-SO400M + InternViT-300M)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StemConfig:
    channels: list[int] = field(default_factory=lambda: [128, 256, 512, 768])
    kernel_size: int = 3
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if len(self.channels) != 4:
            raise ValueError(f"StemConfig.channels must have length 4 (got {len(self.channels)})")
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"StemConfig.activation must be gelu/relu/silu, got {self.activation}")


@dataclass
class SaliencyConfig:
    hidden_channels: int = 192


@dataclass
class TokenizerConfig:
    default_token_budget: int = 512
    max_token_budget: int = 2048
    budget_split: dict[str, float] = field(
        default_factory=lambda: {"global": 0.05, "region": 0.25, "detail": 0.70}
    )
    budget_floors: dict[str, int] = field(
        default_factory=lambda: {"global": 8, "region": 16, "detail": 0}
    )
    saliency_smooth_kernel: dict[str, int] = field(
        default_factory=lambda: {"region": 7, "detail": 3}
    )
    nms_radius: dict[str, int] = field(
        default_factory=lambda: {"region": 3, "detail": 1}
    )
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.05
    temperature_anneal_steps: int = 50000
    warmup_steps_fixed_grid: int = 10000
    roi_align_size: int = 4

    def __post_init__(self) -> None:
        total = sum(self.budget_split.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"budget_split must sum to 1.0 (got {total:.4f})")
        required = {"global", "region", "detail"}
        if set(self.budget_split.keys()) != required:
            raise ValueError(f"budget_split must have keys {required}")
        if set(self.budget_floors.keys()) != required:
            raise ValueError(f"budget_floors must have keys {required}")
        if self.default_token_budget > self.max_token_budget:
            raise ValueError(
                f"default_token_budget ({self.default_token_budget}) > max_token_budget ({self.max_token_budget})"
            )
        if self.gumbel_temperature_end >= self.gumbel_temperature_start:
            raise ValueError("Gumbel temperature must anneal down (start > end)")


@dataclass
class SpatialConfig:
    embed_dim: int = 768
    depth: int = 24
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: int = 4
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    use_pool_token: bool = True

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError(
                f"num_heads ({self.num_heads}) * head_dim ({self.head_dim}) "
                f"= {self.num_heads * self.head_dim}, must equal embed_dim ({self.embed_dim})"
            )
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")


@dataclass
class LossConfig:
    primary: str = "sigmoid_siglip"
    hct_alpha: float = 0.5
    saliency_beta: float = 0.005
    saliency_lambda_smooth: float = 4.0
    lambda_hct: float = 1.0
    lambda_sal: float = 0.5
    lambda_distill: float = 0.5
    learnable_temperature_init: float = 2.30
    learnable_bias_init: float = -10.0

    def __post_init__(self) -> None:
        if self.primary not in {"sigmoid_siglip", "infonce_clip"}:
            raise ValueError(f"primary must be sigmoid_siglip or infonce_clip, got {self.primary}")


@dataclass
class DistillationConfig:
    """Knowledge distillation settings.

    Two frozen teacher models guide the encoder toward rich representations:
    - SigLIP-SO400M: strong natural-image teacher (1152-dim → projected to 768)
    - InternViT-300M: document/text-in-image teacher (768-dim, no projection needed)

    Teachers are discarded after training — only the student (keural-mid) is deployed.
    """
    enabled: bool = True
    siglip_model: str = "google/siglip-so400m-patch14-384"
    siglip_embed_dim: int = 1152
    internvit_model: str = "OpenGVLab/InternViT-300M-448px"
    internvit_embed_dim: int = 768
    lambda_siglip: float = 0.5
    lambda_internvit: float = 0.5


@dataclass
class TrainingConfig:
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True
    optimizer: str = "adamw"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.05
    lr: float = 5e-4
    lr_schedule: str = "cosine_with_warmup"
    warmup_steps: int = 10000
    max_steps: int = 500000
    batch_size_per_device: int = 128
    gradient_accumulation: int = 4
    log_every: int = 100
    save_every: int = 10000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Invalid precision: {self.precision}")
        if self.optimizer != "adamw":
            raise ValueError(f"Only adamw supported, got {self.optimizer}")
        if self.lr_schedule not in {"cosine_with_warmup", "constant", "linear_with_warmup"}:
            raise ValueError(f"Invalid lr_schedule: {self.lr_schedule}")
        if len(self.betas) != 2:
            raise ValueError(f"betas must have length 2, got {self.betas}")


@dataclass
class DataConfig:
    dataset_name: str = "datacomp+coyo"
    images_root: str = "/home/work/keural-model/datacomp_medium"
    captions_path: str = "/home/work/keural-model/indexes/datacomp_medium_train.tsv"
    image_buckets: list[list[int]] = field(
        default_factory=lambda: [
            [224, 224], [384, 384], [448, 448], [512, 512],
        ]
    )
    pixel_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    pixel_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    max_caption_length: int = 77
    num_workers: int = 8
    text_encoder_name: str = "openai/clip-vit-large-patch14"
    extra_sources: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        for b in self.image_buckets:
            if len(b) != 2:
                raise ValueError(f"Each image_bucket must be [H, W], got {b}")
            if b[0] % 16 != 0 or b[1] % 16 != 0:
                raise ValueError(f"Image bucket {b} must be divisible by 16.")


@dataclass
class KeuralMidConfig:
    """Top-level config for the Keural mid-level vision encoder (183.5M params)."""

    name: str = "keural-mid"
    stem: StemConfig = field(default_factory=StemConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if self.stem.channels[-1] != self.spatial.embed_dim:
            raise ValueError(
                f"stem.channels[-1] ({self.stem.channels[-1]}) must equal "
                f"spatial.embed_dim ({self.spatial.embed_dim})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeuralMidConfig:
        kwargs: dict[str, Any] = {}
        if "name" in data:
            kwargs["name"] = data["name"]
        if "stem" in data:
            kwargs["stem"] = StemConfig(**data["stem"])
        if "saliency" in data:
            kwargs["saliency"] = SaliencyConfig(**data["saliency"])
        if "tokenizer" in data:
            kwargs["tokenizer"] = TokenizerConfig(**data["tokenizer"])
        if "spatial" in data:
            kwargs["spatial"] = SpatialConfig(**data["spatial"])
        if "loss" in data:
            kwargs["loss"] = LossConfig(**data["loss"])
        if "distillation" in data:
            kwargs["distillation"] = DistillationConfig(**data["distillation"])
        if "training" in data:
            kwargs["training"] = TrainingConfig(**data["training"])
        if "data" in data:
            d = dict(data["data"])
            d.setdefault("extra_sources", [])
            kwargs["data"] = DataConfig(**d)
        return cls(**kwargs)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> KeuralMidConfig:
        with Path(path).open("r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls.from_dict(data)
