"""
Frozen teacher model wrappers for knowledge distillation.

Loads SigLIP-SO400M and InternViT-300M in eval mode. Both remain frozen
for the entire training run. Only their CLS/pooled vectors are used.

Usage:
    teachers = DistillationTeachers(config.distillation, device)
    with torch.no_grad():
        siglip_pooled, internvit_pooled = teachers(pixel_values)
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from PIL import Image

from keural_mid_config import DistillationConfig


class SigLIPTeacher(nn.Module):
    """Frozen SigLIP-SO400M image encoder. Returns pooled CLS vector (B, 1152)."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import SiglipVisionModel, SiglipProcessor
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(torch.bfloat16)  # frozen teacher; bf16 halves inference cost (matches InternViT)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, 3, H, W) already preprocessed by keural pipeline.
        # SigLIP expects its own preprocessing — we do a forward pass directly
        # on the vision model to get the CLS pooled output.
        out = self.model(pixel_values=pixel_values.to(torch.bfloat16))
        # out.pooler_output: (B, 1152), cast back for the fp32 loss path
        return out.pooler_output.float()


class InternViTTeacher(nn.Module):
    """Frozen InternViT-300M image encoder. Returns CLS token (B, 1024)."""

    EXPECTED_SIZE = 448  # InternViT-300M native resolution

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(torch.bfloat16)  # flash attention requires bf16/fp16

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Resize to 448×448 if input is a different resolution
        _, _, H, W = pixel_values.shape
        if H != self.EXPECTED_SIZE or W != self.EXPECTED_SIZE:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(self.EXPECTED_SIZE, self.EXPECTED_SIZE),
                mode="bicubic", align_corners=False,
            )
        # InternViT flash attention requires bfloat16/float16
        pixel_values = pixel_values.to(torch.bfloat16)
        out = self.model(pixel_values=pixel_values)
        # last_hidden_state[:, 0] is the CLS token: (B, 1024) for InternViT-300M
        return out.last_hidden_state[:, 0].float()


class DistillationTeachers(nn.Module):
    """Container for both frozen teacher models.

    Handles lazy loading — if a teacher fails to load (e.g. not yet
    downloaded), the distillation for that teacher is silently skipped.

    Parameters
    ----------
    config : DistillationConfig
    device : torch.device
    """

    def __init__(self, config: DistillationConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.siglip: SigLIPTeacher | None = None
        self.internvit: InternViTTeacher | None = None

        if not config.enabled:
            print("[distill] Knowledge distillation disabled in config.")
            return

        print(f"[distill] Loading SigLIP teacher: {config.siglip_model}")
        try:
            self.siglip = SigLIPTeacher(config.siglip_model).to(device)
            print(f"[distill] SigLIP loaded ({config.siglip_embed_dim}-dim).")
        except Exception as e:
            warnings.warn(
                f"[distill] Failed to load SigLIP teacher ({config.siglip_model}): {e}. "
                "Training will continue WITHOUT SigLIP distillation — loss will be incomplete.",
                RuntimeWarning,
                stacklevel=2,
            )

        print(f"[distill] Loading InternViT teacher: {config.internvit_model}")
        try:
            self.internvit = InternViTTeacher(config.internvit_model).to(device)
            print(f"[distill] InternViT loaded ({config.internvit_embed_dim}-dim).")
        except Exception as e:
            warnings.warn(
                f"[distill] Failed to load InternViT teacher ({config.internvit_model}): {e}. "
                "Training will continue WITHOUT InternViT distillation — loss will be incomplete.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.siglip is None and self.internvit is None:
            raise RuntimeError(
                "[distill] Knowledge distillation is enabled but ALL teachers failed to load. "
                "Fix the model paths or set distillation.enabled=False in config."
            )

    @torch.no_grad()
    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (siglip_pooled, internvit_pooled). Either may be None if that teacher failed to load."""
        siglip_out = None
        internvit_out = None

        if self.siglip is not None:
            try:
                siglip_out = self.siglip(pixel_values)
            except Exception as e:
                print(f"[distill] SigLIP forward failed: {e}")

        if self.internvit is not None:
            try:
                internvit_out = self.internvit(pixel_values)
            except Exception as e:
                if not getattr(self, '_internvit_warned', False):
                    print(f"[distill] InternViT forward failed (will suppress repeats): {type(e).__name__}: {e}")
                    self._internvit_warned = True

        return siglip_out, internvit_out
