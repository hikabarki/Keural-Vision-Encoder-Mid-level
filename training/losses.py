"""
Keural Mid-Level Vision Encoder — Training Losses.

Extends the POC loss with knowledge distillation:

  L_total = L_primary + λ_hct * L_HCT + λ_sal * L_saliency + λ_distill * L_distill

L_distill = λ_siglip * (1 - cos_sim(keural_pooled, siglip_pooled))
          + λ_internvit * (1 - cos_sim(keural_pooled, internvit_pooled))

Both teachers are frozen. Only their pooled vectors are used — the full
token sequences are discarded to save memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment.output_contract import LEVEL_GLOBAL, KeuralEncoderOutput
from keural_mid_config import LossConfig
from training.distributed import gather_with_grad, get_rank, get_world_size, is_dist


@dataclass
class LossOutput:
    total: torch.Tensor
    primary: torch.Tensor
    hct: torch.Tensor
    saliency: torch.Tensor
    distill: torch.Tensor
    components: dict[str, float]


def sigmoid_siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    image_offset: int = 0,
) -> torch.Tensor:
    """SigLIP sigmoid loss.

    ``image_embeds`` are this rank's local images (rows). ``text_embeds`` may be
    the *gathered* texts from all ranks (columns) when training distributed.
    ``image_offset`` is the column index at which this rank's local images live
    inside the gathered text block, so the positive (matching) pairs land on the
    shifted diagonal ``targets[i, image_offset + i] = +1``. With the default
    offset 0 and equal row/col counts this reduces to the original square loss.
    """
    R = image_embeds.shape[0]      # local rows (images on this rank)
    C = text_embeds.shape[0]       # columns (texts; gathered when distributed)
    device = image_embeds.device
    similarity = image_embeds @ text_embeds.T          # (R, C)
    logits = similarity * logit_scale.exp() + logit_bias
    targets = -torch.ones(R, C, device=device, dtype=logits.dtype)
    # Positive pair for local image i is the global text at (image_offset + i).
    rows = torch.arange(R, device=device)
    cols = rows + image_offset
    targets[rows, cols] = 1.0
    loss = F.softplus(-targets * logits)
    # Normalise by local rows (SigLIP convention: sum over pairs / batch).
    return loss.sum() / R


def hct_global_text_loss(
    encoder_output: KeuralEncoderOutput,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    gather: bool = False,
    image_offset: int = 0,
) -> torch.Tensor:
    global_mask = (encoder_output.level_ids == LEVEL_GLOBAL) & encoder_output.attention_mask.bool()
    masked_tokens = encoder_output.tokens * global_mask.unsqueeze(-1).to(encoder_output.tokens.dtype)
    global_sum = masked_tokens.sum(dim=1)
    counts = global_mask.sum(dim=1, keepdim=True).clamp(min=1).to(global_sum.dtype)
    global_pooled = global_sum / counts
    image_embeds = F.normalize(global_pooled, dim=-1)
    text_normed = F.normalize(text_embeds, dim=-1)
    if gather:
        # Local images vs. all-rank texts (matching the primary loss).
        text_normed = gather_with_grad(text_normed)
    return sigmoid_siglip_loss(image_embeds, text_normed, logit_scale, logit_bias, image_offset)


def saliency_variance_regularizer(
    saliency_map: torch.Tensor,
    eps: float = 1e-3,
    lambda_smooth: float = 4.0,
) -> torch.Tensor:
    flat = saliency_map.flatten(1)
    variance = flat.var(dim=1, unbiased=False)
    anti_collapse = (1.0 / (variance + eps)).mean()
    dx = saliency_map[:, :, :, 1:] - saliency_map[:, :, :, :-1]
    dy = saliency_map[:, :, 1:, :] - saliency_map[:, :, :-1, :]
    tv = dx.pow(2).mean() + dy.pow(2).mean()
    return anti_collapse + lambda_smooth * tv


def distillation_loss(
    keural_pooled: torch.Tensor,
    teacher_siglip: torch.Tensor | None,
    teacher_internvit: torch.Tensor | None,
    lambda_siglip: float = 0.5,
    lambda_internvit: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Cosine-distance distillation from two frozen teachers.

    For each teacher whose pooled vector is provided, computes
    mean(1 - cosine_similarity(keural, teacher)) and weights by lambda.

    Returns (total_loss, detail_dict) where detail_dict has per-teacher metrics.
    """
    device = keural_pooled.device
    loss = torch.tensor(0.0, device=device, dtype=keural_pooled.dtype)
    details = {}

    keural_norm = F.normalize(keural_pooled, dim=-1)

    if teacher_siglip is not None:
        siglip_norm = F.normalize(teacher_siglip.to(keural_pooled.dtype), dim=-1)
        cos_sim = (keural_norm * siglip_norm).sum(dim=-1)
        details["cos_sim_siglip"] = cos_sim.mean().item()
        details["distill_siglip"] = (lambda_siglip * (1.0 - cos_sim).mean()).item()
        loss = loss + lambda_siglip * (1.0 - cos_sim).mean()

    if teacher_internvit is not None:
        internvit_norm = F.normalize(teacher_internvit.to(keural_pooled.dtype), dim=-1)
        cos_sim = (keural_norm * internvit_norm).sum(dim=-1)
        details["cos_sim_internvit"] = cos_sim.mean().item()
        details["distill_internvit"] = (lambda_internvit * (1.0 - cos_sim).mean()).item()
        loss = loss + lambda_internvit * (1.0 - cos_sim).mean()

    return loss, details


class KeuralMidLoss(nn.Module):
    """Combined training loss for keural-mid.

    Owns learnable temperature/bias and an optional projection from
    SigLIP's 1152-dim pooled vector down to 768-dim for distillation.
    """

    def __init__(self, config: LossConfig, siglip_embed_dim: int = 1152,
                 internvit_embed_dim: int = 1024, embed_dim: int = 768,
                 gather_distributed: bool = True) -> None:
        super().__init__()
        self.config = config
        # When True and torch.distributed is initialised, the contrastive losses
        # gather text embeddings from all ranks so every local image is scored
        # against the global batch of negatives (see training/distributed.py).
        self.gather_distributed = gather_distributed

        self.logit_scale = nn.Parameter(
            torch.tensor(config.learnable_temperature_init, dtype=torch.float32)
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(config.learnable_bias_init, dtype=torch.float32)
        )

        # SigLIP outputs 1152-dim; project to match keural-mid's 768-dim before cosine distillation.
        if siglip_embed_dim != embed_dim:
            self.siglip_proj = nn.Linear(siglip_embed_dim, embed_dim, bias=False)
        else:
            self.siglip_proj = nn.Identity()

        # InternViT outputs 1024-dim; project to match keural-mid's 768-dim.
        if internvit_embed_dim != embed_dim:
            self.internvit_proj = nn.Linear(internvit_embed_dim, embed_dim, bias=False)
        else:
            self.internvit_proj = nn.Identity()

    def forward(
        self,
        encoder_output: KeuralEncoderOutput,
        text_embeds: torch.Tensor,
        saliency_map: torch.Tensor | None = None,
        teacher_siglip_pooled: torch.Tensor | None = None,
        teacher_internvit_pooled: torch.Tensor | None = None,
    ) -> LossOutput:
        if encoder_output.pooled.shape[0] != text_embeds.shape[0]:
            raise ValueError(
                f"Batch size mismatch: encoder {encoder_output.pooled.shape[0]} vs text {text_embeds.shape[0]}"
            )
        if encoder_output.pooled.shape[-1] != text_embeds.shape[-1]:
            raise ValueError(
                f"Embedding dim mismatch: pooled D={encoder_output.pooled.shape[-1]} vs text D={text_embeds.shape[-1]}"
            )

        image_embeds = F.normalize(encoder_output.pooled, dim=-1)
        text_normed = F.normalize(text_embeds, dim=-1)

        # Distributed contrastive: gather texts across ranks so each local image
        # is contrasted against the full global batch of texts. The matching
        # (positive) text for local row i sits at column (image_offset + i) of
        # the gathered block, where image_offset = rank * local_batch.
        gather = self.gather_distributed and is_dist() and get_world_size() > 1
        if gather:
            local_batch = image_embeds.shape[0]
            image_offset = get_rank() * local_batch
            text_primary = gather_with_grad(text_normed)
        else:
            image_offset = 0
            text_primary = text_normed

        primary = sigmoid_siglip_loss(
            image_embeds, text_primary, self.logit_scale, self.logit_bias, image_offset
        )
        hct = hct_global_text_loss(
            encoder_output, text_embeds, self.logit_scale, self.logit_bias,
            gather=gather, image_offset=image_offset,
        )

        if saliency_map is not None:
            saliency = saliency_variance_regularizer(saliency_map, lambda_smooth=self.config.saliency_lambda_smooth)
        else:
            saliency = torch.tensor(0.0, device=primary.device, dtype=primary.dtype)

        # Project teacher pooled vectors before distillation
        siglip_proj = None
        if teacher_siglip_pooled is not None:
            siglip_proj = self.siglip_proj(teacher_siglip_pooled.to(encoder_output.pooled.dtype))

        internvit_proj = None
        if teacher_internvit_pooled is not None:
            internvit_proj = self.internvit_proj(teacher_internvit_pooled.to(encoder_output.pooled.dtype))

        distill, distill_details = distillation_loss(
            encoder_output.pooled,
            teacher_siglip=siglip_proj,
            teacher_internvit=internvit_proj,
            lambda_siglip=self.config.lambda_distill * 0.5,
            lambda_internvit=self.config.lambda_distill * 0.5,
        )

        total = (
            primary
            + self.config.lambda_hct * hct
            + self.config.lambda_sal * self.config.saliency_beta * saliency
            + distill
        )

        # Token budget utilization
        budget_used = encoder_output.token_budget_used.float().mean().item()

        components = {
            "loss/total": total.detach().item(),
            "loss/primary": primary.detach().item(),
            "loss/hct": hct.detach().item(),
            "loss/saliency": saliency.detach().item(),
            "loss/distill": distill.detach().item(),
            "loss/logit_scale": self.logit_scale.detach().item(),
            "loss/logit_bias": self.logit_bias.detach().item(),
            "token_budget_used": budget_used,
            **distill_details,
        }

        return LossOutput(
            total=total, primary=primary, hct=hct,
            saliency=saliency, distill=distill, components=components,
        )
