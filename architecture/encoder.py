"""
Keural Vision Encoder — Full Model.

This module is the integration point. It composes:
  - Stem + ATB tokenizer (modules 4-8) → tokens with metadata
  - 12 spatial transformer blocks (module 10)
  - Pool token + level embeddings
  - Output assembly into KeuralEncoderOutput (module 1)

This is the public model class. It accepts a config and produces a
KeuralEncoderOutput per the public contract.

Reference: docs/03_architecture_design.md §2 (high-level forward pass).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_REGION,
    KeuralEncoderOutput,
    NUM_LEVELS,
)
from architecture.spatial.block import RMSNorm, SpatialBlock
from keural_mid_config import KeuralMidConfig as KeuralConfig
from tokenizer.atb_tokenizer import ATBTokenizer


class KeuralVisionEncoder(nn.Module):
    """The full Keural Vision Encoder.

    Forward signature:
        out: KeuralEncoderOutput = encoder(
            pixel_values,           # (B, 3, H, W)
            token_budget=None,      # int or None for default
            step=None,              # for warmup / temperature schedule
        )

    Parameters
    ----------
    config : KeuralConfig
    """

    def __init__(self, config: KeuralConfig) -> None:
        super().__init__()
        self.config = config

        # Stage A: tokenizer (stem + saliency + sampling + token construction)
        self.tokenizer = ATBTokenizer(config)

        # Stage B: level embeddings - learned per-level bias added to tokens.
        # This is one of the three HCT specialization mechanisms (per docs §4.2).
        # Each token gets E_level[level_id] added to its embedding before
        # entering the spatial encoder.
        self.level_embeddings = nn.Parameter(
            torch.zeros(NUM_LEVELS, config.spatial.embed_dim)
        )
        nn.init.trunc_normal_(self.level_embeddings, std=0.02)

        # Stage C: pool token. A learnable embedding prepended to the
        # sequence; its final state is the pooled image vector.
        self.pool_token = nn.Parameter(torch.zeros(1, 1, config.spatial.embed_dim))
        nn.init.trunc_normal_(self.pool_token, std=0.02)
        # The pool token also gets a fixed spatial metadata: full image,
        # treated like a global token for RoPE purposes.
        # We register it as a buffer so it lives on the right device.
        pool_meta = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
        self.register_buffer("pool_metadata", pool_meta, persistent=True)

        # Stage D: spatial transformer blocks
        self.blocks = nn.ModuleList(
            [SpatialBlock(config.spatial) for _ in range(config.spatial.depth)]
        )

        # Stage E: final norm
        self.final_norm = RMSNorm(
            config.spatial.embed_dim, eps=config.spatial.norm_eps
        )

        # Apply HF-style trunc-normal init to all linear/conv weights.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard trunc-normal init for linear/conv layers."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def embed_dim(self) -> int:
        """Public accessor for the model's token dim."""
        return self.config.spatial.embed_dim

    def forward(
        self,
        pixel_values: torch.Tensor,
        token_budget: int | None = None,
        valid_mask: torch.Tensor | None = None,
        return_hierarchy: bool = True,
        step: int | None = None,
        return_saliency_map: bool = False,
    ) -> KeuralEncoderOutput | tuple[KeuralEncoderOutput, torch.Tensor]:
        """Run the full encoder forward pass.

        Parameters
        ----------
        pixel_values
            Shape (B, 3, H, W). H, W must be multiples of 16 (stem stride).
        token_budget
            Total tokens to allocate. If None, uses config.tokenizer.default_token_budget.
        valid_mask
            Optional shape (B, h, w) on the saliency grid — set to 0 for
            padded regions so they don't receive tokens.
        return_hierarchy
            Reserved for future use. Currently always returns the full
            structured output.
        step
            Current training step. Used for the warmup/temperature schedule
            in the tokenizer. None for inference.
        return_saliency_map
            If True, also returns the per-position saliency map (B, 1, h, w).
            Used by the training loop for the saliency-variance regularizer.

        Returns
        -------
        KeuralEncoderOutput, or (KeuralEncoderOutput, saliency_map) when
        return_saliency_map=True.
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must have shape (B, 3, H, W); got {tuple(pixel_values.shape)}"
            )
        B = pixel_values.shape[0]
        D = self.embed_dim
        device = pixel_values.device

        # Resolve token budget.
        if token_budget is None:
            token_budget = self._resolve_default_budget(pixel_values)

        # Stage A: tokenize
        tokenized = self.tokenizer(
            pixel_values=pixel_values,
            token_budget=token_budget,
            valid_mask=valid_mask,
            step=step,
        )
        # tokenized.tokens: (B, N, D) where N = K_g + K_r + K_d
        # tokenized.level_ids: (B, N)
        # tokenized.spatial_metadata: (B, N, 4)

        N = tokenized.tokens.shape[1]

        # Stage B: add level embeddings.
        # level_embeddings: (NUM_LEVELS, D). Index by level_ids: (B, N) -> (B, N, D)
        level_emb = self.level_embeddings[tokenized.level_ids]  # (B, N, D)
        x = tokenized.tokens + level_emb

        # Stage C: prepend pool token. Its metadata is the full-image one.
        pool_tokens = self.pool_token.expand(B, 1, D).to(x.dtype)
        x = torch.cat([pool_tokens, x], dim=1)  # (B, N+1, D)

        pool_meta = self.pool_metadata.unsqueeze(0).expand(B, 1, 4).to(x.dtype)
        full_metadata = torch.cat([pool_meta, tokenized.spatial_metadata], dim=1)

        # Stage D: full attention mask. All N+1 tokens are real (the tokenizer
        # produces real tokens for the entire budget; padding to a max-N for
        # batched buckets happens at the OUTPUT assembly stage, not here).
        block_attn_mask = torch.ones(B, N + 1, device=device, dtype=torch.int64)

        # Run blocks. Each block applies pre-norm + attn + residual + pre-norm + mlp + residual.
        # Optional gradient checkpointing for memory savings during training.
        use_grad_ckpt = (
            self.config.training.gradient_checkpointing
            and self.training
            and x.requires_grad
        )
        for block in self.blocks:
            if use_grad_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, full_metadata, block_attn_mask, use_reentrant=False
                )
            else:
                x = block(x, full_metadata, block_attn_mask)

        x = self.final_norm(x)

        # Stage E: split pool token from the rest.
        pooled = x[:, 0]                  # (B, D)
        seq_tokens = x[:, 1:]              # (B, N, D)

        # Build the final output. Per the contract, attention_mask is all
        # ones for the produced tokens (we always emit exactly token_budget
        # real tokens; padding for variable-length batching is handled by
        # the data pipeline, not here).
        attn_mask = torch.ones(B, N, device=device, dtype=torch.int64)
        token_budget_used = torch.full(
            (B,), token_budget, device=device, dtype=torch.int64
        )

        result = KeuralEncoderOutput(
            tokens=seq_tokens,
            attention_mask=attn_mask,
            level_ids=tokenized.level_ids,
            spatial_metadata=tokenized.spatial_metadata,
            saliency_scores=tokenized.saliency_scores,
            pooled=pooled,
            token_budget_used=token_budget_used,
        )
        if return_saliency_map:
            return result, tokenized.saliency_map
        return result

    def encode(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple-mode interface — returns just the (B, N, D) tokens.

        Per docs/04_revisions.md §4 (Revision B). Drop-in replacement for
        any code expecting a ViT/CLIP-style tensor output.
        """
        out = self.forward(pixel_values=pixel_values, **kwargs)
        # Mask out padding (for the encoder there is none, but downstream
        # data pipelines may pad — be safe).
        masked = out.tokens * out.attention_mask.unsqueeze(-1).to(out.tokens.dtype)
        return masked

    def _resolve_default_budget(self, pixel_values: torch.Tensor) -> int:
        """Auto-scale token budget by input resolution.

        Per docs/04_revisions.md §3 (Revision A): if no budget is given,
        return config.default_token_budget for inputs at or below 256x256,
        and scale up linearly with pixel count for larger inputs.
        """
        cfg = self.config.tokenizer
        _, _, H, W = pixel_values.shape
        num_pixels = H * W
        if num_pixels <= 256 * 256:
            return cfg.default_token_budget
        scaled = int(round(num_pixels / (256 * 256) * cfg.default_token_budget))
        return min(cfg.max_token_budget, scaled)
