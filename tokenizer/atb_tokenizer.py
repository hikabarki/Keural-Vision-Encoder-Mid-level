"""
Keural Vision Encoder — ATB Tokenizer (the headline novelty).

This module ties modules 4–7 together. Image goes in, hierarchical tokens
with full contract metadata come out:

    image (B, 3, H, W)
        |
        v
    [ConvStem]                          → stem features (B, D, h, w)
        |
        v
    [SaliencyPredictor]                 → saliency map (B, 1, h, w)
        |
        v
    [Budget Splitter]                   → (K_g, K_r, K_d) given runtime budget
        |
        v
    [Region Sampler] x 2 (region/detail)→ SampledRegions per level
    [Global Query Constructor]          → K_g global tokens
        |
        v
    [Token Constructors] x 3 levels     → per-level token vectors
        |
        v
    Concatenate + spatial metadata + level_ids + saliency_scores
        |
        v
    Tokens ready for the spatial encoder

Reference: docs/03_architecture_design.md §3 (ATB) and §4 (HCT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from alignment.output_contract import LEVEL_DETAIL, LEVEL_GLOBAL, LEVEL_REGION
from keural_mid_config import KeuralMidConfig as KeuralConfig
from tokenizer.region_sampler import (
    RegionSampler,
    SampledRegions,
    deterministic_grid_regions,
)
from tokenizer.saliency import SaliencyPredictor
from tokenizer.stem import ConvStem
from tokenizer.token_constructor import (
    GlobalQueryConstructor,
    RegionTokenConstructor,
    compute_spatial_metadata,
    global_token_metadata,
)


@dataclass
class TokenizedImage:
    """Output of ATBTokenizer (pre-spatial-encoder).

    Attributes
    ----------
    tokens : Tensor of shape (B, N, D) where N = K_g + K_r + K_d.
    level_ids : Tensor of shape (B, N), int64. Values in {0, 1, 2}.
    spatial_metadata : Tensor of shape (B, N, 4). Normalized cx, cy, w, h.
    saliency_scores : Tensor of shape (B, N). Per-token confidence in [0, 1].
    K_g, K_r, K_d : ints, the per-level token counts.
    saliency_map : Tensor of shape (B, 1, h, w). The full predicted saliency
        map. Useful for the saliency-variance regularizer in the loss.
    """

    tokens: torch.Tensor
    level_ids: torch.Tensor
    spatial_metadata: torch.Tensor
    saliency_scores: torch.Tensor
    K_g: int
    K_r: int
    K_d: int
    saliency_map: torch.Tensor


def split_budget(
    token_budget: int,
    split: dict[str, float],
    floors: dict[str, int],
) -> tuple[int, int, int]:
    """Allocate token budget across the three hierarchy levels.

    Parameters
    ----------
    token_budget : int
        Total tokens to allocate.
    split : dict
        Ratios for {global, region, detail}. Must sum to ~1.
    floors : dict
        Minimum tokens per level. Floors are applied first; the remainder
        is split by ratio.

    Returns
    -------
    (K_g, K_r, K_d) : ints. Sum to token_budget.
    """
    K_g = max(floors["global"], math.ceil(token_budget * split["global"]))
    K_r = max(floors["region"], math.ceil(token_budget * split["region"]))
    # Make sure the partial sum doesn't exceed the budget.
    K_g = min(K_g, token_budget)
    K_r = min(K_r, token_budget - K_g)
    K_d = token_budget - K_g - K_r
    # Apply detail floor if room (the detail level has a floor of 0 by
    # default — set in TokenizerConfig).
    if K_d < floors["detail"] and K_g + K_r + floors["detail"] <= token_budget:
        K_d = floors["detail"]
    if K_d < 0:
        K_d = 0
    return K_g, K_r, K_d


class ATBTokenizer(nn.Module):
    """Adaptive Token Budget tokenizer with hierarchical output.

    Parameters
    ----------
    config : KeuralConfig
        Full Keural config. We use stem, saliency, tokenizer, and spatial
        sub-configs.

    Notes on the warmup curriculum
    ------------------------------
    The first `tokenizer.warmup_steps_fixed_grid` steps use deterministic
    grid sampling instead of the learned allocator. This stabilizes
    spatial-encoder gradients while the saliency predictor is random.
    After warmup, the allocator turns on with high Gumbel temperature,
    which anneals to a low-temperature regime.

    The training step counter is passed in via `step` argument to
    `forward()`. Eval-mode forward (training=False or step=None) always
    uses the learned allocator without Gumbel noise.
    """

    def __init__(self, config: KeuralConfig) -> None:
        super().__init__()
        self.config = config

        # Sub-modules
        self.stem = ConvStem(config.stem)
        self.saliency = SaliencyPredictor(
            config.saliency, in_channels=self.stem.out_channels
        )

        # Compute max global tokens for the learned query pool.
        # Upper bound: ceil(max_token_budget * split_global).
        ts = config.tokenizer
        max_global = max(
            ts.budget_floors["global"],
            math.ceil(ts.max_token_budget * ts.budget_split["global"]),
        )

        # Use a small fraction of spatial heads for global cross-attention,
        # capped to fit head_dim. Use spatial.num_heads if it cleanly divides
        # embed_dim; otherwise default to 4.
        if config.spatial.embed_dim % config.spatial.num_heads == 0:
            global_heads = config.spatial.num_heads
        else:
            global_heads = 4

        self.global_constructor = GlobalQueryConstructor(
            embed_dim=config.spatial.embed_dim,
            num_heads=global_heads,
            max_global_tokens=max_global,
        )

        # Two level-specific region constructors. Separate weights → HCT
        # specialization mechanism (per docs §4.2).
        self.region_constructor = RegionTokenConstructor(
            embed_dim=config.spatial.embed_dim, mlp_ratio=2
        )
        self.detail_constructor = RegionTokenConstructor(
            embed_dim=config.spatial.embed_dim, mlp_ratio=2
        )

        # Two level-specific samplers. Region uses larger smoothing kernel
        # and bigger NMS radius — so it samples object-scale regions.
        # Detail uses tighter receptive field for fine-grained features.
        self.region_sampler = RegionSampler(
            smooth_kernel=ts.saliency_smooth_kernel["region"],
            nms_radius=ts.nms_radius["region"],
            soft_weight_sigma=ts.saliency_smooth_kernel["region"] / 4.0,
        )
        self.detail_sampler = RegionSampler(
            smooth_kernel=ts.saliency_smooth_kernel["detail"],
            nms_radius=ts.nms_radius["detail"],
            soft_weight_sigma=ts.saliency_smooth_kernel["detail"] / 4.0,
        )

        # Receptive field sizes for spatial metadata (per docs §3.5).
        self._region_receptive_field = ts.saliency_smooth_kernel["region"]
        self._detail_receptive_field = ts.saliency_smooth_kernel["detail"]

    def _resolve_temperature(self, step: int | None) -> float:
        """Return the current Gumbel temperature given training step."""
        ts = self.config.tokenizer
        if step is None:
            return 0.0  # Eval mode — no noise
        # During warmup (fixed grid), temperature is irrelevant.
        if step < ts.warmup_steps_fixed_grid:
            return 0.0
        # After warmup, anneal linearly from start → end.
        progress = (step - ts.warmup_steps_fixed_grid) / max(
            1, ts.temperature_anneal_steps
        )
        progress = max(0.0, min(1.0, progress))
        return (
            ts.gumbel_temperature_start
            + progress * (ts.gumbel_temperature_end - ts.gumbel_temperature_start)
        )

    def _use_warmup(self, step: int | None) -> bool:
        """Whether to use deterministic grid sampling instead of learned allocator."""
        if step is None:
            return False
        return step < self.config.tokenizer.warmup_steps_fixed_grid

    def forward(
        self,
        pixel_values: torch.Tensor,
        token_budget: int,
        valid_mask: torch.Tensor | None = None,
        step: int | None = None,
    ) -> TokenizedImage:
        """Tokenize an image batch.

        Parameters
        ----------
        pixel_values
            Shape (B, 3, H, W). H and W must be multiples of stem stride (16).
        token_budget
            Total tokens to allocate. Must be in [4, max_token_budget].
        valid_mask
            Optional shape (B, h, w) — for padded inputs. Forces saliency
            to 0 in padded regions and prevents cross-attention to them.
        step
            Current training step. None for inference. Controls the warmup
            curriculum and Gumbel temperature schedule.

        Returns
        -------
        TokenizedImage with all hierarchical tokens and metadata.
        """
        ts = self.config.tokenizer
        if token_budget > ts.max_token_budget:
            raise ValueError(
                f"token_budget ({token_budget}) > max_token_budget ({ts.max_token_budget})"
            )
        if token_budget < (
            ts.budget_floors["global"] + ts.budget_floors["region"]
        ):
            raise ValueError(
                f"token_budget ({token_budget}) is below the sum of floors "
                f"({ts.budget_floors['global']} + {ts.budget_floors['region']})"
            )

        B = pixel_values.shape[0]
        device = pixel_values.device

        # Step 1: stem features. (B, D, h, w)
        stem_feat = self.stem(pixel_values)
        h, w = stem_feat.shape[-2:]

        # Step 2: saliency map. (B, 1, h, w) in (0, 1).
        sal = self.saliency(stem_feat, valid_mask=valid_mask)

        # Step 3: split the budget.
        K_g, K_r, K_d = split_budget(token_budget, ts.budget_split, ts.budget_floors)

        # Step 4: produce global tokens via cross-attention.
        global_tokens = self.global_constructor(
            stem_features=stem_feat, K_g=K_g, valid_mask=valid_mask
        )  # (B, K_g, D)

        # Step 5: sample region/detail regions. During warmup, use grid fallback.
        use_warmup = self._use_warmup(step)
        gumbel_t = self._resolve_temperature(step)

        if use_warmup:
            region_picks = deterministic_grid_regions(h, w, K_r, B, device=device)
            detail_picks = deterministic_grid_regions(h, w, K_d, B, device=device) if K_d > 0 else None
        else:
            region_picks = self.region_sampler(
                sal, K=K_r, gumbel_temperature=gumbel_t, training=self.training
            )
            if K_d > 0:
                detail_picks = self.detail_sampler(
                    sal, K=K_d, gumbel_temperature=gumbel_t, training=self.training
                )
            else:
                detail_picks = None

        # Step 6: construct level tokens.
        region_tokens = self.region_constructor(stem_feat, region_picks)  # (B, K_r, D)

        if detail_picks is not None and K_d > 0:
            detail_tokens = self.detail_constructor(stem_feat, detail_picks)  # (B, K_d, D)
        else:
            detail_tokens = pixel_values.new_zeros(B, 0, self.config.spatial.embed_dim)

        # Step 7: assemble metadata. We follow the level order [global, region, detail].
        global_meta = global_token_metadata(B, K_g, device=device, dtype=stem_feat.dtype)
        region_meta = compute_spatial_metadata(
            region_picks.coords, h, w, self._region_receptive_field
        ).to(stem_feat.dtype)

        # Saliency scores: 1.0 for global tokens (they summarize the whole image),
        # actual scores for region/detail. We clamp into (0, 1).
        global_saliency = stem_feat.new_full((B, K_g), 1.0)
        region_saliency = region_picks.scores.clamp(0.0, 1.0).to(stem_feat.dtype)

        # Concatenate.
        if K_d > 0 and detail_picks is not None:
            detail_meta = compute_spatial_metadata(
                detail_picks.coords, h, w, self._detail_receptive_field
            ).to(stem_feat.dtype)
            detail_saliency = detail_picks.scores.clamp(0.0, 1.0).to(stem_feat.dtype)
            tokens = torch.cat([global_tokens, region_tokens, detail_tokens], dim=1)
            spatial_meta = torch.cat([global_meta, region_meta, detail_meta], dim=1)
            saliency_scores = torch.cat(
                [global_saliency, region_saliency, detail_saliency], dim=1
            )
        else:
            tokens = torch.cat([global_tokens, region_tokens], dim=1)
            spatial_meta = torch.cat([global_meta, region_meta], dim=1)
            saliency_scores = torch.cat([global_saliency, region_saliency], dim=1)

        # Level IDs in the same order as concatenation.
        level_ids = torch.zeros(B, K_g + K_r + K_d, dtype=torch.int64, device=device)
        level_ids[:, :K_g] = LEVEL_GLOBAL
        level_ids[:, K_g : K_g + K_r] = LEVEL_REGION
        level_ids[:, K_g + K_r :] = LEVEL_DETAIL

        return TokenizedImage(
            tokens=tokens,
            level_ids=level_ids,
            spatial_metadata=spatial_meta,
            saliency_scores=saliency_scores,
            K_g=K_g,
            K_r=K_r,
            K_d=K_d,
            saliency_map=sal,
        )
