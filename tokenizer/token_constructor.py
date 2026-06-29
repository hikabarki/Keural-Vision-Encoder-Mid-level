"""
Keural Vision Encoder — Token Constructor.

Converts the output of the region sampler (or learned global queries) into
the actual D-dim token vectors that the spatial encoder will process.

Three construction paths:

1. **Global tokens**: K_g learnable queries cross-attend to all stem features.
   Each query produces one global token. This is a tiny Perceiver-style
   block, scoped only to the global level.

2. **Region tokens**: For each sampled region, take a weighted sum of stem
   features using the region's soft-attention map. Pass the result through
   a region-specific MLP.

3. **Detail tokens**: Same mechanism as region, but with its own MLP
   weights so the network can learn level-specific feature transforms.

Reference: docs/03_architecture_design.md §3.7 and §4.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.region_sampler import SampledRegions


class GlobalQueryConstructor(nn.Module):
    """Produces K_g global tokens via cross-attention from learned queries.

    Architecture
    ------------
    A small single-layer cross-attention block. Learnable queries Q of
    shape (K_g_max, D) attend to flattened stem features (B, h*w, D),
    producing (B, K_g, D) global tokens. Only the first K_g queries are
    used at inference; the rest are unused (allowing flexible budget).

    Parameters
    ----------
    embed_dim : int
        Token embedding dim D. Must match stem.out_channels.
    num_heads : int
        Number of attention heads.
    max_global_tokens : int
        Upper bound on K_g. Determines the size of the learned query pool.
        K_g_max = max(budget_floors['global'], ceil(max_token_budget * budget_split['global']))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_global_tokens: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if max_global_tokens < 1:
            raise ValueError(f"max_global_tokens must be >= 1, got {max_global_tokens}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_global_tokens = max_global_tokens

        # Learnable query pool. Trunc-normal init around zero — queries
        # specialize during training.
        self.queries = nn.Parameter(torch.empty(max_global_tokens, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Cross-attention projection layers. Q from the learned queries,
        # K and V from the stem features.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Output norm for stable training when these tokens are appended
        # to the spatial encoder input.
        self.out_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self,
        stem_features: torch.Tensor,
        K_g: int,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute global tokens.

        Parameters
        ----------
        stem_features
            Shape (B, D, h, w). Stem feature map.
        K_g
            Number of global tokens to produce. Must be <= max_global_tokens.
        valid_mask
            Optional shape (B, h, w) or (B, 1, h, w). Positions with
            mask=0 are masked out of the attention key/value pool. Used
            so global tokens never attend to padded regions.

        Returns
        -------
        Tensor of shape (B, K_g, D) — the global tokens.
        """
        if K_g > self.max_global_tokens:
            raise ValueError(
                f"Requested K_g={K_g} exceeds max_global_tokens={self.max_global_tokens}"
            )
        if K_g <= 0:
            raise ValueError(f"K_g must be > 0, got {K_g}")

        B, D, h, w = stem_features.shape
        if D != self.embed_dim:
            raise ValueError(
                f"stem_features have channel dim {D}, expected {self.embed_dim}"
            )

        # Flatten spatial: (B, h*w, D)
        kv_input = stem_features.flatten(2).transpose(1, 2)  # (B, h*w, D)

        # Take the first K_g queries from the pool, broadcast over batch.
        # (1, K_g, D) -> (B, K_g, D)
        q_input = self.queries[:K_g].unsqueeze(0).expand(B, K_g, D)

        # Project Q, K, V
        q = self.q_proj(q_input)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Reshape for multi-head: (B, num_heads, seq, head_dim)
        q = q.view(B, K_g, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, h * w, self.num_heads, self.head_dim).transpose(1, 2)

        # Build attention mask if valid_mask provided. shape (B, num_heads, K_g, h*w).
        attn_mask: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.dim() == 4:
                valid_mask = valid_mask.squeeze(1)
            if valid_mask.shape != (B, h, w):
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} != ({B}, {h}, {w})"
                )
            # (B, h*w) bool: True = valid
            valid_flat = valid_mask.flatten(1).bool()
            # Convert to additive mask: 0 for valid, -inf for invalid.
            # F.scaled_dot_product_attention expects shape broadcastable to
            # (..., K_g, h*w). We use (B, 1, 1, h*w) and broadcast.
            attn_mask = torch.zeros(
                B, 1, 1, h * w, device=stem_features.device, dtype=q.dtype
            )
            attn_mask = attn_mask.masked_fill(~valid_flat[:, None, None, :], float("-inf"))

        # Scaled dot-product attention. PyTorch's built-in is fast and stable.
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # (B, num_heads, K_g, head_dim) -> (B, K_g, D)
        attended = attended.transpose(1, 2).contiguous().view(B, K_g, D)

        out = self.out_proj(attended)
        out = self.out_norm(out)
        return out


class RegionTokenConstructor(nn.Module):
    """Produces region or detail tokens from sampled regions.

    Each token is a weighted sum of stem features using the region's
    soft-attention map, passed through an MLP. Region and Detail levels
    each have their own instance of this class so they learn different
    feature transforms (one of the three HCT specialization mechanisms,
    per docs/03_architecture_design.md §4.2).

    Parameters
    ----------
    embed_dim : int
        Token embedding dim D.
    mlp_ratio : int
        Hidden dim of the MLP = mlp_ratio * embed_dim.
    """

    def __init__(self, embed_dim: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        hidden = mlp_ratio * embed_dim
        # Two-layer MLP with GELU. Compact enough that we can have one per level.
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        # Output norm
        self.out_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self,
        stem_features: torch.Tensor,
        regions: SampledRegions,
    ) -> torch.Tensor:
        """Compute level tokens.

        Parameters
        ----------
        stem_features
            Shape (B, D, h, w). Stem feature map.
        regions
            SampledRegions output from RegionSampler. We use:
            - regions.soft_weights: (B, K, h, w) — for differentiable pooling.

        Returns
        -------
        Tensor of shape (B, K, D) — the level tokens.
        """
        B, D, h, w = stem_features.shape
        K = regions.soft_weights.shape[1]
        if regions.soft_weights.shape != (B, K, h, w):
            raise ValueError(
                f"soft_weights shape {tuple(regions.soft_weights.shape)} "
                f"!= expected ({B}, {K}, {h}, {w})"
            )

        # Differentiable pooling: weighted sum of features per region.
        # einsum("bchw,bkhw->bkc"): for each batch b and region k, sum over (h, w)
        # of feat[b, c, h, w] * weight[b, k, h, w]
        pooled = torch.einsum(
            "bchw,bkhw->bkc",
            stem_features,
            regions.soft_weights.to(stem_features.dtype),
        )
        # pooled: (B, K, D)

        # Per-level MLP transform with residual
        x = pooled + self.mlp(pooled)
        x = self.out_norm(x)
        return x


def compute_spatial_metadata(
    coords: torch.Tensor,
    grid_h: int,
    grid_w: int,
    receptive_field: int,
) -> torch.Tensor:
    """Compute (cx, cy, w, h) spatial metadata for sampled regions.

    All four values are normalized to [0, 1] relative to the original
    image dimensions, regardless of grid size — so the same metadata
    is comparable across resolutions and across hierarchy levels.

    Parameters
    ----------
    coords : Tensor of shape (B, K, 2). int64. (y, x) grid centers.
    grid_h, grid_w : int. Saliency-grid dimensions.
    receptive_field : int. Approximate cell-extent for this level.

    Returns
    -------
    Tensor of shape (B, K, 4). [cx, cy, w, h] in [0, 1].
    """
    B, K, _ = coords.shape
    device = coords.device
    cy_norm = (coords[..., 0].float() + 0.5) / grid_h
    cx_norm = (coords[..., 1].float() + 0.5) / grid_w
    # Width and height as fractions of the image. The receptive field is
    # in grid cells, so divide by grid dims to get image fraction.
    w_norm = torch.full((B, K), float(receptive_field) / grid_w, device=device)
    h_norm = torch.full((B, K), float(receptive_field) / grid_h, device=device)
    # Stack as (cx, cy, w, h) — note the order is x first, matching the
    # public contract documentation.
    return torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=-1)


def global_token_metadata(
    batch_size: int,
    K_g: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Spatial metadata for global tokens.

    Global tokens summarize the whole image, so their metadata is the
    full image: cx=cy=0.5, w=h=1.0.

    Returns
    -------
    Tensor of shape (B, K_g, 4).
    """
    meta = torch.zeros(batch_size, K_g, 4, device=device, dtype=dtype)
    meta[..., 0] = 0.5  # cx
    meta[..., 1] = 0.5  # cy
    meta[..., 2] = 1.0  # w
    meta[..., 3] = 1.0  # h
    return meta
