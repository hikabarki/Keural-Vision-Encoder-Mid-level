"""
Keural Vision Encoder — 2D-RoPE-from-Metadata.

Standard 2D rotary positional embeddings, but indexed by continuous
spatial metadata (cx, cy) instead of grid coordinates. This is what makes
the encoder genuinely resolution-elastic: there is no fixed-size positional
table to interpolate.

Why RoPE and not learned embeddings or ALiBi?
---------------------------------------------
- RoPE handles continuous, variable positions cleanly without a fixed-size table.
- It generalizes across resolutions because positions in [0, 1] are
  comparable regardless of input image size.
- ALiBi has no obvious 2D extension.
- Learned embeddings need a fixed grid.

This is supporting infrastructure, not a headline contribution. We use a
known technique (2D RoPE) applied to a context that needs it (irregular
variable-position tokens).

Reference: docs/03_architecture_design.md §5.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE2DFromMetadata(nn.Module):
    """2D rotary positional embedding indexed by continuous (cx, cy) values.

    Each attention head's dimension is split in half:
      - First half rotates by cx (column position).
      - Second half rotates by cy (row position).

    The frequency schedule follows the standard RoPE formulation:
      theta_i = base ** (-2i / head_dim_per_axis)
    where head_dim_per_axis = head_dim / 2.

    Parameters
    ----------
    head_dim : int
        Per-head attention dim. Must be divisible by 4 (because we split
        head_dim in half for x/y, then each half is split into pairs for
        the 2D rotation).
    base : float, default 10000.0
        Standard RoPE frequency base.
    max_pos : float, default 1.0
        The maximum value of cx/cy. Since spatial_metadata is normalized
        to [0, 1], this is 1.0 — but we expose it for flexibility.
    pos_scale : float, default 100.0
        Scale factor for normalized positions. RoPE was designed for
        integer positions; we multiply normalized [0, 1] coordinates by
        this scale so the rotation magnitudes match the regime RoPE
        works well in. 100.0 means a position of 1.0 gets a rotation
        of 100 * theta_0 = 100 radians at the lowest frequency, which
        is equivalent to roughly a 100-step grid.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_pos: float = 1.0,
        pos_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 4 for 2D RoPE."
            )
        self.head_dim = head_dim
        self.half_dim = head_dim // 2  # Per-axis dim
        self.base = base
        self.max_pos = max_pos
        self.pos_scale = pos_scale

        # Frequencies for one axis: shape (half_dim/2,)
        # theta_i = base ** (-2i / half_dim)
        i = torch.arange(0, self.half_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (i / self.half_dim))
        # Register as buffer so it moves with the module (not a parameter).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Standard RoPE rotation primitive: rotate halves of last dim."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope_axis(
        self, x: torch.Tensor, pos: torch.Tensor, axis_dim_start: int
    ) -> torch.Tensor:
        """Apply RoPE rotation to a slice of the head dim.

        Parameters
        ----------
        x : Tensor of shape (B, num_heads, N, head_dim).
            Full Q or K. We rotate the slice [axis_dim_start : axis_dim_start + half_dim].
        pos : Tensor of shape (B, N).
            Positions for this axis (cx or cy values, scaled to RoPE-friendly range).
        axis_dim_start : int
            Index in head_dim where this axis's slice begins (0 for x, half_dim for y).

        Returns
        -------
        Tensor of same shape as x with the slice rotated.
        """
        # Slice out the axis portion: (B, num_heads, N, half_dim)
        x_axis = x[..., axis_dim_start : axis_dim_start + self.half_dim]

        # Compute angles: (B, N, half_dim/2)
        # pos: (B, N) -> (B, N, 1) * inv_freq: (half_dim/2,) -> (B, N, half_dim/2)
        angles = pos.unsqueeze(-1) * self.inv_freq.to(pos.dtype)

        # Build cos/sin to shape (B, 1, N, half_dim) by repeating each freq
        # twice along the last dim (matching the rotate_half pattern).
        cos = angles.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = angles.sin().repeat_interleave(2, dim=-1).unsqueeze(1)

        rotated_axis = x_axis * cos + self._rotate_half(x_axis) * sin

        # Splice back into x (without modifying the unrotated portion).
        out = x.clone()
        out[..., axis_dim_start : axis_dim_start + self.half_dim] = rotated_axis
        return out

    def rotate_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        spatial_metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D RoPE to Q and K based on token spatial metadata.

        (Renamed from `apply` to avoid colliding with `nn.Module.apply`,
        which is used by PyTorch for recursive module-traversal init.)

        Parameters
        ----------
        q : Tensor of shape (B, num_heads, N, head_dim). Query tensor.
        k : Tensor of shape (B, num_heads, N, head_dim). Key tensor.
        spatial_metadata : Tensor of shape (B, N, 4).
            (cx, cy, w, h) for each token, normalized to [0, 1].
            We use cx and cy; w and h are ignored here (used as scale
            biases in the attention block, see architecture/spatial/block.py).

        Returns
        -------
        (q_rope, k_rope) : both same shape as input.
        """
        if q.shape != k.shape:
            raise ValueError(
                f"q and k must have the same shape, got {tuple(q.shape)} vs {tuple(k.shape)}"
            )
        if q.shape[-1] != self.head_dim:
            raise ValueError(
                f"q's last dim ({q.shape[-1]}) != head_dim ({self.head_dim})"
            )
        B, N = q.shape[0], q.shape[2]
        if spatial_metadata.shape[:2] != (B, N):
            raise ValueError(
                f"spatial_metadata shape {tuple(spatial_metadata.shape)} "
                f"!= ({B}, {N}, 4)"
            )

        # Extract cx and cy, scale to RoPE-friendly range.
        cx = spatial_metadata[..., 0] * self.pos_scale  # (B, N)
        cy = spatial_metadata[..., 1] * self.pos_scale

        # Apply RoPE to the x slice [0:half_dim] using cx, then y slice using cy.
        q_rope = self._apply_rope_axis(q, cx, axis_dim_start=0)
        q_rope = self._apply_rope_axis(q_rope, cy, axis_dim_start=self.half_dim)
        k_rope = self._apply_rope_axis(k, cx, axis_dim_start=0)
        k_rope = self._apply_rope_axis(k_rope, cy, axis_dim_start=self.half_dim)

        return q_rope, k_rope


def compute_scale_bias(
    spatial_metadata: torch.Tensor,
    bias_dim: int,
    learned_scale_embedding: nn.Parameter,
) -> torch.Tensor:
    """Compute a per-token scale bias added to query vectors.

    Per docs/03_architecture_design.md §5.2: tokens at the same location
    but different spatial scales (region vs detail over the same point)
    should be distinguishable. RoPE gives them the same rotation. We add
    a small learned scale-aware bias that depends on (w * h) — the area
    fraction the token covers.

    Parameters
    ----------
    spatial_metadata : Tensor of shape (B, N, 4).
    bias_dim : int. Dimension of the bias (matches the embed_dim / num_heads * num_heads).
    learned_scale_embedding : Parameter of shape (num_scale_buckets, bias_dim).
        Learned bias values per scale bucket. We bucket area into `num_scale_buckets`
        log-spaced bins.

    Returns
    -------
    Tensor of shape (B, N, bias_dim).
    """
    num_buckets = learned_scale_embedding.shape[0]
    # Area = w * h, in [0, 1]. Take log and bucket.
    area = (spatial_metadata[..., 2] * spatial_metadata[..., 3]).clamp(min=1e-6)
    log_area = area.log()  # in (-inf, 0]
    # Map log_area to bucket index in [0, num_buckets - 1].
    # Empirically log_area ranges from ~ -10 (very tiny) to 0 (full image).
    bucket = ((log_area + 10.0) / 10.0 * (num_buckets - 1)).clamp(0, num_buckets - 1)
    bucket = bucket.long()  # (B, N)
    return learned_scale_embedding[bucket]
