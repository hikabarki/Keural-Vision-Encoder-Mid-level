"""
Keural Vision Encoder — Spatial Transformer Block.

A pre-norm transformer block with:
  - RMSNorm (pre-norm)
  - Multi-head attention with 2D-RoPE applied to Q and K
  - Scale-bias added to Q (for differentiating tokens at the same position
    but different spatial scales)
  - SwiGLU MLP

Per docs/03_architecture_design.md §5.3, this block is deliberately
conventional. Innovation lives in the tokenizer, not here.

Reference: docs/03_architecture_design.md §5.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.spatial.rope_2d import RoPE2DFromMetadata, compute_scale_bias
from keural_mid_config import SpatialConfig


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    Per the spec, we use RMSNorm rather than LayerNorm for stability and
    a small speed-up. Standard formulation: y = x * rsqrt(mean(x^2) + eps) * w
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability under bf16/fp16 training.
        in_dtype = x.dtype
        x_fp = x.float()
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        normed = (x_fp * rms).to(in_dtype)
        # Cast weight to input dtype to preserve the original dtype.
        return normed * self.weight.to(in_dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU activation MLP — two parallel projections combined as silu(a) * b.

    Hidden dim is mlp_ratio * embed_dim. We round to nearest multiple of
    256 for hardware-friendly shapes.
    """

    def __init__(self, embed_dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        # SwiGLU uses 2/3 of the standard MLP hidden size (a published rule
        # of thumb that keeps parameter count comparable to a 4x GELU MLP).
        hidden = int(embed_dim * mlp_ratio * 2 / 3)
        # Round up to multiple of 64 for matmul efficiency.
        hidden = ((hidden + 63) // 64) * 64

        self.w_gate = nn.Linear(embed_dim, hidden, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SpatialAttention(nn.Module):
    """Multi-head self-attention with 2D-RoPE on Q and K.

    Applies 2D rotary positional embedding from spatial metadata (cx, cy)
    and adds a learned scale bias to Q (so tokens at the same position
    but different scales are distinguishable).

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    head_dim : int. Must satisfy num_heads * head_dim == embed_dim.
    rope_base : float. RoPE frequency base.
    num_scale_buckets : int, default 8. Number of log-area buckets for the
        scale bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        num_scale_buckets: int = 8,
    ) -> None:
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(
                f"num_heads * head_dim ({num_heads * head_dim}) != embed_dim ({embed_dim})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RoPE2DFromMetadata(head_dim=head_dim, base=rope_base)

        # Learned scale bias parameters: per-bucket bias applied to Q.
        self.scale_bias = nn.Parameter(torch.zeros(num_scale_buckets, embed_dim))
        nn.init.trunc_normal_(self.scale_bias, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        spatial_metadata: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (B, N, D). Pre-norm input.
        spatial_metadata : Tensor of shape (B, N, 4).
        attention_mask : Optional Tensor of shape (B, N).
            1 for real tokens, 0 for padding. Padded positions are masked
            from attention (so real tokens never attend to padding).

        Returns
        -------
        Tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        # Add scale bias to the input before projecting Q.
        scale_bias = compute_scale_bias(
            spatial_metadata, bias_dim=D, learned_scale_embedding=self.scale_bias
        )  # (B, N, D)
        q_in = x + scale_bias

        q = self.q_proj(q_in)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE to Q and K.
        q, k = self.rope.rotate_qk(q, k, spatial_metadata)

        # Build attention mask if provided. SDPA expects shape broadcastable
        # to (B, num_heads, N, N) with -inf at masked positions.
        attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            mask = attention_mask.bool()  # (B, N)
            # Token i can attend to token j only if j is real.
            # Shape (B, 1, 1, N) — broadcasts over query and head dims.
            attn_mask = torch.zeros(
                B, 1, 1, N, device=x.device, dtype=q.dtype
            )
            attn_mask = attn_mask.masked_fill(~mask[:, None, None, :], float("-inf"))

        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # (B, num_heads, N, head_dim) -> (B, N, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(attended)


class SpatialBlock(nn.Module):
    """One pre-norm transformer block: norm-attn-residual + norm-mlp-residual."""

    def __init__(self, config: SpatialConfig) -> None:
        super().__init__()

        self.norm1 = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.attn = SpatialAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            rope_base=config.rope_base,
        )
        self.norm2 = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(embed_dim=config.embed_dim, mlp_ratio=config.mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        spatial_metadata: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-norm forward: x = x + attn(norm(x)); x = x + mlp(norm(x))."""
        x = x + self.attn(self.norm1(x), spatial_metadata, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x
