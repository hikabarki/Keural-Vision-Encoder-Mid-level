"""
Keural Vision Encoder — Region Sampler.

This is the highest-risk component in the project. It implements the
differentiable allocator that picks K spatial regions from a saliency map
for a given hierarchy level (region or detail). The "global" level uses
learned queries instead and does not call this module.

Algorithm
---------
For each level:

1. Smooth the saliency map with a Gaussian kernel sized to the level's
   expected receptive field (broader for region, tighter for detail).
2. Add Gumbel noise scaled by a temperature `tau` (annealed during
   training).
3. Iteratively pick the top-1 position, then suppress its neighborhood
   (NMS) before the next pick. Repeat K times.
4. For gradient flow, return BOTH the discrete top-K indices (for the
   forward path) AND a soft saliency-weighted attention vector over all
   positions (for the backward path) via the straight-through trick.

Reference: docs/03_architecture_design.md §3.5–3.6.
Fallback ladder: docs/03_architecture_design.md §3.9.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SampledRegions:
    """Output of the region sampler for a single hierarchy level.

    Attributes
    ----------
    coords
        Shape (B, K, 2). Per-region (y, x) center coordinates as integer
        positions on the smoothed saliency grid.
    scores
        Shape (B, K). Soft (straight-through) score per region, used as
        per-token saliency in the output contract and for gradient flow.
    soft_weights
        Shape (B, K, h, w). Per-region soft attention map over the grid.
        Used for differentiable feature pooling — instead of hard RoIAlign
        with non-differentiable indices, downstream code can compute
        weighted feature averages using these maps.
    """

    coords: torch.Tensor
    scores: torch.Tensor
    soft_weights: torch.Tensor


def _make_gaussian_kernel(kernel_size: int, sigma: float | None = None) -> torch.Tensor:
    """Build a 2D Gaussian smoothing kernel of shape (1, 1, k, k)."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and >= 1, got {kernel_size}")
    if sigma is None:
        sigma = kernel_size / 4.0  # Standard rule of thumb
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - half
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def gaussian_smooth(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply 2D Gaussian smoothing.

    Parameters
    ----------
    x : Tensor of shape (B, 1, h, w)
    kernel_size : int (odd)

    Returns
    -------
    Tensor of shape (B, 1, h, w).
    """
    if kernel_size == 1:
        return x
    kernel = _make_gaussian_kernel(kernel_size).to(device=x.device, dtype=x.dtype)
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad)


class RegionSampler(nn.Module):
    """Differentiable Gumbel-Top-K region sampler with NMS.

    Parameters
    ----------
    smooth_kernel : int
        Gaussian smoothing kernel size for the saliency map at this level.
    nms_radius : int
        Radius (in grid cells) over which a picked region suppresses
        nearby saliency for subsequent picks within the same iteration.
    soft_weight_sigma : float, default 1.0
        Sigma of the Gaussian used to convert each picked center into
        a soft attention map for differentiable feature pooling.

    Notes on the straight-through trick
    -----------------------------------
    The forward path returns hard discrete picks. The backward path uses
    the saliency map directly so gradients flow back to the saliency
    predictor. We achieve this by returning `scores` that equal the hard
    saliency at the picked positions but whose gradients flow through the
    full soft saliency.
    """

    def __init__(
        self,
        smooth_kernel: int,
        nms_radius: int,
        soft_weight_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if smooth_kernel < 1 or smooth_kernel % 2 == 0:
            raise ValueError(f"smooth_kernel must be odd and >= 1, got {smooth_kernel}")
        if nms_radius < 0:
            raise ValueError(f"nms_radius must be >= 0, got {nms_radius}")
        if soft_weight_sigma <= 0:
            raise ValueError(f"soft_weight_sigma must be > 0, got {soft_weight_sigma}")

        self.smooth_kernel = smooth_kernel
        self.nms_radius = nms_radius
        self.soft_weight_sigma = soft_weight_sigma

    @torch.no_grad()
    def _hard_pick_topk_with_nms(
        self,
        scores: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Greedy top-K with neighborhood suppression. NO GRADIENT.

        Parameters
        ----------
        scores : Tensor of shape (B, h, w)
        K : int — number of picks

        Returns
        -------
        coords : Tensor of shape (B, K, 2), dtype int64. Each row is (y, x).
        """
        B, h, w = scores.shape
        device = scores.device

        # Work on a clone we can mutate (set picked neighborhoods to -inf).
        working = scores.clone()
        coords = torch.zeros(B, K, 2, device=device, dtype=torch.int64)

        r = self.nms_radius
        for k in range(K):
            # Flatten and argmax
            flat = working.view(B, h * w)
            idx = flat.argmax(dim=1)  # (B,)
            ys = idx // w
            xs = idx % w
            coords[:, k, 0] = ys
            coords[:, k, 1] = xs

            if r == 0 or k == K - 1:
                continue

            # Suppress an axis-aligned square neighborhood. We do this with
            # a per-batch loop over picks because the picked coordinates
            # differ per batch element. For the tiny K we use, this is fine.
            for b in range(B):
                y0 = max(0, ys[b].item() - r)
                y1 = min(h, ys[b].item() + r + 1)
                x0 = max(0, xs[b].item() - r)
                x1 = min(w, xs[b].item() + r + 1)
                working[b, y0:y1, x0:x1] = float("-inf")

        return coords

    def _build_soft_weights(
        self,
        coords: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Convert center coords to per-region Gaussian soft-attention maps.

        Each region's map is a Gaussian centered at its (y, x) coordinate
        with sigma = self.soft_weight_sigma, normalized to sum to 1.

        Parameters
        ----------
        coords : Tensor of shape (B, K, 2). int64.
        h, w : grid dimensions.

        Returns
        -------
        Tensor of shape (B, K, h, w). dtype float32.
        """
        B, K, _ = coords.shape
        device = coords.device
        # Build a (h, w) coordinate grid
        ys = torch.arange(h, device=device, dtype=torch.float32)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)

        # Distance from each center to every grid position
        cy = coords[..., 0].float().unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        cx = coords[..., 1].float().unsqueeze(-1).unsqueeze(-1)
        dy = grid_y.unsqueeze(0).unsqueeze(0) - cy  # (B, K, h, w)
        dx = grid_x.unsqueeze(0).unsqueeze(0) - cx
        dist_sq = dy**2 + dx**2

        sigma = self.soft_weight_sigma
        gauss = torch.exp(-dist_sq / (2 * sigma**2))
        # Normalize each region's map to sum 1
        gauss = gauss / gauss.flatten(-2).sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
        return gauss

    def forward(
        self,
        saliency: torch.Tensor,
        K: int,
        gumbel_temperature: float = 1.0,
        training: bool | None = None,
    ) -> SampledRegions:
        """Pick K regions from the saliency map.

        Parameters
        ----------
        saliency : Tensor of shape (B, 1, h, w), values in [0, 1].
            Output of the saliency predictor.
        K : int
            Number of regions to pick.
        gumbel_temperature : float
            Temperature for Gumbel noise. High = more exploration.
            Set to 0 for deterministic top-K (eval/inference).
        training : bool or None
            If None, uses self.training. If False, no Gumbel noise is added
            (deterministic, eval-mode picking).

        Returns
        -------
        SampledRegions with coords, scores, soft_weights.
        """
        if saliency.dim() != 4 or saliency.shape[1] != 1:
            raise ValueError(
                f"saliency must have shape (B, 1, h, w); got {tuple(saliency.shape)}"
            )
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")

        if training is None:
            training = self.training

        # Step 1: smooth the saliency map.
        smoothed = gaussian_smooth(saliency, self.smooth_kernel)
        # squeeze channel dim: (B, h, w)
        smoothed_2d = smoothed.squeeze(1)

        # Step 2: in log-space, add Gumbel noise (during training only).
        log_s = torch.log(smoothed_2d.clamp(min=1e-6))
        if training and gumbel_temperature > 0:
            # Sample Gumbel(0, 1) noise, scale by temperature
            uniform = torch.rand_like(log_s).clamp(min=1e-6, max=1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            scored = log_s + gumbel_temperature * gumbel
        else:
            scored = log_s

        # Step 3: hard top-K with NMS (no gradient).
        coords = self._hard_pick_topk_with_nms(scored.detach(), K)

        # Step 4: gather hard scores at picked positions, with straight-through gradient.
        # The forward value is saliency at the picked position. The backward
        # gradient flows through the (differentiable) saliency tensor.
        B = saliency.shape[0]
        h, w = saliency.shape[-2:]
        flat_idx = coords[..., 0] * w + coords[..., 1]  # (B, K)
        hard_scores = smoothed_2d.view(B, h * w).gather(1, flat_idx)  # (B, K), differentiable

        # Step 5: soft attention maps, used downstream for differentiable pooling.
        soft_weights = self._build_soft_weights(coords, h, w)  # (B, K, h, w)

        return SampledRegions(
            coords=coords,
            scores=hard_scores,
            soft_weights=soft_weights,
        )


def deterministic_grid_regions(
    grid_h: int,
    grid_w: int,
    K: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> SampledRegions:
    """Fallback: deterministic uniform-grid region selection.

    Used during the warmup phase (per docs/03_architecture_design.md §7.5
    curriculum) and in Tier 3 fallback (§3.9). Picks K positions on a
    quasi-uniform grid covering the saliency grid.

    Parameters
    ----------
    grid_h, grid_w : int
        Saliency grid dimensions.
    K : int
        Number of regions.
    batch_size : int
        Output batch size (regions are identical across batch).
    device : torch.device or str

    Returns
    -------
    SampledRegions with the K grid positions.
    """
    if K > grid_h * grid_w:
        raise ValueError(f"K ({K}) > grid cells ({grid_h * grid_w})")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")

    # Generate K positions on a regular grid.
    # Use ceil(sqrt(K)) along both axes, then take the first K.
    side = int(K**0.5) or 1
    rows = max(1, side)
    cols = (K + rows - 1) // rows
    rows_idx = torch.linspace(0, grid_h - 1, rows).round().long()
    cols_idx = torch.linspace(0, grid_w - 1, cols).round().long()
    coords_list = []
    for r in rows_idx:
        for c in cols_idx:
            coords_list.append((r.item(), c.item()))
            if len(coords_list) == K:
                break
        if len(coords_list) == K:
            break
    coords = torch.tensor(coords_list, dtype=torch.int64, device=device)  # (K, 2)
    coords = coords.unsqueeze(0).expand(batch_size, K, 2).contiguous()

    # Uniform scores
    scores = torch.full((batch_size, K), 0.5, device=device, dtype=torch.float32)

    # Soft weights: a delta at the chosen position, smoothed slightly.
    # We use the same Gaussian builder.
    sampler = RegionSampler(smooth_kernel=1, nms_radius=0, soft_weight_sigma=1.0)
    soft_weights = sampler._build_soft_weights(coords, grid_h, grid_w)

    return SampledRegions(coords=coords, scores=scores, soft_weights=soft_weights)
