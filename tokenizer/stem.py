"""
Keural Vision Encoder — Variable-Resolution Conv Stem.

The stem reduces input spatial resolution by 16x via 4 strided conv layers
with normalization and non-linearity between them. Output is a feature
grid (B, C_out, H/16, W/16) where C_out equals the model's embed_dim.

Why a conv stem and not direct patch embedding?
-----------------------------------------------
ViT-style patchify-then-project produces hard patch boundaries that are
known to cause aliasing artifacts. A small conv stem (4 strided 3x3 convs)
costs little, provides smoother early features, and matches the inductive
bias we want for content-aware tokenization downstream.

Reference: docs/03_architecture_design.md §3.3 (revised by docs/04_revisions.md
§8 for variable-resolution support).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from keural_mid_config import StemConfig


def _activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class ConvStem(nn.Module):
    """4-layer strided conv stem with cumulative stride 16.

    Each layer halves spatial resolution and (typically) doubles channels.
    Layers use 3x3 convolutions with `padding=1` to preserve the
    "input H == 16 * output H" property when input H is a multiple of 16.

    Parameters
    ----------
    config : StemConfig
        Stem configuration (channel counts, kernel size, activation).
    in_channels : int, default 3
        Input channels. 3 for RGB images.

    Attributes
    ----------
    out_channels : int
        Output channel count, equal to config.channels[-1].
    stride : int
        Cumulative spatial stride (always 16 in V1).
    """

    def __init__(self, config: StemConfig, in_channels: int = 3) -> None:
        super().__init__()

        self._config = config
        self.in_channels = in_channels
        self.out_channels = config.channels[-1]
        self.stride = 16  # 2^4 with 4 strided layers

        ks = config.kernel_size
        pad = ks // 2

        # Build layers. Each "layer" is conv -> norm -> activation.
        # We use GroupNorm with 8 groups for resolution-independence
        # (BatchNorm is poor for small batches and varying resolutions).
        layers: list[nn.Module] = []
        prev = in_channels
        for ch in config.channels:
            layers.append(
                nn.Conv2d(
                    in_channels=prev,
                    out_channels=ch,
                    kernel_size=ks,
                    stride=2,
                    padding=pad,
                    bias=False,  # GroupNorm has its own affine bias
                )
            )
            # Group count: min(8, ch). Some channels may not be divisible by 8.
            num_groups = min(8, ch)
            while ch % num_groups != 0:
                num_groups -= 1
            layers.append(nn.GroupNorm(num_groups, ch))
            layers.append(_activation(config.activation))
            prev = ch

        self.layers = nn.Sequential(*layers)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute stem features.

        Parameters
        ----------
        pixel_values
            Shape (B, in_channels, H, W). H and W must each be multiples
            of `self.stride` (16 by default). Use `pad_to_stride` upstream
            if your data may have other sizes.

        Returns
        -------
        Tensor of shape (B, out_channels, H // stride, W // stride).
        """
        if pixel_values.dim() != 4:
            raise ValueError(
                f"ConvStem expects (B, C, H, W); got shape {tuple(pixel_values.shape)}"
            )
        B, C, H, W = pixel_values.shape
        if C != self.in_channels:
            raise ValueError(
                f"ConvStem expects {self.in_channels} channels, got {C}"
            )
        if H % self.stride != 0 or W % self.stride != 0:
            raise ValueError(
                f"ConvStem input H ({H}) and W ({W}) must each be multiples "
                f"of stride={self.stride}. Pad upstream with pad_to_stride()."
            )
        return self.layers(pixel_values)


def pad_to_stride(
    pixel_values: torch.Tensor, stride: int = 16
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Right/bottom-pad an image batch to make H, W multiples of `stride`.

    Used to support truly arbitrary input resolutions (per Revision F).
    The data pipeline normally produces bucket-aligned sizes, but when
    a user calls the encoder directly with an off-bucket size we want it
    to "just work."

    Parameters
    ----------
    pixel_values
        Shape (B, C, H, W).
    stride
        Spatial stride to align to. Default 16 (matches the conv stem).

    Returns
    -------
    padded : Tensor of shape (B, C, H', W') with H' >= H, W' >= W, both
        multiples of stride.
    pad_amounts : (pad_left, pad_right, pad_top, pad_bottom). Useful if
        downstream code needs to mask out the padded region from saliency.
    """
    if pixel_values.dim() != 4:
        raise ValueError(
            f"pad_to_stride expects (B, C, H, W); got {tuple(pixel_values.shape)}"
        )
    _, _, H, W = pixel_values.shape
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return pixel_values, (0, 0, 0, 0)
    # F.pad with (left, right, top, bottom). Pad with zeros — the saliency
    # mask in the data pipeline forces saliency=0 in padded areas.
    padded = nn.functional.pad(pixel_values, (0, pad_w, 0, pad_h), value=0.0)
    return padded, (0, pad_w, 0, pad_h)
