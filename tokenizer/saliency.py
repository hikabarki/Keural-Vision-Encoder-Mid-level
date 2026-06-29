"""
Keural Vision Encoder — Saliency Predictor.

A small conv head on top of stem features produces a per-spatial-position
saliency score in [0, 1]. Downstream, the ATB allocator uses these scores
to decide where to place tokens.

Why sigmoid (not softmax)?
--------------------------
Saliency is per-position — a region is salient or not, independent of
other regions. Multiple regions can be highly salient simultaneously.
Softmax would force them to compete for a fixed mass of probability,
which is the wrong inductive bias.

Reference: docs/03_architecture_design.md §3.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from keural_mid_config import SaliencyConfig


class SaliencyPredictor(nn.Module):
    """Two-conv head producing a (B, 1, h, w) sigmoid map.

    Architecture
    ------------
    Conv 3x3 (in_channels -> hidden) -> GELU -> Conv 1x1 (hidden -> 1) -> sigmoid

    The input is a stem feature map (B, in_channels, h, w). The output is
    a spatial map of shape (B, 1, h, w) with values in (0, 1).

    Parameters
    ----------
    config : SaliencyConfig
        Configuration for the head.
    in_channels : int
        Channel count of the stem features (== stem.out_channels).
    """

    def __init__(self, config: SaliencyConfig, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = config.hidden_channels

        self.conv_hidden = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv_score = nn.Conv2d(
            in_channels=config.hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Initialize the score head's bias so initial saliency is roughly
        # uniform around 0.5 (sigmoid(0) == 0.5). Trunc-normal small
        # weights and zero bias do this.
        nn.init.trunc_normal_(self.conv_score.weight, std=0.01)
        nn.init.zeros_(self.conv_score.bias)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute saliency map.

        Parameters
        ----------
        features
            Shape (B, in_channels, h, w). Stem feature map.
        valid_mask
            Optional shape (B, 1, h, w) or (B, h, w) tensor of {0, 1}.
            Positions with mask=0 (e.g., padded regions of the input
            image) get their saliency forced to 0 so the allocator does
            not spend tokens on padding.

        Returns
        -------
        Tensor of shape (B, 1, h, w) with values in (0, 1) — at masked
        positions, exactly 0.
        """
        if features.dim() != 4:
            raise ValueError(
                f"SaliencyPredictor expects (B, C, h, w); got {tuple(features.shape)}"
            )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"SaliencyPredictor: expected {self.in_channels} channels, "
                f"got {features.shape[1]}"
            )

        h = self.act(self.conv_hidden(features))
        logits = self.conv_score(h)
        saliency = torch.sigmoid(logits)

        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)  # (B, 1, h, w)
            if valid_mask.shape[-2:] != saliency.shape[-2:]:
                raise ValueError(
                    f"valid_mask spatial shape {tuple(valid_mask.shape[-2:])} "
                    f"does not match saliency {tuple(saliency.shape[-2:])}"
                )
            saliency = saliency * valid_mask.to(saliency.dtype)

        return saliency
