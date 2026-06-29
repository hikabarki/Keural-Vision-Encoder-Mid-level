"""
Keural Vision Encoder — Output Contract.

This module defines KeuralEncoderOutput, the structured output of every
Keural vision encoder forward pass. It is the interface contract between
the encoder and any downstream consumer (LLMs, retrieval heads, projectors).

Stability guarantee
-------------------
The fields in KeuralEncoderOutput are part of the public API. They will not
be removed or have their semantics changed in any version of the v0.x line.
New fields may be added (always with default values) — never removed.

Why a dataclass and not a Tensor or dict?
-----------------------------------------
- A bare Tensor cannot expose level_ids, spatial_metadata, etc.
- A dict has no type safety and no autocomplete in editors.
- A dataclass gives users tab-completion, mypy/pyright support, and
  forward-compatibility through default values.

Reference: docs/02_release_plan.md §4 and docs/04_revisions.md §3 (Revision A).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch import Tensor


# Hierarchy level constants. These IDs are part of the public contract.
# They are stable forever — downstream code is allowed to hardcode them.
LEVEL_GLOBAL: int = 0
LEVEL_REGION: int = 1
LEVEL_DETAIL: int = 2

NUM_LEVELS: int = 3

LEVEL_NAMES: dict[int, str] = {
    LEVEL_GLOBAL: "global",
    LEVEL_REGION: "region",
    LEVEL_DETAIL: "detail",
}


@dataclass
class KeuralEncoderOutput:
    """Structured output of a Keural Vision Encoder forward pass.

    All tensors are batch-first. The leading dimension is always the batch
    size B. The token dimension N is the *padded* sequence length — only
    the first attention_mask.sum(dim=1)[i] tokens are real for batch item i.

    Attributes
    ----------
    tokens
        Shape (B, N, D). Per-token embedding vectors. Padding positions hold
        zeros (or arbitrary values — consumers must respect attention_mask).
    attention_mask
        Shape (B, N). Boolean-valued (stored as int or bool tensor).
        1 = real token, 0 = padding. Use this when feeding tokens to an LLM
        or any layer that needs to know which positions are real.
    level_ids
        Shape (B, N). Integer in {LEVEL_GLOBAL=0, LEVEL_REGION=1, LEVEL_DETAIL=2}
        per token. Padding positions hold a sentinel value of -1.
    spatial_metadata
        Shape (B, N, 4). Per-token (cx, cy, w, h) all normalized to [0, 1]
        relative to the original image dimensions, regardless of any internal
        resizing or padding. Padding positions hold zeros.
    saliency_scores
        Shape (B, N). Per-token saliency confidence in [0, 1] from the
        allocator. Useful for attention biasing and for understanding
        which tokens the encoder considered most informative.
    pooled
        Shape (B, D). Single-vector summary of the image. Comes from the
        learned [POOL] token in the spatial encoder. Use this for retrieval,
        zero-shot classification, or any task that needs a single image
        embedding regardless of token count.
    token_budget_used
        The actual budget used for this forward pass, after applying floors
        and auto-defaulting. Reported per-batch as a list of ints, since
        bucketed batching may use different budgets per micro-batch.
        Stored as a 1D tensor of shape (B,) for tensor-friendliness.

    Notes on shape stability
    ------------------------
    - D (token embedding dim) is fixed by the model config and never changes.
    - N (token count) varies between forward passes — it depends on the
      requested budget and on what the allocator decides per image. Use
      attention_mask to know what's real.

    See Also
    --------
    docs/02_release_plan.md : §4 — full output contract specification.
    docs/04_revisions.md : §3 (Revision A — token_budget_used field).
    """

    tokens: Tensor
    attention_mask: Tensor
    level_ids: Tensor
    spatial_metadata: Tensor
    saliency_scores: Tensor
    pooled: Tensor
    token_budget_used: Tensor

    # --- Convenience accessors. These are not state; they're computed from
    # the canonical fields. Kept as methods rather than properties to make
    # call sites explicit. ---

    def num_real_tokens(self) -> Tensor:
        """Return number of real (non-padding) tokens per batch item.

        Returns
        -------
        Tensor of shape (B,), integer dtype.
        """
        return self.attention_mask.sum(dim=1)

    def get_level_mask(self, level: int) -> Tensor:
        """Boolean mask of tokens at a specific hierarchy level.

        Parameters
        ----------
        level
            One of LEVEL_GLOBAL, LEVEL_REGION, LEVEL_DETAIL.

        Returns
        -------
        Tensor of shape (B, N), bool. True = token is at this level AND is
        a real (non-padding) token. False otherwise.
        """
        if level not in LEVEL_NAMES:
            raise ValueError(
                f"level must be one of {list(LEVEL_NAMES)}, got {level}. "
                f"Use the LEVEL_GLOBAL/LEVEL_REGION/LEVEL_DETAIL constants."
            )
        is_level = self.level_ids == level
        is_real = self.attention_mask.bool()
        return is_level & is_real

    def get_tokens_for_level(self, level: int) -> tuple[Tensor, Tensor]:
        """Extract tokens belonging to a specific level.

        Because token counts vary per batch item, this returns a flat
        (M, D) tensor along with a (M,) batch-index tensor saying which
        batch each token came from. This is the common pattern for working
        with variable-length grouped data in PyTorch.

        Parameters
        ----------
        level
            One of LEVEL_GLOBAL, LEVEL_REGION, LEVEL_DETAIL.

        Returns
        -------
        tokens_at_level : Tensor of shape (M, D)
        batch_index : Tensor of shape (M,), int64
        """
        mask = self.get_level_mask(level)
        # Find batch indices for each True position
        batch_idx = torch.arange(self.tokens.shape[0], device=self.tokens.device)
        batch_idx = batch_idx.unsqueeze(1).expand_as(mask)[mask]
        tokens_at_level = self.tokens[mask]
        return tokens_at_level, batch_idx

    def to(self, *args: Any, **kwargs: Any) -> KeuralEncoderOutput:
        """Move all tensor fields to a device/dtype. Mirrors Tensor.to().

        Integer-typed fields (attention_mask, level_ids, token_budget_used)
        are *device-moved* but *not* dtype-cast when the target dtype is
        a float type. They retain their integer dtype always. Casting them
        to float would silently corrupt the sentinel values and break
        downstream code that uses them as indices.

        Parameters
        ----------
        *args, **kwargs
            Passed to each tensor's .to() method, with float-dtype casts
            filtered out for integer fields.

        Returns
        -------
        A new KeuralEncoderOutput with all fields moved.
        """
        # Names of fields that MUST stay integer-typed regardless of `to(dtype=...)`.
        # These fields hold indices, masks, and counts — semantically integer.
        _INT_FIELDS = {"attention_mask", "level_ids", "token_budget_used"}

        new_values: dict[str, Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, Tensor):
                new_values[f.name] = value
                continue

            if f.name in _INT_FIELDS:
                # Strip dtype kwarg if it's a float dtype; keep device move.
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
                # Also strip dtype from positional args. torch.Tensor.to accepts
                # dtype as a positional arg; we conservatively pass only kwargs
                # that we recognize. This is the standard pattern in HF code.
                positional_dtype = next(
                    (a for a in args if isinstance(a, torch.dtype)), None
                )
                filtered_args = tuple(a for a in args if not isinstance(a, torch.dtype))
                # If the target dtype is integer, allow it; otherwise drop.
                if positional_dtype is not None and not positional_dtype.is_floating_point:
                    filtered_args = (*filtered_args, positional_dtype)
                if "dtype" in kwargs and not kwargs["dtype"].is_floating_point:
                    filtered_kwargs["dtype"] = kwargs["dtype"]
                new_values[f.name] = value.to(*filtered_args, **filtered_kwargs)
            else:
                new_values[f.name] = value.to(*args, **kwargs)
        return KeuralEncoderOutput(**new_values)

    def detach(self) -> KeuralEncoderOutput:
        """Detach all tensor fields from the autograd graph.

        Returns
        -------
        A new KeuralEncoderOutput with all tensors detached.
        """
        return KeuralEncoderOutput(
            **{f.name: getattr(self, f.name).detach() for f in fields(self)}
        )

    def validate(self) -> None:
        """Raise AssertionError if shapes/dtypes are inconsistent.

        Use during development and in unit tests. Production code can
        skip this since the encoder construction enforces the contract.
        """
        B, N, D = self.tokens.shape

        assert self.attention_mask.shape == (B, N), (
            f"attention_mask shape {tuple(self.attention_mask.shape)} "
            f"!= expected ({B}, {N})"
        )
        assert self.level_ids.shape == (B, N), (
            f"level_ids shape {tuple(self.level_ids.shape)} != expected ({B}, {N})"
        )
        assert self.spatial_metadata.shape == (B, N, 4), (
            f"spatial_metadata shape {tuple(self.spatial_metadata.shape)} "
            f"!= expected ({B}, {N}, 4)"
        )
        assert self.saliency_scores.shape == (B, N), (
            f"saliency_scores shape {tuple(self.saliency_scores.shape)} "
            f"!= expected ({B}, {N})"
        )
        assert self.pooled.shape == (B, D), (
            f"pooled shape {tuple(self.pooled.shape)} != expected ({B}, {D})"
        )
        assert self.token_budget_used.shape == (B,), (
            f"token_budget_used shape {tuple(self.token_budget_used.shape)} "
            f"!= expected ({B},)"
        )

        # Level IDs must be valid (-1 for padding, 0/1/2 for real tokens).
        # We don't check this elementwise (slow); we check the bounds.
        max_level = self.level_ids.max().item()
        min_level = self.level_ids.min().item()
        assert max_level <= NUM_LEVELS - 1, (
            f"level_ids contains value {max_level} > max valid level {NUM_LEVELS - 1}"
        )
        assert min_level >= -1, (
            f"level_ids contains value {min_level} < -1 (padding sentinel)"
        )

        # Spatial metadata must be in [0, 1] for real tokens. We allow
        # any value in padding positions; checking that here is expensive.

    def __repr__(self) -> str:  # pragma: no cover — for debugging only
        B, N, D = self.tokens.shape
        return (
            f"KeuralEncoderOutput(B={B}, N={N}, D={D}, "
            f"avg_real_tokens={self.num_real_tokens().float().mean().item():.1f}, "
            f"device={self.tokens.device})"
        )


def empty_output(
    batch_size: int,
    max_tokens: int,
    embed_dim: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> KeuralEncoderOutput:
    """Construct a fully-padded KeuralEncoderOutput for testing or as a template.

    All real-token fields are zero-filled. attention_mask is all zeros (no
    real tokens). level_ids is filled with -1 (the padding sentinel).

    This is useful in unit tests, in code that needs to allocate output
    buffers, and as a template that downstream code can fill incrementally.

    Parameters
    ----------
    batch_size : int
        Number of items in the batch.
    max_tokens : int
        Padded sequence length N.
    embed_dim : int
        Token embedding dim D.
    device, dtype
        Standard torch placement controls. Integer fields (level_ids,
        attention_mask, token_budget_used) ignore dtype and use int64.

    Returns
    -------
    KeuralEncoderOutput with all fields allocated and zeroed/sentinel-filled.
    """
    return KeuralEncoderOutput(
        tokens=torch.zeros(batch_size, max_tokens, embed_dim, device=device, dtype=dtype),
        attention_mask=torch.zeros(batch_size, max_tokens, device=device, dtype=torch.int64),
        level_ids=torch.full(
            (batch_size, max_tokens), fill_value=-1, device=device, dtype=torch.int64
        ),
        spatial_metadata=torch.zeros(
            batch_size, max_tokens, 4, device=device, dtype=dtype
        ),
        saliency_scores=torch.zeros(batch_size, max_tokens, device=device, dtype=dtype),
        pooled=torch.zeros(batch_size, embed_dim, device=device, dtype=dtype),
        token_budget_used=torch.zeros(batch_size, device=device, dtype=torch.int64),
    )
