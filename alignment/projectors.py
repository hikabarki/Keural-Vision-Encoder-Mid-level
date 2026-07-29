"""
Keural Vision Encoder — Projectors for LLM Integration.

Three projector variants per docs/04_revisions.md §6 (Revision D). All
inherit from KeuralProjectorBase for swappable use.

Choice guide
------------
| If you...                                                  | Use                  |
|------------------------------------------------------------|----------------------|
| ...are migrating from CLIP/ViT                             | SimpleProjector      |
| ...are building a new VLM from scratch                     | LevelAwareProjector  |
| ...need spatial reasoning, OCR, or document understanding  | MetadataProjector    |

All projectors take a KeuralEncoderOutput and return a Tensor of shape
(B, N, llm_dim). The output is masked-aware — padding positions carry
the projection of zero (or, with metadata projectors, of zero metadata),
which is fine because downstream attention masks block them out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from alignment.output_contract import KeuralEncoderOutput, NUM_LEVELS


class KeuralProjectorBase(nn.Module):
    """Base class for all Keural→LLM projectors.

    Subclasses implement `forward(output: KeuralEncoderOutput) -> Tensor`.
    All projectors expose `encoder_dim` and `llm_dim` for introspection.
    """

    def __init__(self, encoder_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

    def forward(self, output: KeuralEncoderOutput) -> torch.Tensor:
        raise NotImplementedError


class SimpleProjector(KeuralProjectorBase):
    """Drop-in replacement for the linear projector in LLaVA-style VLMs.

    Maps Keural tokens to LLM hidden dim with a 2-layer MLP. Ignores
    level_ids, spatial_metadata, and saliency_scores — use this when
    migrating an existing VLM from CLIP/ViT to Keural with minimal
    code changes.

    Architecture: Linear → GELU → Linear

    Parameters
    ----------
    encoder_dim : int
        Keural encoder embedding dim D.
    llm_dim : int
        Target LLM hidden dim.
    hidden_dim : int, optional
        Hidden dim of the 2-layer MLP. Defaults to llm_dim.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__(encoder_dim, llm_dim)
        if hidden_dim is None:
            hidden_dim = llm_dim
        self.fc1 = nn.Linear(encoder_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, output: KeuralEncoderOutput) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(output.tokens)))


class LevelAwareProjector(KeuralProjectorBase):
    """Recommended default for new Keural-based VLMs.

    Projects tokens through an MLP, then adds a learned per-level bias.
    The level bias gives the LLM an explicit signal of what type of token
    it is consuming (global summary vs region vs detail), without the LLM
    having to infer it from feature magnitudes.

    Architecture: (Linear → GELU → Linear) + LevelEmbedding[level_ids]

    Parameters
    ----------
    encoder_dim : int
    llm_dim : int
    hidden_dim : int, optional. Defaults to llm_dim.
    num_levels : int, default NUM_LEVELS (3). Should not be changed in v0.x.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        hidden_dim: int | None = None,
        num_levels: int = NUM_LEVELS,
    ) -> None:
        super().__init__(encoder_dim, llm_dim)
        if hidden_dim is None:
            hidden_dim = llm_dim
        self.fc1 = nn.Linear(encoder_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        # Level bias table — one bias vector per level, including a slot
        # for the padding sentinel (-1). We index with `level_ids + 1` so
        # padding (-1 → 0) gets a learned-but-irrelevant bias (it's masked
        # out downstream anyway).
        self.level_embed = nn.Embedding(num_levels + 1, llm_dim)
        nn.init.zeros_(self.level_embed.weight)

    def forward(self, output: KeuralEncoderOutput) -> torch.Tensor:
        projected = self.fc2(self.act(self.fc1(output.tokens)))
        # level_ids is in {-1, 0, 1, 2}; shift by +1 for embedding lookup.
        level_bias = self.level_embed(output.level_ids + 1)
        return projected + level_bias


class MetadataProjector(KeuralProjectorBase):
    """For tasks needing spatial awareness — OCR, grounding, documents.

    Adds spatial-metadata fusion on top of the level-aware projector.
    Each token receives:
      base projection
      + level bias
      + projection of (cx, cy, w, h, saliency_score)

    The spatial metadata gives the LLM an explicit feature space encoding
    of WHERE each token came from in the image.

    Architecture:
        token_proj = Linear(D, llm_dim) ∘ GELU ∘ Linear(llm_dim, llm_dim)
        meta_proj  = Linear(5, llm_dim/4) ∘ GELU ∘ Linear(llm_dim/4, llm_dim)
        out        = token_proj + level_embed + meta_proj
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        hidden_dim: int | None = None,
        num_levels: int = NUM_LEVELS,
        meta_hidden: int | None = None,
    ) -> None:
        super().__init__(encoder_dim, llm_dim)
        if hidden_dim is None:
            hidden_dim = llm_dim
        if meta_hidden is None:
            meta_hidden = max(32, llm_dim // 4)

        # Token MLP — same as LevelAwareProjector
        self.fc1 = nn.Linear(encoder_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

        # Level bias
        self.level_embed = nn.Embedding(num_levels + 1, llm_dim)
        nn.init.zeros_(self.level_embed.weight)

        # Metadata MLP: 5 inputs (cx, cy, w, h, saliency) → llm_dim
        self.meta_fc1 = nn.Linear(5, meta_hidden)
        self.meta_fc2 = nn.Linear(meta_hidden, llm_dim)
        # Init the second metadata layer to zeros so the projector starts
        # behaving exactly like LevelAwareProjector at init. Specialization
        # toward metadata-aware behavior emerges through training.
        nn.init.zeros_(self.meta_fc2.weight)
        nn.init.zeros_(self.meta_fc2.bias)

    def forward(self, output: KeuralEncoderOutput) -> torch.Tensor:
        projected = self.fc2(self.act(self.fc1(output.tokens)))
        level_bias = self.level_embed(output.level_ids + 1)

        # Concat metadata: (B, N, 5)
        meta_in = torch.cat(
            [output.spatial_metadata, output.saliency_scores.unsqueeze(-1)],
            dim=-1,
        )
        meta_proj = self.meta_fc2(self.act(self.meta_fc1(meta_in)))

        return projected + level_bias + meta_proj
