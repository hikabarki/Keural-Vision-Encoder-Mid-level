"""Keural alignment — output contract, projectors, and VLM helpers.

Public API
----------
KeuralEncoderOutput
    The structured output of every Keural Vision Encoder forward pass.
LEVEL_GLOBAL, LEVEL_REGION, LEVEL_DETAIL
    Stable integer constants identifying hierarchy levels in level_ids.
LEVEL_NAMES
    Mapping {level_id: name} for human-readable display.
NUM_LEVELS
    Number of hierarchy levels (always 3 in v0.x).
empty_output
    Factory for a fully-padded output, useful in tests and as a buffer template.
"""

from alignment.output_contract import (
    LEVEL_DETAIL,
    LEVEL_GLOBAL,
    LEVEL_NAMES,
    LEVEL_REGION,
    NUM_LEVELS,
    KeuralEncoderOutput,
    empty_output,
)
from alignment.projectors import (
    KeuralProjectorBase,
    LevelAwareProjector,
    MetadataProjector,
    SimpleProjector,
)
from alignment.vlm_helpers import VLMInputs, build_vlm_inputs

__all__ = [
    "KeuralEncoderOutput",
    "LEVEL_GLOBAL",
    "LEVEL_REGION",
    "LEVEL_DETAIL",
    "LEVEL_NAMES",
    "NUM_LEVELS",
    "empty_output",
    # Projectors (ported from PoC, adapted for Mid's 768-dim encoder)
    "KeuralProjectorBase",
    "SimpleProjector",
    "LevelAwareProjector",
    "MetadataProjector",
    # VLM helpers
    "VLMInputs",
    "build_vlm_inputs",
]
