"""
HuggingFace `transformers`-compatible wrapper for the Keural Vision Encoder.

Why this exists at module #3 (Day 1)
------------------------------------
Per docs/04_revisions.md §5 (Revision C), HF integration is moved to Day 1
because it is the most failure-prone integration point in the project.
Catching `AutoModel.from_pretrained()` failures on a randomly-initialized
skeleton is much cheaper than catching them at hour 30 of training.

This skeleton can save and load weights now. The actual model logic is
filled in by `architecture/encoder.py` (module #11). When that module
exists, this wrapper imports it and uses it as `self.encoder`.

Public API
----------
KeuralVisionConfig
    HuggingFace PretrainedConfig that wraps our KeuralConfig.
KeuralVisionModel
    HuggingFace PreTrainedModel that wraps the actual encoder.

Usage
-----
After training and `model.save_pretrained("./out")`, users can do:

    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained("./out")
    model = AutoModel.from_pretrained("./out")

This works because we register the config and model classes with
transformers' Auto* registries via Auto*.register() calls below.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

from alignment.output_contract import KeuralEncoderOutput, empty_output
from architecture.encoder import KeuralVisionEncoder
from keural_mid_config import KeuralMidConfig as KeuralConfig


class KeuralVisionConfig(PretrainedConfig):
    """HuggingFace config wrapping a KeuralConfig.

    The full KeuralConfig is serialized to/from JSON via this class. We
    flatten KeuralConfig.to_dict() into the HF config's __dict__ so that
    `config.embed_dim`, `config.depth`, etc. work for backwards compat
    with HF inspection tools, while the canonical structured form is
    available as `config.keural_config`.

    The `model_type` "keural_vision" is the HF identifier under which
    AutoModel finds this class.
    """

    model_type = "keural_vision"

    def __init__(self, **kwargs: Any) -> None:
        # If a full keural_config dict was passed (the way we save), use it.
        # Otherwise build a default KeuralConfig.
        keural_dict = kwargs.pop("keural_config", None)
        if keural_dict is not None:
            self.keural_config: KeuralConfig = KeuralConfig.from_dict(keural_dict)
        else:
            self.keural_config = KeuralConfig()

        # Mirror common fields at the top level for HF tooling. These are
        # informational only — the canonical source is self.keural_config.
        self.embed_dim: int = self.keural_config.spatial.embed_dim
        self.depth: int = self.keural_config.spatial.depth
        self.num_heads: int = self.keural_config.spatial.num_heads
        self.default_token_budget: int = self.keural_config.tokenizer.default_token_budget
        self.max_token_budget: int = self.keural_config.tokenizer.max_token_budget

        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON dumping. Stores keural_config nested."""
        out = super().to_dict()
        # Replace mirrored fields with the structured config — single source of truth.
        out["keural_config"] = self.keural_config.to_dict()
        return out


class KeuralVisionModel(PreTrainedModel):
    """HuggingFace-compatible wrapper around the Keural Vision Encoder.

    At module #3 this is a SKELETON. The internal `encoder` attribute is
    a placeholder nn.Module with no real architecture. Module #11 replaces
    the placeholder with the actual encoder.

    The forward signature, output type, and save/load behavior are all
    final. Only the internals change.

    Parameters
    ----------
    config : KeuralVisionConfig
        HF config wrapping a KeuralConfig.
    """

    config_class = KeuralVisionConfig
    base_model_prefix = "keural"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: KeuralVisionConfig) -> None:
        super().__init__(config)

        self._keural_config: KeuralConfig = config.keural_config

        # Real encoder. Module #11 has landed.
        self.encoder: nn.Module = KeuralVisionEncoder(self._keural_config)

        self.post_init()

    # --- HF lifecycle hooks ---

    def _init_weights(self, module: nn.Module) -> None:  # noqa: D401
        """HF-required weight init hook. Trunc-normal for linear & conv layers."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # --- Forward ---

    def forward(
        self,
        pixel_values: torch.Tensor,
        token_budget: int | None = None,
        return_hierarchy: bool = True,
    ) -> KeuralEncoderOutput:
        """Forward pass.

        Parameters
        ----------
        pixel_values
            Shape (B, 3, H, W). Normalized image tensor.
        token_budget
            Runtime token budget. If None, uses cfg.tokenizer.default_token_budget
            (or auto-scaled by resolution per docs/04_revisions.md §3).
        return_hierarchy
            Reserved for future use. Currently always returns the full
            KeuralEncoderOutput.

        Returns
        -------
        KeuralEncoderOutput.
        """
        if token_budget is None:
            token_budget = self._resolve_default_budget(pixel_values)
        return self.encoder(
            pixel_values=pixel_values,
            token_budget=token_budget,
            return_hierarchy=return_hierarchy,
        )

    def encode(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Simple-mode interface — returns just (B, N, D) tokens.

        Padding positions are zeroed so naive consumers (concat with other
        embeddings) get the right behavior even if they ignore the
        attention_mask.

        Per docs/04_revisions.md §4 (Revision B) — drop-in replacement for
        ViT/CLIP-style usage.
        """
        out = self.forward(pixel_values=pixel_values, **kwargs)
        masked = out.tokens * out.attention_mask.unsqueeze(-1).to(out.tokens.dtype)
        return masked

    # --- Helpers ---

    def _resolve_default_budget(self, pixel_values: torch.Tensor) -> int:
        """Auto-scale token budget by input resolution when none is given.

        See docs/04_revisions.md §3 (Revision A).
        """
        cfg = self._keural_config.tokenizer
        _, _, H, W = pixel_values.shape
        num_pixels = H * W
        if num_pixels <= 256 * 256:
            return cfg.default_token_budget
        scaled = int(round(num_pixels / (256 * 256) * cfg.default_token_budget))
        return min(cfg.max_token_budget, scaled)


class _PlaceholderEncoder(nn.Module):
    """Stand-in encoder for module #3.

    Returns a structurally-valid KeuralEncoderOutput (right shapes, right
    dtypes, valid level_ids, normalized spatial_metadata) populated with
    zeros and a small learnable parameter for non-trivial save/load tests.

    Replaced by `architecture.encoder.KeuralVisionEncoder` at module #11.
    """

    def __init__(self, config: KeuralConfig) -> None:
        super().__init__()
        self._config = config
        # One real parameter so save/load actually has something to round-trip.
        self.placeholder_param = nn.Parameter(
            torch.zeros(config.spatial.embed_dim)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        token_budget: int,
        return_hierarchy: bool = True,
    ) -> KeuralEncoderOutput:
        B = pixel_values.shape[0]
        D = self._config.spatial.embed_dim
        device = pixel_values.device

        out = empty_output(
            batch_size=B,
            max_tokens=token_budget,
            embed_dim=D,
            device=device,
            dtype=pixel_values.dtype,
        )
        # Fill `tokens` with the placeholder parameter so a loss touching
        # `tokens` would have a path to a real parameter — useful for
        # smoke-testing the training loop later.
        out.tokens = out.tokens + self.placeholder_param.view(1, 1, D)
        # Mark all token slots as real so attention_mask isn't all-zero
        # (which would make every loss undefined).
        out.attention_mask = torch.ones(B, token_budget, device=device, dtype=torch.int64)
        out.token_budget_used = torch.full((B,), token_budget, device=device, dtype=torch.int64)
        # Distribute level_ids in proportion to budget_split.
        ts = self._config.tokenizer
        K_g = max(ts.budget_floors["global"], int(round(token_budget * ts.budget_split["global"])))
        K_r = max(ts.budget_floors["region"], int(round(token_budget * ts.budget_split["region"])))
        K_g = min(K_g, token_budget)
        K_r = min(K_r, token_budget - K_g)
        K_d = token_budget - K_g - K_r
        out.level_ids[:, :K_g] = 0  # GLOBAL
        out.level_ids[:, K_g : K_g + K_r] = 1  # REGION
        out.level_ids[:, K_g + K_r :] = 2  # DETAIL
        # Pooled = mean of tokens (so save/load has effect on this field too).
        out.pooled = out.tokens.mean(dim=1)
        return out


# --- HF Auto* registration ---
# This is what makes AutoModel.from_pretrained("./out") and
# AutoConfig.from_pretrained("./out") find our classes.

AutoConfig.register("keural_vision", KeuralVisionConfig)
AutoModel.register(KeuralVisionConfig, KeuralVisionModel)
