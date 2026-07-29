"""
Keural Vision Encoder — VLM Helpers.

The `build_vlm_inputs` utility — the recommended path for plugging Keural
into LLaVA-style VLMs. Handles concatenation of vision and text token
embeddings, attention masks, and position IDs correctly when token counts
vary across the batch.

Per docs/04_revisions.md §7 (Revision E).

Why this utility exists
-----------------------
Variable-length sequences are where naive integrations break SILENTLY.
If a user concats vision tokens and text tokens without a correct
attention mask, the LLM produces output but with attention reaching
into padding. Models trained this way produce confident garbage.

This utility prevents that entire class of bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from alignment.output_contract import KeuralEncoderOutput
from alignment.projectors import KeuralProjectorBase


@dataclass
class VLMInputs:
    """Bundle of inputs ready to feed an LLM in a VLM pipeline.

    Attributes
    ----------
    inputs_embeds : Tensor of shape (B, vision_N + text_N, llm_dim).
        Concatenated [vision_embeds | text_embeds] along sequence dim.
    attention_mask : Tensor of shape (B, vision_N + text_N), int64.
        Combined mask: 1 for real tokens, 0 for padding.
    position_ids : Tensor of shape (B, vision_N + text_N), int64.
        Sequential 0..L-1 for each batch item, restarting per item.
        For padding positions the value is still sequential — the LLM
        ignores it via the attention mask.
    vision_token_count : Tensor of shape (B,), int64.
        Number of vision tokens per batch item. Useful for users who need
        to slice out the vision portion of LLM hidden states later.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    vision_token_count: torch.Tensor


def build_vlm_inputs(
    encoder_output: KeuralEncoderOutput,
    projector: KeuralProjectorBase,
    text_embeds: torch.Tensor,
    text_attention_mask: torch.Tensor,
) -> VLMInputs:
    """Build LLM inputs from encoder output + projector + text embeddings.

    Parameters
    ----------
    encoder_output
        Output of a Keural Vision Encoder forward pass.
    projector
        A KeuralProjectorBase instance (Simple/LevelAware/Metadata).
    text_embeds
        Shape (B, text_N, llm_dim). Token embeddings produced by the
        LLM's input embedding layer (e.g., `llm.get_input_embeddings()`).
    text_attention_mask
        Shape (B, text_N), int or bool. 1 for real text tokens, 0 for padding.

    Returns
    -------
    VLMInputs ready to feed `llm(inputs_embeds=..., attention_mask=..., position_ids=...)`.

    Notes
    -----
    The vision portion comes BEFORE the text portion in the sequence. This
    matches LLaVA's convention. Place a sentinel like "<image>" in your
    prompt to mark the splice point, then construct text_embeds for
    everything except the sentinel.
    """
    if encoder_output.tokens.shape[0] != text_embeds.shape[0]:
        raise ValueError(
            f"Batch size mismatch: encoder {encoder_output.tokens.shape[0]} "
            f"vs text {text_embeds.shape[0]}"
        )
    if encoder_output.tokens.shape[0] != text_attention_mask.shape[0]:
        raise ValueError(
            f"Batch size mismatch in text_attention_mask: {text_attention_mask.shape[0]}"
        )

    B = encoder_output.tokens.shape[0]
    vision_N = encoder_output.tokens.shape[1]
    text_N = text_embeds.shape[1]
    device = text_embeds.device

    # Project vision tokens to LLM dim
    vision_embeds = projector(encoder_output)  # (B, vision_N, llm_dim)

    # Move to the same device/dtype as text_embeds for clean concat
    vision_embeds = vision_embeds.to(device=device, dtype=text_embeds.dtype)

    # Concat embeddings along sequence dim
    inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

    # Build combined attention mask
    vision_mask = encoder_output.attention_mask.to(device=device, dtype=torch.int64)
    text_mask_int = text_attention_mask.to(device=device, dtype=torch.int64)
    attention_mask = torch.cat([vision_mask, text_mask_int], dim=1)

    # Build position IDs. We use simple sequential 0..L-1 within each
    # batch item. Some LLMs (RoPE-based) compute their own positions
    # from this — anything more clever is left to the LLM.
    L = vision_N + text_N
    position_ids = torch.arange(L, device=device, dtype=torch.int64).unsqueeze(0).expand(B, L).contiguous()

    # Vision token count per item — sum the vision portion of the mask.
    vision_token_count = vision_mask.sum(dim=1)

    return VLMInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        vision_token_count=vision_token_count,
    )
