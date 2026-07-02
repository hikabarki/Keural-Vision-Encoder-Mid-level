"""
Distributed helpers for multi-GPU contrastive training.

Two responsibilities:

1. ``gather_with_grad`` — an all-gather that preserves gradients for the
   *local* rank's slice. ``torch.distributed.all_gather`` is not
   differentiable (the gathered copies from other ranks are detached and the
   local copy loses its graph), so SigLIP-style contrastive losses computed on
   the gathered tensor would receive no gradient. This wrapper re-inserts the
   local rank's original (grad-carrying) tensor into the gathered list so the
   backward pass flows correctly into the local embeddings.

2. ``broadcast_module_params`` / ``all_reduce_grads`` — keep parameters that
   live *outside* DDP (the loss module and the text projection) identical
   across ranks. DDP only synchronises the module it wraps; these helpers cover
   everything else that is trainable.

All functions are no-ops when distributed is not initialised, so the same code
path works for single-GPU and CPU runs.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


class _GatherWithGrad(torch.autograd.Function):
    """Differentiable all-gather along dim 0.

    Forward: returns the concatenation of every rank's tensor.
    Backward: each rank keeps only the gradient slice that corresponds to the
    samples it contributed (the other slices are reduced away). We all-reduce
    the incoming gradient so every rank agrees, then slice out this rank's part.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.local_batch = tensor.shape[0]
        ctx.rank = get_rank()
        world_size = get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        # Re-insert the local, grad-carrying tensor so autograd has a path.
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # grad_output covers the full (world_size * local_batch) concatenation.
        # All-reduce so every rank sees the same gradient, then take this rank's
        # slice. (all_gather replicates the forward value; the matching backward
        # is a reduce-scatter, which we implement as all_reduce + slice.)
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.local_batch
        end = start + ctx.local_batch
        return grad_input[start:end]


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather ``tensor`` along dim 0, preserving gradient for the local slice.

    Returns ``tensor`` unchanged when not running distributed.
    """
    if not is_dist() or get_world_size() == 1:
        return tensor
    return _GatherWithGrad.apply(tensor)


@torch.no_grad()
def broadcast_module_params(module: nn.Module, src: int = 0) -> None:
    """Broadcast all params and buffers of ``module`` from ``src`` to all ranks.

    Used once at startup so modules that are not wrapped in DDP (loss, text
    projection) start identical on every rank.
    """
    if not is_dist() or get_world_size() == 1:
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


@torch.no_grad()
def all_reduce_grads(module: nn.Module) -> None:
    """Average gradients of ``module`` across ranks.

    Call after ``backward`` and before ``optimizer.step`` for modules that are
    not wrapped in DDP (DDP already averages the module it wraps). Keeps the
    loss/text parameters identical across ranks instead of letting them diverge.
    """
    if not is_dist() or get_world_size() == 1:
        return
    world_size = get_world_size()
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size
