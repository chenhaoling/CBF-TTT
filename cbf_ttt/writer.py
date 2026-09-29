"""Opt-in task-trained transform of frozen In-Place TTT candidates."""

import torch
import torch.nn.functional as F
from torch import nn

from .runtime import CBFSession


class LowRankWriter(nn.Module):
    """Transform current-content updates with a per-layer Frobenius norm cap."""

    def __init__(self, shapes: dict[int, tuple[int, int]], rank: int = 8):
        super().__init__()
        if not shapes or rank < 1:
            raise ValueError("layer shapes and a positive rank are required")
        self.shapes = {int(layer): tuple(shape) for layer, shape in shapes.items()}
        self.rank = rank
        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()
        for layer, (rows, _) in self.shapes.items():
            self.a[str(layer)] = nn.Parameter(torch.randn(rank, rows) / rows ** 0.5)
            self.b[str(layer)] = nn.Parameter(torch.zeros(rows, rank))

    def forward(self, candidates: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if set(candidates) != set(self.shapes):
            raise ValueError("writer and candidate layers differ")
        result = {}
        for layer, shape in self.shapes.items():
            delta = candidates[layer].detach().float()
            if tuple(delta.shape) != shape:
                raise ValueError("candidate shape differs from writer checkpoint")
            value = delta + self.b[str(layer)] @ (self.a[str(layer)] @ delta)
            # Cap every layer separately; the writer cannot gain solely by increasing update norm.
            factor = (delta.norm() / value.norm().clamp_min(1e-12)).clamp(max=1.0)
            result[layer] = value * factor
        return result


def memory_nll(model, memory: dict[int, torch.Tensor], query_ids: list[int],
               answer_ids: list[int]) -> torch.Tensor:
    """Read a fresh KV cache while preserving gradients from loss to supplied memory."""
    if not query_ids or not answer_ids:
        raise ValueError("query and answer must be nonempty")
    session = CBFSession(model)
    dtype = next(model.parameters()).dtype
    session.cache.cbf_memory = {layer: value.to(dtype) for layer, value in memory.items()}
    ids = torch.tensor([query_ids + answer_ids[:-1]], device=session.device, dtype=torch.long)
    hidden = model.model(input_ids=ids, past_key_values=session.cache, use_cache=True).last_hidden_state
    logits = model.lm_head(hidden[:, len(query_ids) - 1:]).float()
    target = torch.tensor(answer_ids, device=session.device, dtype=torch.long)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target)
