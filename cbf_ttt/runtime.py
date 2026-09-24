"""One-pass CBF inference using the existing Qwen3/LLaMA TTT modules and KV cache."""

import copy
import math

import torch
import torch.nn.functional as F


def cbf_forward_mlp(mlp, x: torch.Tensor, target: torch.Tensor, cache, layer_idx: int) -> torch.Tensor:
    """Read W0+M for output, then stage the original TTT candidate for a later shared commit."""
    if x.shape[0] != 1 or target.shape != (1, x.shape[1], mlp.hidden_size):
        raise ValueError("CBF-TTT requires batch size one and matching target states")
    h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    base = mlp.down_proj.weight
    memory = cache.cbf_memory.get(layer_idx)
    effective = base if memory is None else base + memory
    output = F.linear(h, effective, mlp.down_proj.bias)
    if cache.cbf_collect:
        if x.shape[1] != mlp.ttt_chunk:
            raise ValueError("candidate updates require exactly one complete TTT chunk")
        # This is P.T @ DWConv(target).T @ h, with the same P storage convention as baseline.
        target_conv = mlp.ttt_conv(target.transpose(1, 2)).transpose(1, 2)
        delta = target_conv[0].transpose(0, 1) @ h[0]
        if mlp.ttt_proj is not None:
            delta = mlp.ttt_proj.weight.transpose(0, 1) @ delta
        cache.cbf_candidates[layer_idx] = delta * mlp.ttt_lr
    return output


def token_statistics(ids: torch.Tensor) -> torch.Tensor:
    tokens = ids.reshape(-1).tolist()
    count = len(tokens)
    if count < 2:
        raise ValueError("a controller chunk needs at least two tokens")
    frequencies = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    entropy = -sum((n / count) * math.log(n / count) for n in frequencies.values())
    bigrams = list(zip(tokens, tokens[1:]))
    repetition = 1.0 - len(set(bigrams)) / len(bigrams)
    return torch.tensor([entropy, len(frequencies) / count, repetition], dtype=torch.float32, device=ids.device)


@torch.inference_mode()
def _nll_from_hidden(model, hidden: torch.Tensor, labels: torch.Tensor, block: int = 128) -> float:
    if hidden.shape[0] != 1 or labels.numel() != hidden.shape[1]:
        raise ValueError("NLL hidden states and labels have mismatched lengths")
    total = 0.0
    for start in range(0, labels.numel(), block):
        logits = model.lm_head(hidden[:, start : start + block]).float()
        total += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels[start : start + block], reduction="sum"
        ).item()
    return total / labels.numel()


def _memory_statistics(base: torch.Tensor, memory: torch.Tensor | None, delta: torch.Tensor) -> torch.Tensor:
    base_norm = torch.linalg.vector_norm(base.float()).clamp_min(1e-8)
    delta_float = delta.float()
    delta_norm = torch.linalg.vector_norm(delta_float)
    if memory is None:
        memory_norm = base_norm.new_zeros(())
        alignment = base_norm.new_zeros(())
    else:
        memory_float = memory.float()
        memory_norm = torch.linalg.vector_norm(memory_float)
        alignment = (memory_float * delta_float).sum() / (memory_norm * delta_norm + 1e-8)
    return torch.stack((torch.log1p(memory_norm / base_norm), torch.log1p(delta_norm / base_norm), alignment))


class CBFSession:
    """Session-local KV, fast memory, and candidate buffers. The backbone remains frozen."""

    def __init__(self, model, controller=None, cache=None):
        from inference_model.hf_llama3.modeling_llama import TTTDynamicCache as LlamaCache
        from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache as QwenCache

        if not getattr(model.config, "ttt_mode", False):
            raise ValueError("CBF-TTT requires a TTT-enabled checkpoint")
        if model.training:
            raise ValueError("CBF-TTT inference and counterfactual branches require model.eval()")
        self.model = model
        self.controller = controller
        self.layers = tuple(sorted(model.config.ttt_layers))
        if not self.layers:
            raise ValueError("the checkpoint has no TTT layers")
        if any(layer_idx < 0 or layer_idx >= len(model.model.layers) or
               not hasattr(model.model.layers[layer_idx].mlp, "ttt_conv") for layer_idx in self.layers):
            raise ValueError("ttt_layers must refer to configured TTT decoder layers")
        self.chunk_size = int(model.config.ttt_chunk)
        if self.chunk_size < 2:
            raise ValueError("ttt_chunk must be at least two")
        if cache is None:
            cache_class = {
                "qwen3": QwenCache,
                "llama": LlamaCache,
            }.get(model.config.model_type)
            if cache_class is None:
                raise ValueError("CBF-TTT supports Qwen3 and LLaMA inference models")
            cache = cache_class(config=model.config)
            cache.cbf_memory = {}
            cache.cbf_candidates = {}
            cache.cbf_collect = False
            cache.cbf_enabled = True
        self.cache = cache

    @property
    def device(self):
        return next(self.model.parameters()).device

    def clone(self):
        """Deep-copy all KV and fast-memory state for an independent counterfactual branch."""
        return CBFSession(self.model, self.controller, copy.deepcopy(self.cache))

    def _forward(self, ids: list[int], collect: bool = False) -> torch.Tensor:
        if not ids:
            raise ValueError("the input token sequence is empty")
        self.cache.cbf_collect = collect
        self.cache.cbf_candidates = {}
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.model(input_ids=tokens, past_key_values=self.cache, use_cache=True)
        self.cache = output.past_key_values
        return output.last_hidden_state

    @torch.inference_mode()
    def observe(self, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(ids) != self.chunk_size:
            raise ValueError("observe requires one complete chunk")
        hidden = self._forward(ids, collect=True)
        tokens = torch.tensor(ids, dtype=torch.long, device=self.device)
        surprise = _nll_from_hidden(self.model, hidden[:, :-1], tokens[1:])
        scalars = [torch.tensor([surprise], device=self.device), token_statistics(tokens)]
        for layer_idx in self.layers:
            delta = self.cache.cbf_candidates.get(layer_idx)
            if delta is None:
                raise RuntimeError(f"TTT layer {layer_idx} did not produce a candidate update")
            base = self.model.model.layers[layer_idx].mlp.down_proj.weight
            scalars.append(_memory_statistics(base, self.cache.cbf_memory.get(layer_idx), delta))
        semantic = hidden.float().mean(dim=1).detach()
        return semantic, torch.cat(scalars).unsqueeze(0).detach()

    def commit(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if set(self.cache.cbf_candidates) != set(self.layers):
            raise RuntimeError("a complete pending chunk is required before commit")
        with torch.inference_mode():
            for layer_idx in self.layers:
                delta = self.cache.cbf_candidates[layer_idx]
                old = self.cache.cbf_memory.get(layer_idx)
                self.cache.cbf_memory[layer_idx] = delta if old is None else (1.0 - alpha) * old + delta
        self.cache.cbf_candidates = {}
        self.cache.cbf_collect = False

    def step(self, ids: list[int], policy: str = "controller") -> float:
        semantic, scalars = self.observe(ids)
        if policy == "baseline":
            alpha = 0.0
        elif policy == "controller":
            if self.controller is None:
                raise ValueError("controller policy requires a trained controller")
            with torch.inference_mode():
                alpha = self.controller.predict(semantic, scalars).item()
        else:
            alpha = float(policy)
        self.commit(alpha)
        return alpha

    def consume(self, ids: list[int], policy: str = "controller") -> list[float]:
        alphas = []
        for start in range(0, len(ids) - len(ids) % self.chunk_size, self.chunk_size):
            alphas.append(self.step(ids[start : start + self.chunk_size], policy))
        tail = ids[len(ids) - len(ids) % self.chunk_size :]
        if tail:
            self._forward(tail)
        return alphas

    def score_answer(self, query_ids: list[int], answer_ids: list[int]) -> float:
        if not query_ids or not answer_ids:
            raise ValueError("query_ids and answer_ids must both be nonempty")
        branch = self.clone()
        hidden = branch._forward(query_ids + answer_ids[:-1])
        labels = torch.tensor(answer_ids, dtype=torch.long, device=self.device)
        start = len(query_ids) - 1
        return _nll_from_hidden(self.model, hidden[:, start : start + len(answer_ids)], labels)

    def generate(self, query_ids: list[int], max_new_tokens: int) -> list[int]:
        if not query_ids or max_new_tokens < 1:
            raise ValueError("query_ids must be nonempty and max_new_tokens positive")
        hidden = self._forward(query_ids)
        generated = []
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for index in range(max_new_tokens):
            with torch.inference_mode():
                next_id = int(self.model.lm_head(hidden[:, -1:]).argmax(dim=-1).item())
            generated.append(next_id)
            if next_id in eos_ids:
                break
            if index + 1 < max_new_tokens:
                hidden = self._forward([next_id])
        return generated
