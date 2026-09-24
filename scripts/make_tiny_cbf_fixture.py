"""Create a random tiny TTT checkpoint for pipeline smoke tests only."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.chunk_size < 2:
        parser.error("--chunk-size must be at least two")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_source, use_fast=True)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=max(4096, args.chunk_size * 8),
        ttt_mode=True, ttt_layers=[0, 1], ttt_proj=True,
        ttt_lr=0.05, ttt_chunk=args.chunk_size, ttt_target="hidden_states",
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.ttt_conv.weight.zero_()
            layer.mlp.ttt_conv.weight[:, :, 2] = 0.01
            layer.mlp.ttt_proj.weight.copy_(torch.eye(config.hidden_size))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"Random smoke-test checkpoint saved to {output}; it has no learned task capability.")


if __name__ == "__main__":
    main()
