"""Overlay TTT modules on local Qwen3/LLaMA weights for cost profiling only.

The inserted TTT projection is untrained and its depthwise conv is zero. This
fixture cannot measure CBF quality; it preserves parameter and cache shapes.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Local HF Qwen3 or LLaMA checkpoint directory")
    parser.add_argument("--output", required=True, help="New directory for symlinks and TTT config")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--ttt-layers", nargs="+", type=int, default=[0, 6, 12, 18, 24, 30])
    parser.add_argument("--ttt-lr", type=float, default=0.3)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    output = Path(args.output)
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in {"qwen3", "llama"}:
        parser.error("--base must contain a Qwen3 or LLaMA config")
    if args.chunk_size < 2 or not args.ttt_layers or len(set(args.ttt_layers)) != len(args.ttt_layers):
        parser.error("chunk size and TTT layers must be valid")
    if any(layer < 0 or layer >= config["num_hidden_layers"] for layer in args.ttt_layers):
        parser.error("TTT layer index is outside the base model")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be a new or empty directory")
    if not list(base.glob("*.safetensors")):
        parser.error("--base has no safetensors weight files")
    if not (base / "model.safetensors.index.json").exists():
        parser.error("--base needs a model.safetensors.index.json shard index")
    output.mkdir(parents=True, exist_ok=True)
    config.update({
        "ttt_mode": True, "ttt_layers": sorted(args.ttt_layers),
        "ttt_chunk": args.chunk_size, "ttt_lr": args.ttt_lr,
        "ttt_proj": True, "ttt_target": "hidden_states",
    })
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for pattern in ("*.safetensors", "*.safetensors.index.json", "tokenizer*", "special_tokens_map.json", "generation_config.json"):
        for source in base.glob(pattern):
            if source.is_file():
                (output / source.name).symlink_to(source)
    # HF's missing-weight initialization can leave depthwise conv tensors nonfinite
    # when loading via the meta-device path. Store explicit zero updates for sizing.
    ttt_weights = {
        f"model.layers.{layer}.mlp.ttt_conv.weight": torch.zeros(
            config["hidden_size"], 1, 5, dtype=torch.float16
        )
        for layer in sorted(args.ttt_layers)
    }
    save_file(ttt_weights, str(output / "resource_ttt.safetensors"))
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"].update({name: "resource_ttt.safetensors" for name in ttt_weights})
        metadata = index.setdefault("metadata", {})
        metadata["total_size"] = metadata.get("total_size", 0) + sum(
            t.numel() * t.element_size() for t in ttt_weights.values()
        )
        index_path.unlink()
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Resource fixture: {output}; inserted TTT weights are untrained")


if __name__ == "__main__":
    main()
