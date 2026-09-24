"""Download the official Qwen3-4B base weights used for CBF resource profiling."""

import argparse

from huggingface_hub import snapshot_download


REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Destination for official Qwen3-4B HF files")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    path = snapshot_download(
        repo_id="Qwen/Qwen3-4B",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "merges.txt", "vocab.json"],
        max_workers=args.max_workers,
    )
    print(f"Downloaded Qwen/Qwen3-4B at {args.revision} to {path}")


if __name__ == "__main__":
    main()
