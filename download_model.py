#!/usr/bin/env python3
"""Download the Gemma 3 4B IT model from Hugging Face.

The official repository is gated: accept the Gemma license at
https://huggingface.co/google/gemma-3-4b-it, create an access token at
https://huggingface.co/settings/tokens, then run:

    pip install huggingface_hub
    HF_TOKEN=hf_... python download_model.py
"""

import argparse
import os
import sys

# Only what gemma3.c needs: weights, tokenizer and config (~8.6 GB)
ALLOW_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Gemma 3 model weights from Hugging Face."
    )
    parser.add_argument(
        "--repo",
        default="google/gemma-3-4b-it",
        help="Hugging Face repo ID (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--output-dir",
        default="./gemma-3-4b-it",
        help="Local directory to place the model files (default: ./gemma-3-4b-it)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional repo revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var).",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed: pip install huggingface_hub", file=sys.stderr)
        return 1

    args = parse_args()
    local_dir = os.path.abspath(args.output_dir)

    print(f"Downloading {args.repo} to {local_dir} (~8.6 GB)...")
    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=local_dir,
            revision=args.revision,
            token=args.token,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as e:  # huggingface_hub raises several error types here
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            print(
                "\nAccess denied. google/gemma-3-4b-it is a gated model:\n"
                "  1. Accept the license at https://huggingface.co/google/gemma-3-4b-it\n"
                "  2. Create a token at https://huggingface.co/settings/tokens\n"
                "  3. Run: HF_TOKEN=hf_... python download_model.py",
                file=sys.stderr,
            )
        else:
            print(f"\nDownload failed: {msg}", file=sys.stderr)
        return 1

    print("Download complete. Run:  ./gemma3 -p \"Hello!\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
