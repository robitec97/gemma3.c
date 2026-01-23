#!/usr/bin/env python3
"""Download the Gemma 3 4B IT model from Hugging Face."""

import argparse
import os
import sys

from huggingface_hub import snapshot_download


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
        help="Local directory to place the model files.",
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
    args = parse_args()
    local_dir = os.path.abspath(args.output_dir)

    print(f"Downloading {args.repo} to {local_dir}...")
    snapshot_download(
        repo_id=args.repo,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
