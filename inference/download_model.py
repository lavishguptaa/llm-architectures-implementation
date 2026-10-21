"""Download any Hugging Face model into inference/hf_models/ so it can be used offline.

If the local copy is already complete the download is skipped; otherwise only the missing
or incomplete files are fetched (interrupted downloads resume).

Usage:
    python inference/download_model.py Qwen/Qwen1.5-0.5B
    python inference/download_model.py Qwen/Qwen1.5-0.5B --local-dir D:\\models\\qwen
    python inference/download_model.py org/private-model --token hf_xxx --revision main
"""

import argparse
import sys
from pathlib import Path

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "hf_models"

# Only what is needed to load a model with transformers (skip .bin / .h5 / .msgpack duplicates)
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.py"]

# Used when the Hub cannot be reached, to decide whether the local copy is usable
REQUIRED_FILES = ["config.json"]


def default_local_dir(model_id):
    # "org/name" -> inference/hf_models/org__name
    return DEFAULT_MODELS_DIR / model_id.replace("/", "__")


def matches_allowed(name):
    return any(Path(name).match(pattern) for pattern in ALLOW_PATTERNS)


def remote_manifest(api, model_id, revision, token):
    # {filename: size_in_bytes} for the files we want, as listed on the Hub
    info = api.model_info(model_id, revision=revision, token=token, files_metadata=True)

    return {
        s.rfilename: s.size
        for s in info.siblings
        if matches_allowed(s.rfilename) and s.size is not None
    }


def missing_files(manifest, local_dir):
    # Files that are absent or whose size differs from the Hub copy
    return [
        name
        for name, size in manifest.items()
        if not (local_dir / name).is_file() or (local_dir / name).stat().st_size != size
    ]


def has_weights(local_dir):
    return any(local_dir.glob("*.safetensors"))


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model for offline use.")
    parser.add_argument("model_id", help="Hub repo id, for example Qwen/Qwen1.5-0.5B")
    parser.add_argument("--local-dir", type=Path, default=None,
                        help="Where to store the model (default: inference/hf_models/<org>__<name>)")
    parser.add_argument("--revision", default=None, help="Branch, tag or commit to download")
    parser.add_argument("--token", default=None, help="Access token for gated or private models")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    local_dir = (args.local_dir or default_local_dir(args.model_id)).expanduser().resolve()

    print(f"Model      : {args.model_id}")
    print(f"Local path : {local_dir}")

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = remote_manifest(HfApi(), args.model_id, args.revision, args.token)
    except Exception as exc:
        # Offline or Hub unavailable: fall back to checking the essential files only
        print(f"Could not reach the Hugging Face Hub ({exc.__class__.__name__}: {exc})")

        if all((local_dir / f).is_file() for f in REQUIRED_FILES) and has_weights(local_dir):
            print("Local copy looks complete; nothing to download.")
            return 0

        print("No complete local copy found. Check the model id / network and try again.", file=sys.stderr)
        return 1

    todo = missing_files(manifest, local_dir)

    if not todo:
        print(f"Model is already complete ({len(manifest)} files); skipping download.")
        return 0

    print(f"{len(todo)} of {len(manifest)} files missing or incomplete: {', '.join(todo)}")
    print("Downloading ...")

    try:
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            token=args.token,
            local_dir=str(local_dir),
            allow_patterns=ALLOW_PATTERNS,
        )
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the script again to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Download failed ({exc.__class__.__name__}): {exc}", file=sys.stderr)
        return 1

    still_missing = missing_files(manifest, local_dir)

    if still_missing:
        print(f"Download finished but these files are still incomplete: {', '.join(still_missing)}", file=sys.stderr)
        return 1

    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
