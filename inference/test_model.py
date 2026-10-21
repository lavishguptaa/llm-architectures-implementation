"""Load a downloaded Hugging Face model from local files only (no network) and generate text.

--model is either a Hub id that was fetched with download_model.py, or a path to any local
model folder.

Usage:
    python inference/test_model.py --model Qwen/Qwen1.5-0.5B
    python inference/test_model.py --model "D:\\models\\qwen" --prompt "Hello" --max-new-tokens 60
"""

import argparse
import sys
from pathlib import Path

from download_model import default_local_dir


def resolve_model_dir(model):
    # An existing folder is used as is; otherwise treat the value as a Hub id
    path = Path(model).expanduser()

    return (path if path.is_dir() else default_local_dir(model)).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a text-generation test on a local model.")
    parser.add_argument("--model", required=True, help="Hub id (downloaded earlier) or path to a model folder")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom modeling code shipped with the model (only for models you trust)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model)

    if not (model_dir / "config.json").is_file():
        print(f"Model not found at {model_dir}", file=sys.stderr)
        print(f"Run: python inference/download_model.py {args.model}", file=sys.stderr)
        return 1

    try:
        import torch
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: pip install torch", file=sys.stderr)
        return 1

    # A text model does not need torchvision, and a mismatched install breaks the transformers import
    sys.modules["torchvision"] = None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        print(f"Could not import transformers ({exc.__class__.__name__}): {exc}", file=sys.stderr)
        return 1

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.dtype == "auto":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    print(f"Loading from {model_dir} on {device} ({dtype}) ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        print(f"Failed to load the model ({exc.__class__.__name__}): {exc}", file=sys.stderr)
        print("If the files are incomplete, run download_model.py again.", file=sys.stderr)
        return 1

    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    except Exception as exc:
        print(f"Generation failed ({exc.__class__.__name__}): {exc}", file=sys.stderr)
        return 1

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n{'=' * 50}\nPrompt : {args.prompt}\n{'=' * 50}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
