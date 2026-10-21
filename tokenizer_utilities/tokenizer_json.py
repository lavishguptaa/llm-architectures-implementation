"""Load and test a Hugging Face `tokenizer.json` file with the `tokenizers` library.

Usage:
    python tokenizer_utilities/tokenizer_json.py "Alibaba/qwen3/Qwen3 (0.6B)/tokenizer.json"
    python tokenizer_utilities/tokenizer_json.py path/to/tokenizer.json --text "Hello, how are you?"

In your own code:
    from tokenizer_json import load_tokenizer, test_tokenizer
    tokenizer = load_tokenizer("path/to/tokenizer.json")
    test_tokenizer(tokenizer, "Hello")
"""

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_TEXT = "Hello, how are you?"


def load_tokenizer(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"tokenizer file not found: {path}")

    return Tokenizer.from_file(str(path))


def test_tokenizer(tokenizer, text=DEFAULT_TEXT):
    encoding = tokenizer.encode(text)
    decoded = tokenizer.decode(encoding.ids)

    print(f"Vocabulary size : {tokenizer.get_vocab_size()}")
    print(f"Text            : {text!r}")
    print(f"Tokens          : {encoding.tokens}")
    print(f"Token ids       : {encoding.ids}")
    print(f"Decoded         : {decoded!r}")
    print(f"Round trip      : {'ok' if decoded == text else 'differs (normal for some tokenizers)'}")

    return encoding


def main():
    parser = argparse.ArgumentParser(description="Load and test a tokenizer.json file.")
    parser.add_argument("path", help="Path to a tokenizer.json file")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to encode and decode")
    args = parser.parse_args()

    try:
        tokenizer = load_tokenizer(args.path)
    except Exception as exc:
        print(f"Could not load the tokenizer ({exc.__class__.__name__}): {exc}", file=sys.stderr)
        return 1

    test_tokenizer(tokenizer, args.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
