"""Load and test a Hugging Face `tokenizer_config.json` file.

Prints the tokenizer class, special tokens and context length, and renders the chat template
(if the file has one) on a sample conversation.

Usage:
    python tokenizer_utilities/tokenizer_config.py path/to/tokenizer_config.json
    python tokenizer_utilities/tokenizer_config.py path/to/tokenizer_config.json --text "Hi there"

In your own code:
    from tokenizer_config import load_tokenizer_config, test_tokenizer_config
    config = load_tokenizer_config("path/to/tokenizer_config.json")
    test_tokenizer_config(config)
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TEXT = "Hello, how are you?"
SPECIAL_TOKEN_KEYS = ["bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"]


def load_tokenizer_config(path):
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"tokenizer config not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def token_text(value):
    # A special token is either a plain string or {"content": "...", ...}
    return value.get("content") if isinstance(value, dict) else value


def render_chat_template(config, messages):
    # Needs the optional jinja2 package (already installed together with torch)
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception

    return env.from_string(config["chat_template"]).render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=token_text(config.get("bos_token")) or "",
        eos_token=token_text(config.get("eos_token")) or "",
    )


def test_tokenizer_config(config, text=DEFAULT_TEXT):
    print(f"Tokenizer class   : {config.get('tokenizer_class')}")
    print(f"Max length        : {config.get('model_max_length')}")

    for key in SPECIAL_TOKEN_KEYS:
        if config.get(key) is not None:
            print(f"{key:<18}: {token_text(config[key])!r}")

    added = config.get("added_tokens_decoder") or {}
    print(f"Added tokens      : {len(added)}")

    template = config.get("chat_template")

    if not template:
        print("Chat template     : none")
        return

    # Some files store several named templates as a list of {"name": ..., "template": ...}
    if isinstance(template, list):
        print(f"Chat template     : {len(template)} named templates ({', '.join(t['name'] for t in template)})")
        config = dict(config, chat_template=template[0]["template"])
    else:
        print("Chat template     : present")

    messages = [{"role": "user", "content": text}]

    try:
        rendered = render_chat_template(config, messages)
    except Exception as exc:
        print(f"Could not render the chat template ({exc.__class__.__name__}): {exc}")
        return

    print(f"\nRendered chat template for {messages}:\n{'-' * 50}\n{rendered}\n{'-' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Load and test a tokenizer_config.json file.")
    parser.add_argument("path", help="Path to a tokenizer_config.json file")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="User message for the chat-template test")
    args = parser.parse_args()

    try:
        config = load_tokenizer_config(args.path)
    except Exception as exc:
        print(f"Could not load the config ({exc.__class__.__name__}): {exc}", file=sys.stderr)
        return 1

    test_tokenizer_config(config, args.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
