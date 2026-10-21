# Tokenizer utilities

Small, standalone scripts for loading and testing tokenizer files. There is one script per file type, so each one stays simple.

| Script | Tests this file | What it does |
|---|---|---|
| [`tokenizer_json.py`](tokenizer_json.py) | `tokenizer.json` | Loads the tokenizer, then encodes and decodes a text and checks the round trip |
| [`tokenizer_config.py`](tokenizer_config.py) | `tokenizer_config.json` | Prints the tokenizer class, special tokens and context length, and renders the chat template |

## Requirements

```bash
pip install tokenizers jinja2
```

- `tokenizers` is needed by `tokenizer_json.py` (it is already in the repository's `requirements.txt`).
- `jinja2` is needed by `tokenizer_config.py` only to render the chat template. It is installed together with `torch`, so you usually have it already.

Python 3.10 or newer is recommended.

## How to run

Run every command from the **repository root** (the folder that contains `README.md`). Put paths that contain spaces or parentheses in double quotes. Windows paths (`D:\...`) and forward slashes both work.

### `tokenizer_json.py`: test a `tokenizer.json`

Every model folder in this repository already contains a `tokenizer.json`.

```bash
# Basic test with the default text ("Hello, how are you?")
python tokenizer_utilities/tokenizer_json.py "Alibaba/qwen3/Qwen3 (0.6B)/tokenizer.json"

# Your own text
python tokenizer_utilities/tokenizer_json.py "Alibaba/qwen3/Qwen3 (0.6B)/tokenizer.json" --text "Tokenizers split text into pieces."

# A file outside the repository (Windows)
python tokenizer_utilities/tokenizer_json.py "D:\models\my-model\tokenizer.json" --text "Hello"

# See all options
python tokenizer_utilities/tokenizer_json.py --help
```

Example output:

```text
Vocabulary size : 151669
Text            : 'Hello, how are you?'
Tokens          : ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
Token ids       : [9707, 11, 1246, 525, 498, 30]
Decoded         : 'Hello, how are you?'
Round trip      : ok
```

`Round trip : differs` can be normal: some tokenizers lowercase text, add spaces or normalise characters, so the decoded text is not always identical to the input.

### `tokenizer_config.py`: test a `tokenizer_config.json`

The model folders in this repository do **not** include a `tokenizer_config.json`. To get one, download a model with the shared script in [`inference/`](../inference) (it also fetches the tokenizer files):

```bash
python inference/download_model.py Qwen/Qwen1.5-0.5B
```

The files are stored in `inference/hf_models/Qwen__Qwen1.5-0.5B/`. Then run:

```bash
# Basic test: special tokens, context length and the chat template on a sample message
python tokenizer_utilities/tokenizer_config.py "inference/hf_models/Qwen__Qwen1.5-0.5B/tokenizer_config.json"

# Your own user message for the chat-template test
python tokenizer_utilities/tokenizer_config.py "inference/hf_models/Qwen__Qwen1.5-0.5B/tokenizer_config.json" --text "What is a tokenizer?"

# A file outside the repository (Windows)
python tokenizer_utilities/tokenizer_config.py "D:\models\my-model\tokenizer_config.json"

# See all options
python tokenizer_utilities/tokenizer_config.py --help
```

Example output (for a config with a ChatML-style template):

```text
Tokenizer class   : PreTrainedTokenizerFast
Max length        : 32768
eos_token         : '<|im_end|>'
pad_token         : '<|endoftext|>'
Added tokens      : 2
Chat template     : present

Rendered chat template for [{'role': 'user', 'content': 'Hi there'}]:
--------------------------------------------------
<|im_start|>user
Hi there<|im_end|>
<|im_start|>assistant

--------------------------------------------------
```

If a model's template uses helper functions this script does not provide, it prints `Could not render the chat template` with the reason, and the rest of the output is still shown.

## Use the functions in your own code

Add the folder to `sys.path` (or run your code from inside `tokenizer_utilities/`), then import:

```python
import sys
sys.path.insert(0, "tokenizer_utilities")

from tokenizer_json import load_tokenizer, test_tokenizer
from tokenizer_config import load_tokenizer_config, test_tokenizer_config

tokenizer = load_tokenizer("Alibaba/qwen3/Qwen3 (0.6B)/tokenizer.json")
test_tokenizer(tokenizer, "Hello")

config = load_tokenizer_config("inference/hf_models/Qwen__Qwen1.5-0.5B/tokenizer_config.json")
test_tokenizer_config(config, "Hello")
```

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `tokenizer file not found` / `tokenizer config not found` | The path is wrong. Run from the repository root and quote paths that contain spaces or parentheses. |
| `ModuleNotFoundError: No module named 'tokenizers'` | Run `pip install tokenizers`. |
| `ModuleNotFoundError: No module named 'jinja2'` | Run `pip install jinja2` (only needed for the chat-template test). |
| `Could not load the tokenizer` | The file is not a valid `tokenizer.json`, or it is damaged. Download it again. |

## Adding a new script

Add it next to the existing ones, follow the same pattern (a `load_*` function, a `test_*` function and a small command-line `main()` with `argparse`), and add a row to the table at the top of this file.
