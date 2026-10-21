# Tokenizer utilities

Small scripts for loading and testing tokenizer files. One script per file type, so each stays simple.

| Script | File type | Status |
|---|---|---|
| [`tokenizer_json.py`](tokenizer_json.py) | `tokenizer.json` | ✅ Available |
| `tokenizer_config.py` | `tokenizer_config.json` | Planned |
| `vocab_json.py` | `vocab.json` | Planned |
| `merges_txt.py` | `merges.txt` | Planned |

Each script follows the same pattern: a `load_*` function, a `test_*` function, and a small command-line `main()`. Add new scripts here next to `tokenizer_json.py` and list them in the table above.

```bash
python tokenizer_utilities/tokenizer_json.py "Alibaba/qwen3/Qwen3 (0.6B)/tokenizer.json" --text "Hello"
```
