# model_theoretical

Tools that tell you how big a model is **from this repository's code alone**: no weights are downloaded, no GPU is needed, and no memory is allocated. Everything is computed on PyTorch's `meta` device, so even the 744B models are counted in seconds.

## Files

| File | Purpose |
|---|---|
| [`count_parameters.py`](count_parameters.py) | Counts the parameters (weights) of any implemented model, shows a breakdown by component, and can compare the count with Hugging Face `transformers`. |
| `README.md` | This file. |

## Requirements

The same as the rest of the repository (`pip install -r requirements.txt` in the repository root: `torch` and `tokenizers`).
Only the optional `--compare-hf` flag needs one more package: `pip install transformers`.

## Quick start

Run these from the repository root. Put model folders that contain spaces or parentheses in quotes.

```bash
# One model: a breakdown by component
python model_theoretical/count_parameters.py "Ai2/OLMo 2 (7B)"

# The same, plus a comparison with the Hugging Face model
python model_theoretical/count_parameters.py "Ai2/OLMo 2 (7B)" --compare-hf

# Every implemented model, one line each
python model_theoretical/count_parameters.py --all

# Which folders can be counted
python model_theoretical/count_parameters.py --list
```

You can also run it from any other directory, using the full path to the script. The model folder is read relative to the repository root, or as any path that exists:

```bash
python "D:\llm_lib\github_push_code\model_theoretical\count_parameters.py" "Google/Gemma 3/Gemma 3 (270M)"
```

## Commands

| Command | What it does |
|---|---|
| `count_parameters.py "<model folder>"` | Prints the breakdown and the total for one model. |
| `... --compare-hf` | Also builds the Hugging Face model from the folder's `config.json` and compares the two totals. |
| `count_parameters.py --all` | Counts every implemented model and prints one table (about 5 minutes for all 90 on the machine this was written on). Progress goes to stderr, the table to stdout, so `> counts.txt` saves only the table. |
| `count_parameters.py --all --compare-hf` | The same table with a Hugging Face column (slower). Exit status 1 if any model differs; models with a vision tower or other unimplemented parts are expected to differ. |
| `count_parameters.py --list` | Lists the 90 model folders that have a `model.py`. |

## Reading the output

```text
Ai2/OLMo 2 (7B)  (layer / expert numbers are shown as *)

Parameter group                Tensors  Shape                     Parameters    Share
----------------------------------------------------------------------------------------
tok_emb.weight                       1  (100352, 4096)           411,041,792    5.63%
blocks.*.attn.W_q.weight            32  (4096, 4096)             536,870,912    7.36%
blocks.*.attn.W_k.weight            32  (4096, 4096)             536,870,912    7.36%
blocks.*.attn.W_v.weight            32  (4096, 4096)             536,870,912    7.36%
blocks.*.attn.q_norm.weight         32  (4096,)                      131,072    0.00%
blocks.*.attn.k_norm.weight         32  (4096,)                      131,072    0.00%
blocks.*.attn.out_proj.weight       32  (4096, 4096)             536,870,912    7.36%
blocks.*.layer1.weight              32  (4096,)                      131,072    0.00%
blocks.*.layer2.weight              32  (4096,)                      131,072    0.00%
blocks.*.ffn.gate_proj.weight       32  (11008, 4096)          1,442,840,576   19.77%
blocks.*.ffn.up_proj.weight         32  (11008, 4096)          1,442,840,576   19.77%
blocks.*.ffn.down_proj.weight       32  (4096, 11008)          1,442,840,576   19.77%
out_norm_layer.weight                1  (4096,)                        4,096    0.00%
out_proj.weight                      1  (100352, 4096)           411,041,792    5.63%
----------------------------------------------------------------------------------------
Total                              355                         7,298,617,344  100.00%

Memory for the weights: 29.19 GB in float32, 14.60 GB in bfloat16/float16, 7.30 GB in int8
Not counted as weights: 1,048,576 buffer elements (RoPE tables, masks, router biases, ...)
(built and counted in 6.5 s, nothing was allocated)

Hugging Face model: 7,298,617,344 parameters -> match
```

- **Parameter group**: the parameter names as they appear in the model, with layer and expert numbers replaced by `*`. One row therefore stands for the same tensor in every layer (and, for mixture-of-experts models, in every expert).
- **Tensors**: how many tensors the row covers. **Shape**: the shape of those tensors, or `N shapes` when the row mixes different shapes (for example layers with different head counts).
- **Parameters / Share**: the number of weights in the row and its share of the total.
- **Memory for the weights**: the total multiplied by 4, 2 or 1 bytes. This is only the weights: activations and the KV cache come on top at run time.
- **Not counted as weights**: *buffers* such as the RoPE sine/cosine tables, causal masks and router bias vectors. They are not trained and Hugging Face does not count them as parameters either.

What the total includes:

- Weights, biases and norm scales are all counted.
- A tensor shared between two layers (tied weights, for example the embedding and the output head of Gemma and Qwen3 0.6B) is counted **once**.
- Only what this repository implements is counted. Multi-token-prediction layers and vision / audio towers are not implemented, so they are not in the total.

## Comparing with Hugging Face (`--compare-hf`)

The script builds the Hugging Face model for the folder's `config.json` (also on the `meta` device, with `trust_remote_code=False`: no downloaded code is ever executed) and compares the two totals.

| Result | Meaning |
|---|---|
| `-> match` | The two totals are identical, exit status 0. |
| `-> DIFFERENT` | The totals differ, exit status 1. Often this is expected: Hugging Face's model contains parts this repository does not implement. Example: Gemma 3 (4B) reports 4,300,079,472 on the Hugging Face side and 3,880,263,168 here. The difference of 419,816,304 is exactly Hugging Face's vision tower (416,866,032) plus its multimodal projector (2,950,272); without them the totals are equal. |
| `comparison skipped: ...` | The installed `transformers` cannot build that model (not released yet, custom-code only, or the folder has no Hugging Face style `config.json`, as for Mistral Large 3). The count itself is still printed. |

## How it works

1. It imports the folder's `model.py`.
2. It runs that file's own `main()` with the `Model` class replaced by a stand-in that only records the arguments `main()` passes (the configuration), so nothing is built and no tokenizer is loaded.
3. It builds the real `Model` with those arguments on the `meta` device and counts the parameters.

Because of step 2, the sizes are exactly the ones the model's own demo uses, and any model that follows the repository layout (a `model.py` with a `Model` class and a `main()` that builds a configuration and calls `Model(cfg)`) works automatically, including models you add later.

## Limitations

- Only folders with a `model.py` can be counted. Folders that only have a `config.json` and a tokenizer are not implemented yet, so there is nothing to count.
- "Theoretical" means the count comes from the code's structure: no checkpoint file is read, so this does not tell you anything about real pretrained weights.
- The count is for the text model as implemented here; published model sizes (for example "744B") can include parts that are not implemented.

## Exit status

| Status | Meaning |
|---|---|
| `0` | Success (and, with `--compare-hf`, the totals match or the comparison was skipped). |
| `1` | `--compare-hf` found a different total. |
| `2` | Usage error (no model given, folder not found) or the model could not be built. |

## Counts for all models

`python model_theoretical/count_parameters.py --all` counted all 90 implemented models in 288 seconds with no failures. The range is 268,098,176 parameters (Gemma 3 270M) up to 743,911,199,232 (GLM-5.1, 744B). The first lines of the table look like this:

```text
Model                                                Parameters     float32    bfloat16
---------------------------------------------------------------------------------------
Ai2/OLMo 2 (7B)                                   7,298,617,344     29.2 GB     14.6 GB
Ai2/OLMo 3 (32B)                                 32,233,522,176    128.9 GB     64.5 GB
Ai2/OLMo 3 (7B)                                   7,298,011,136     29.2 GB     14.6 GB
Alibaba/Qwen1.5/Dense/Qwen1.5-0.5B                  463,987,712      1.9 GB      0.9 GB
Alibaba/Qwen1.5/Dense/Qwen1.5-1.8B                1,836,828,672      7.3 GB      3.7 GB
```
