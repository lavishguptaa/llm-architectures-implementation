<div align="center">

# LLM Architectures Implementation

**Readable, from-scratch PyTorch implementations of modern large language model architectures — one self-contained file per model.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c)
![Models implemented](https://img.shields.io/badge/models%20implemented-85-brightgreen)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Overview

This repository is a growing library of decoder-only LLM architectures written in plain PyTorch, without depending on `transformers` for the model code. Each model lives in its own folder with the pieces needed to read, run and compare it:

| File | Purpose |
|---|---|
| `model.py` | Self-contained PyTorch implementation (attention, RoPE, norms, feed-forward / MoE, full model, greedy generation, and a runnable `main()` demo) |
| `config.json` | The model's official configuration, as published by its authors |
| `tokenizer.json` | The model's Hugging Face tokenizer file, loaded with the `tokenizers` library |
| `*.webp` / `*.png` | Architecture diagram (only some folders; in `assets/` for the Ai2, BOSS Zhipin and GLM models, next to `model.py` elsewhere) |

The goal is to make the differences *between* architectures easy to see: the same building blocks (RoPE, RMSNorm, SwiGLU, GQA) recombined in different ways, plus the genuinely new ideas (sliding-window layer patterns, logit soft-capping, QK-norm, YaRN / `llama3` RoPE scaling, sparse Mixture-of-Experts routing).

> [!IMPORTANT]
> **Project status: work in progress.** Many of the architectures have been checked against a reference implementation (see [Verification status](#verification-status)), but **none has been validated with real pretrained weights**, and a number of the older implementations are still unverified. See [Known limitations](#known-limitations) for issues already identified.

## At a glance

| | Count |
|---|---|
| ✅ Models with a `model.py` implementation | **85** |
| 🚧 Models with only `config.json` + `tokenizer.json` (implementation pending) | **25** |
| 📁 Reserved folders (no config or code yet) | **1** |
| Standalone attention-mechanism implementations | **9** files |

Implemented families: **OLMo 2 / 3, Qwen 1.5 / 2.5 / 3 (dense and MoE), DeepSeek V2 / V3, Nanbeige 4.1, Cisco Antares, GLM-4.5 / 4.5-Air / 4.7 / 5 / 5.1 / 5.2, INTELLECT-3, Step 3.5 Flash, Sarvam 30B, Laguna XS.2 / XS 2.1 / S 2.1, GPT-OSS 20B / 120B, Gemma 1 / 2 / 3, Granite 4.1, LLaMA 1 / 2 / 3 / 3.1 / 3.2 / 4 Maverick, Phi-1 / 1.5 / 4, Mistral Small 3.1 / 4 and Mistral Large 3, SmolLM3, MiniMax M2 / M2.5 / M2.7, Cohere Tiny Aya / North Mini Code / Command A+, VibeThinker-3B.**

## Highlights

- **One file per model.** Open a folder, read `model.py` top to bottom, and you have the whole architecture — no framework indirection, no inheritance maze.
- **Configuration-driven.** Each model is built from a plain `cfg` dictionary that mirrors the official `config.json` keys, so adapting a model to a new size is a config change.
- **Architectural features covered**
  - Positional encoding: RoPE (half-split and interleaved pairing), `llama3` frequency scaling (LLaMA 3.1 / 3.2, Step 3.5 Flash), YaRN (OLMo 3, DeepSeek, Mistral Small 4 / Large 3, GPT-OSS, Laguna), partial rotary embeddings (GLM-4.x, Step, Laguna, MiniMax), per-layer-type RoPE settings (Gemma 3, Step, Laguna), layers with no positional encoding at all (LLaMA 4, SmolLM3, Cohere)
  - Attention: MHA, GQA, **multi-head latent attention** (DeepSeek, GLM-5, Mistral Small 4 / Large 3), **DeepSeek Sparse Attention** with a lightning indexer and cross-layer indexer sharing (GLM-5 / 5.1 / 5.2), sliding-window and chunked-attention layer patterns (Gemma 2/3, OLMo 3, GPT-OSS, Step, Laguna, LLaMA 4), per-head attention sinks (GPT-OSS), per-head output gates (Step, Laguna), per-layer head counts, QK-norm (per head, or over the whole projection in MiniMax), attention-logit and final-logit soft-capping (Gemma 2), position-dependent query scaling (LLaMA 4, Mistral Small 4 / Large 3)
  - Normalisation and scaling: RMSNorm (including zero-centred and weight-free variants), LayerNorm (Phi-1 / 1.5, Cohere), pre-norm, post-norm and **parallel** (attention and FFN read the same input) block layouts, Granite's embedding / residual / logit multipliers
  - Feed-forward: gated SwiGLU-style MLPs (including clamped and biased variants), and many **sparse MoE** designs: sigmoid routing with a selection bias and group-limited top-k (GLM, DeepSeek V3, Step, Sarvam, Laguna, MiniMax), softmax routing (Qwen3, Qwen1.5 MoE, DeepSeek V2, Mistral), top-k-then-softmax (GPT-OSS), top-k-then-sigmoid with averaged shared experts (Cohere), top-1 sigmoid gating that scales the expert input (LLaMA 4), shared experts and dense leading layers
- **Attention mechanisms from first principles.** A separate [`attention-mechanisms/`](attention-mechanisms) folder implements the building blocks on their own — useful as a learning path before reading the full models.
- **Minimal dependencies:** `torch` and `tokenizers` (see [`requirements.txt`](requirements.txt)).

## Repository structure

```text
.
├── attention-mechanisms/        # Standalone attention implementations (see below)
├── Ai2/                         # OLMo 2, OLMo 3
├── Alibaba/                     # Qwen1.5, Qwen2.5, Qwen3 (dense and MoE)
├── BOSS Zhipin/                 # Nanbeige
├── Cisco/                       # Antares
├── Cohere/                      # Tiny Aya, Command A+, North Mini Code
├── Deepseek/                    # DeepSeek V2, V3, R1, V3.2, V4
├── GLM_Series/                  # GLM-4.5, 4.5-Air, 4.7, 5.x
├── Google/                      # Gemma 1, 2, 3, 4
├── Hugging Face/                # SmolLM3
├── IBM/                         # Granite
├── inference/                   # Shared scripts: download a Hub model, run a local generation test
├── Meta/                        # LLaMA 1, 2, 3, 3.1, 3.2, 4, Muse Glimmer
├── Microsoft/                   # Phi
├── MiniMax/                     # MiniMax M2 – M3
├── Mistral AI/                  # Mistral Small / Large
├── Nvidia/                      # Nemotron 3
├── OpenAI/                      # GPT-OSS
├── Poolside/                    # Laguna
├── Prime Intellect/             # INTELLECT-3
├── Sarvam AI/                   # Sarvam
├── StepFun/                     # Step 3.5 Flash
├── Weibo/                       # VibeThinker
├── Xiaomi/                      # MiMo
├── xAI/                         # Grok 2.5 (placeholder)
└── <Vendor>/<Model (Size)>/
    ├── model.py                 # (implemented models only)
    ├── config.json
    ├── tokenizer.json
    └── <diagram>.webp           # (some models; under assets/ for Ai2, BOSS Zhipin, GLM)
```

## Quick start

### 1. Install

```bash
git clone https://github.com/lavishguptaa/llm-architectures-implementation.git
cd llm-architectures-implementation
pip install -r requirements.txt
```

Python 3.10+ and a recent PyTorch (2.5+) are recommended: several models call `F.scaled_dot_product_attention(..., enable_gqa=True)`, which needs a recent release.

### 2. Run a model demo

Each `model.py` loads `tokenizer.json` by relative path, so **run it from inside its own folder**:

```bash
cd "Alibaba/qwen3/Qwen3 (0.6B)"
python model.py
```

The demo builds the model from its `cfg`, tokenizes `"Hello, how are you?"`, and greedily generates 10 tokens.

> [!WARNING]
> - **The demos use randomly initialised weights**, so the generated text is meaningless. They check that the architecture builds and runs end to end; they do not demonstrate model quality.
> - **Large configs allocate every parameter in fp32.** A 70B+ model needs hundreds of GB of memory and will not run on a typical machine. Start with the small models (Qwen3 0.6B, LLaMA 3.2 1B, Gemma 3 1B, Gemma 2B).

### 3. Use a model in your own code

```python
import sys, torch
sys.path.insert(0, "Alibaba/qwen3/Qwen3 (0.6B)")
from model import Model

cfg = {...}                      # copy the cfg dict from that folder's main()
model = Model(cfg).eval()
logits = model(torch.randint(0, cfg["vocab_size"], (1, 16)))   # [batch, seq, vocab]
```

### 4. Download and test a real Hugging Face model (optional)

Two shared scripts in [`inference/`](inference) work with any model id or local path; they use `huggingface_hub` and `transformers` (not the `model.py` files) and are separate from the from-scratch implementations:

```bash
python inference/download_model.py Qwen/Qwen1.5-0.5B          # stored in inference/hf_models/ (git-ignored)
python inference/test_model.py --model Qwen/Qwen1.5-0.5B --prompt "Hello"
python inference/test_model.py --model "D:\models\my-model"     # any local model folder
```

The download is skipped when the local copy is already complete, and `test_model.py` loads from local files only. Run either script with `--help` for the other options (`--local-dir`, `--revision`, `--token`, `--dtype`, `--device`, `--max-new-tokens`, `--trust-remote-code`).

## Model status

**Legend:** ✅ Implemented — `model.py` present with a `main()` demo that uses random weights (see [Verification status](#verification-status)) · 🚧 Config + tokenizer only — implementation pending · 📁 Placeholder — folder reserved, no usable config or code yet.

Expand a vendor to see its models. Each row links to its folder.

<details>
<summary><b>Ai2</b> &nbsp;·&nbsp; 3 implemented / 3 total</summary>


_OLMo 2: QK-norm. OLMo 3: sliding/full attention layer pattern, YaRN RoPE scaling, QK-norm, GQA (32B). OLMo 2 (7B) also has a `model_with_kv_cache.py` that generates with a KV cache._


| Model | Status | Folder |
|---|---|---|
| OLMo 2 (7B) | ✅ Implemented | [`Ai2/OLMo 2 (7B)`](Ai2/OLMo%202%20%287B%29) |
| OLMo 3 (32B) | ✅ Implemented | [`Ai2/OLMo 3 (32B)`](Ai2/OLMo%203%20%2832B%29) |
| OLMo 3 (7B) | ✅ Implemented | [`Ai2/OLMo 3 (7B)`](Ai2/OLMo%203%20%287B%29) |

</details>

<details>
<summary><b>Alibaba</b> &nbsp;·&nbsp; 23 implemented / 23 total</summary>


_Qwen1.5 / Qwen2.5 / Qwen3 dense models (RoPE, RMSNorm, SwiGLU, GQA where applicable); Qwen1.5 14B-A2.7B (softmax-routed MoE with 60 experts plus a shared expert behind a sigmoid gate); and the Qwen3 mixture-of-experts models (per-head QK-norm, softmax routing over 128 experts, top 8): Qwen3 30B-A3B, 235B-A22B and Coder Flash._


| Model | Status | Folder |
|---|---|---|
| Qwen1.5-0.5B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-0.5B`](Alibaba/Qwen1.5/Dense/Qwen1.5-0.5B) |
| Qwen1.5-1.8B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-1.8B`](Alibaba/Qwen1.5/Dense/Qwen1.5-1.8B) |
| Qwen1.5-110B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-110B`](Alibaba/Qwen1.5/Dense/Qwen1.5-110B) |
| Qwen1.5-14B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-14B`](Alibaba/Qwen1.5/Dense/Qwen1.5-14B) |
| Qwen1.5-32B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-32B`](Alibaba/Qwen1.5/Dense/Qwen1.5-32B) |
| Qwen1.5-4B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-4B`](Alibaba/Qwen1.5/Dense/Qwen1.5-4B) |
| Qwen1.5-72B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-72B`](Alibaba/Qwen1.5/Dense/Qwen1.5-72B) |
| Qwen1.5-7B | ✅ Implemented | [`Alibaba/Qwen1.5/Dense/Qwen1.5-7B`](Alibaba/Qwen1.5/Dense/Qwen1.5-7B) |
| Qwen1.5(14B-A2.7B) | ✅ Implemented | [`Alibaba/Qwen1.5/MOE/Qwen1.5(14B-A2.7B)`](Alibaba/Qwen1.5/MOE/Qwen1.5%2814B-A2.7B%29) |
| Qwen2.5-0.5B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-0.5B`](Alibaba/qwen2.5/Qwen2.5-0.5B) |
| Qwen2.5-1.5B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-1.5B`](Alibaba/qwen2.5/Qwen2.5-1.5B) |
| Qwen2.5-14B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-14B`](Alibaba/qwen2.5/Qwen2.5-14B) |
| Qwen2.5-32B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-32B`](Alibaba/qwen2.5/Qwen2.5-32B) |
| Qwen2.5-3B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-3B`](Alibaba/qwen2.5/Qwen2.5-3B) |
| Qwen2.5-72B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-72B`](Alibaba/qwen2.5/Qwen2.5-72B) |
| Qwen2.5-7B | ✅ Implemented | [`Alibaba/qwen2.5/Qwen2.5-7B`](Alibaba/qwen2.5/Qwen2.5-7B) |
| Qwen3 (0.6B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 (0.6B)`](Alibaba/qwen3/Qwen3%20%280.6B%29) |
| Qwen3 (235B-A22B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 (235B-A22B)`](Alibaba/qwen3/Qwen3%20%28235B-A22B%29) |
| Qwen3 (30B-A3B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 (30B-A3B)`](Alibaba/qwen3/Qwen3%20%2830B-A3B%29) |
| Qwen3 (32B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 (32B)`](Alibaba/qwen3/Qwen3%20%2832B%29) |
| Qwen3 (4B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 (4B)`](Alibaba/qwen3/Qwen3%20%284B%29) |
| Qwen3 (8B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 (8B)`](Alibaba/qwen3/Qwen3%20%288B%29) |
| Qwen3 Coder Flash (30B-A3B) | ✅ Implemented | [`Alibaba/qwen3/Qwen3 Coder Flash (30B-A3B)`](Alibaba/qwen3/Qwen3%20Coder%20Flash%20%2830B-A3B%29) |

</details>

<details>
<summary><b>BOSS Zhipin</b> &nbsp;·&nbsp; 1 implemented / 2 total</summary>


_Llama-style GQA decoder._


| Model | Status | Folder |
|---|---|---|
| Nanbeige 4.1 (3B) | ✅ Implemented | [`BOSS Zhipin/Nanbeige 4.1 (3B)`](BOSS%20Zhipin/Nanbeige%204.1%20%283B%29) |
| Nanbeige 4.2 (3B) | 🚧 Config + tokenizer only | [`BOSS Zhipin/Nanbeige 4.2 (3B)`](BOSS%20Zhipin/Nanbeige%204.2%20%283B%29) |

</details>

<details>
<summary><b>Cisco</b> &nbsp;·&nbsp; 1 implemented / 1 total</summary>


_Llama-style GQA decoder._


| Model | Status | Folder |
|---|---|---|
| Antares (1B) | ✅ Implemented | [`Cisco/Antares (1B)`](Cisco/Antares%20%281B%29) |

</details>

<details>
<summary><b>Cohere</b> &nbsp;·&nbsp; 3 implemented / 3 total</summary>


_Cohere2 family: parallel blocks (one norm; attention and FFN outputs both added to the residual), bias-free LayerNorm (RMSNorm in North Mini Code), interleaved RoPE on the sliding-window layers only (the full-attention layers have no positional encoding), `logit_scale`, tied embeddings. North Mini Code and Command A+ (one shared file) add an MoE: top-k-then-sigmoid routing without a selection bias, shared experts averaged with the routed output, dense leading layers. Command A+'s vision tower is not implemented; its language-model settings are read from `text_config`._


| Model | Status | Folder |
|---|---|---|
| Command A+ (218B-A25B) | ✅ Implemented | [`Cohere/Command A+ (218B-A25B)`](Cohere/Command%20A%2B%20%28218B-A25B%29) |
| North Mini Code (30B-A3B) | ✅ Implemented | [`Cohere/North Mini Code (30B-A3B)`](Cohere/North%20Mini%20Code%20%2830B-A3B%29) |
| Tiny Aya (3.35B) | ✅ Implemented | [`Cohere/Tiny Aya (3.35B)`](Cohere/Tiny%20Aya%20%283.35B%29) |

</details>

<details>
<summary><b>DeepSeek</b> &nbsp;·&nbsp; 2 implemented / 7 total</summary>


_DeepSeek-V2 and V3: multi-head latent attention, YaRN RoPE with the `mscale²` softmax scale, and shared-expert MoE after leading dense layers. V3 routes with sigmoid scores plus a selection bias over expert groups; V2 uses softmax group-limited top-k. The multi-token-prediction module is not implemented. R1, V3.2 and V4 are config only._


| Model | Status | Folder |
|---|---|---|
| DeepSeek R1 (671B) | 🚧 Config + tokenizer only | [`Deepseek/DeepSeek R1 (671B)`](Deepseek/DeepSeek%20R1%20%28671B%29) |
| DeepSeek V3 (671B) | ✅ Implemented | [`Deepseek/DeepSeek V3 (671B)`](Deepseek/DeepSeek%20V3%20%28671B%29) |
| DeepSeek V3.2 (671B) | 🚧 Config + tokenizer only | [`Deepseek/DeepSeek V3.2 (671B)`](Deepseek/DeepSeek%20V3.2%20%28671B%29) |
| DeepSeek V4-Flash (284B) | 🚧 Config + tokenizer only | [`Deepseek/DeepSeek V4-Flash (284B)`](Deepseek/DeepSeek%20V4-Flash%20%28284B%29) |
| DeepSeek V4-Pro (1.6T) | 🚧 Config + tokenizer only | [`Deepseek/DeepSeek V4-Pro (1.6T)`](Deepseek/DeepSeek%20V4-Pro%20%281.6T%29) |
| DeepSeek V4.1-Flash (552B) | 🚧 Config + tokenizer only | [`Deepseek/DeepSeek V4.1-Flash (552B)`](Deepseek/DeepSeek%20V4.1-Flash%20%28552B%29) |
| DeepSeek-V2 | ✅ Implemented | [`Deepseek/DeepSeek-V2`](Deepseek/DeepSeek-V2) |

</details>

<details>
<summary><b>GLM (Zhipu AI)</b> &nbsp;·&nbsp; 6 implemented / 7 total</summary>


_GLM-4.5 family: Mixture-of-Experts with sigmoid routing, group-limited top-k, shared experts, dense-first-k layers, partial RoPE, optional QK-norm. GLM-5 / 5.1 / 5.2: multi-head latent attention plus DeepSeek Sparse Attention (a small "lightning indexer" keeps the 2048 best tokens per query) with a shared-expert MoE; GLM-5.2 additionally shares the indexer's selection across layers (21 of its 78 layers run their own indexer). GLM-5 and GLM-5.1 share one file because their configs are identical. The multi-token-prediction layers are not implemented._


| Model | Status | Folder |
|---|---|---|
| GLM-4.5 (355B) | ✅ Implemented | [`GLM_Series/GLM-4.5 (355B)`](GLM_Series/GLM-4.5%20%28355B%29) |
| GLM-4.5-Air (106B) | ✅ Implemented | [`GLM_Series/GLM-4.5-Air (106B)`](GLM_Series/GLM-4.5-Air%20%28106B%29) |
| GLM-4.7 (355B) | ✅ Implemented | [`GLM_Series/GLM-4.7 (355B)`](GLM_Series/GLM-4.7%20%28355B%29) |
| GLM-5 (744B) | ✅ Implemented | [`GLM_Series/GLM-5 (744B)`](GLM_Series/GLM-5%20%28744B%29) |
| GLM-5.1 (744B) | ✅ Implemented | [`GLM_Series/GLM-5.1 (744B)`](GLM_Series/GLM-5.1%20%28744B%29) |
| GLM-5.2 (744B) | ✅ Implemented | [`GLM_Series/GLM-5.2 (744B)`](GLM_Series/GLM-5.2%20%28744B%29) |
| GLM-5.3-Flash (320B) | 🚧 Config + tokenizer only | [`GLM_Series/GLM-5.3-Flash (320B)`](GLM_Series/GLM-5.3-Flash%20%28320B%29) |

</details>

<details>
<summary><b>Google</b> &nbsp;·&nbsp; 9 implemented / 14 total</summary>


_Gemma 1; Gemma 2 (logit soft-capping, sliding/global layers, four norms per block); Gemma 3 1B / 4B / 12B / 27B text models (QK-norm, local/global RoPE bases, linear RoPE scaling on global layers; no vision tower)._


| Model | Status | Folder |
|---|---|---|
| gemma-2-27b | ✅ Implemented | [`Google/Gemma 2/gemma-2-27b`](Google/Gemma%202/gemma-2-27b) |
| gemma-2-2b | ✅ Implemented | [`Google/Gemma 2/gemma-2-2b`](Google/Gemma%202/gemma-2-2b) |
| gemma-2-9b | ✅ Implemented | [`Google/Gemma 2/gemma-2-9b`](Google/Gemma%202/gemma-2-9b) |
| Gemma 3 (270M) | 🚧 Config + tokenizer only | [`Google/Gemma 3/Gemma 3 (270M)`](Google/Gemma%203/Gemma%203%20%28270M%29) |
| gemma-3-12b | ✅ Implemented | [`Google/Gemma 3/gemma-3-12b`](Google/Gemma%203/gemma-3-12b) |
| gemma-3-1b | ✅ Implemented | [`Google/Gemma 3/gemma-3-1b`](Google/Gemma%203/gemma-3-1b) |
| gemma-3-27b | ✅ Implemented | [`Google/Gemma 3/gemma-3-27b`](Google/Gemma%203/gemma-3-27b) |
| gemma-3-4b | ✅ Implemented | [`Google/Gemma 3/gemma-3-4b`](Google/Gemma%203/gemma-3-4b) |
| Gemma 4 (12B) | 🚧 Config + tokenizer only | [`Google/Gemma 4/Gemma 4 (12B)`](Google/Gemma%204/Gemma%204%20%2812B%29) |
| Gemma 4 (31B) | 🚧 Config + tokenizer only | [`Google/Gemma 4/Gemma 4 (31B)`](Google/Gemma%204/Gemma%204%20%2831B%29) |
| Gemma 4 (E2B) | 🚧 Config + tokenizer only | [`Google/Gemma 4/Gemma 4 (E2B)`](Google/Gemma%204/Gemma%204%20%28E2B%29) |
| Gemma 4 (E4B) | 🚧 Config + tokenizer only | [`Google/Gemma 4/Gemma 4 (E4B)`](Google/Gemma%204/Gemma%204%20%28E4B%29) |
| gemma 2B | ✅ Implemented | [`Google/gemma/gemma 2B`](Google/gemma/gemma%202B) |
| gemma-7b | ✅ Implemented | [`Google/gemma/gemma-7b`](Google/gemma/gemma-7b) |

</details>

<details>
<summary><b>Hugging Face</b> &nbsp;·&nbsp; 1 implemented / 1 total</summary>


_SmolLM3: Llama-style GQA decoder with no positional encoding (NoPE) on every 4th layer._


| Model | Status | Folder |
|---|---|---|
| SmolLM3 (3B) | ✅ Implemented | [`Hugging Face/SmolLM3 (3B)`](Hugging%20Face/SmolLM3%20%283B%29) |

</details>

<details>
<summary><b>IBM</b> &nbsp;·&nbsp; 3 implemented / 3 total</summary>


_Granite 4.1 (3B, 8B, 30B): Llama-style GQA decoder with Granite's muP-style multipliers (embeddings ×12, attention scale 1/head_dim, scaled residual branches, scaled logits) and tied embeddings._


| Model | Status | Folder |
|---|---|---|
| Granite 4.1 (30B) | ✅ Implemented | [`IBM/Granite 4.1 (30B)`](IBM/Granite%204.1%20%2830B%29) |
| Granite 4.1 (3B) | ✅ Implemented | [`IBM/Granite 4.1 (3B)`](IBM/Granite%204.1%20%283B%29) |
| Granite 4.1 (8B) | ✅ Implemented | [`IBM/Granite 4.1 (8B)`](IBM/Granite%204.1%20%288B%29) |

</details>

<details>
<summary><b>Meta</b> &nbsp;·&nbsp; 15 implemented / 16 total</summary>


_LLaMA 1 (MHA) through 3.2 (GQA, `llama3` RoPE scaling in 3.1/3.2), and LLaMA 4 Maverick (text model: MoE with a shared expert, chunked attention, layers without positional encoding)._


| Model | Status | Folder |
|---|---|---|
| LLaMA 1 (13B) | ✅ Implemented | [`Meta/LLaMA 1/LLaMA 1 (13B)`](Meta/LLaMA%201/LLaMA%201%20%2813B%29) |
| LLaMA 1 (30B) | ✅ Implemented | [`Meta/LLaMA 1/LLaMA 1 (30B)`](Meta/LLaMA%201/LLaMA%201%20%2830B%29) |
| LLaMA 1 (65B) | ✅ Implemented | [`Meta/LLaMA 1/LLaMA 1 (65B)`](Meta/LLaMA%201/LLaMA%201%20%2865B%29) |
| LLaMA 1 (7B) | ✅ Implemented | [`Meta/LLaMA 1/LLaMA 1 (7B)`](Meta/LLaMA%201/LLaMA%201%20%287B%29) |
| LLaMA 2 (13B) | ✅ Implemented | [`Meta/LLaMA 2/LLaMA 2 (13B)`](Meta/LLaMA%202/LLaMA%202%20%2813B%29) |
| LLaMA 2 (70B) | ✅ Implemented | [`Meta/LLaMA 2/LLaMA 2 (70B)`](Meta/LLaMA%202/LLaMA%202%20%2870B%29) |
| LLaMA 2 (7B) | ✅ Implemented | [`Meta/LLaMA 2/LLaMA 2 (7B)`](Meta/LLaMA%202/LLaMA%202%20%287B%29) |
| LLaMA 3 (70B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3 (70B)`](Meta/LLaMA%203/LLaMA%203%20%2870B%29) |
| LLaMA 3 (8B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3 (8B)`](Meta/LLaMA%203/LLaMA%203%20%288B%29) |
| LLaMA 3.1 (405B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3.1/LLaMA 3.1 (405B)`](Meta/LLaMA%203/LLaMA%203.1/LLaMA%203.1%20%28405B%29) |
| LLaMA 3.1 (70B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3.1/LLaMA 3.1 (70B)`](Meta/LLaMA%203/LLaMA%203.1/LLaMA%203.1%20%2870B%29) |
| LLaMA 3.1 (8B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3.1/LLaMA 3.1 (8B)`](Meta/LLaMA%203/LLaMA%203.1/LLaMA%203.1%20%288B%29) |
| LLaMA 3.2 (1B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3.2/LLaMA 3.2 (1B)`](Meta/LLaMA%203/LLaMA%203.2/LLaMA%203.2%20%281B%29) |
| LLaMA 3.2 (3B) | ✅ Implemented | [`Meta/LLaMA 3/LLaMA 3.2/LLaMA 3.2 (3B)`](Meta/LLaMA%203/LLaMA%203.2/LLaMA%203.2%20%283B%29) |
| LLaMA 4 Maverick (400B) | ✅ Implemented | [`Meta/LLaMA 4/LLaMA 4 Maverick (400B)`](Meta/LLaMA%204/LLaMA%204%20Maverick%20%28400B%29) |
| Muse Glimmer (30B) | 🚧 Config + tokenizer only | [`Meta/Muse Glimmer (30B)`](Meta/Muse%20Glimmer%20%2830B%29) |

</details>

<details>
<summary><b>Microsoft</b> &nbsp;·&nbsp; 3 implemented / 3 total</summary>


_Phi-1 / Phi-1.5 (LayerNorm, GELU, MHA); Phi-4 (RMSNorm, GQA)._


| Model | Status | Folder |
|---|---|---|
| Phi-1 | ✅ Implemented | [`Microsoft/Phi-1`](Microsoft/Phi-1) |
| Phi-1.5 | ✅ Implemented | [`Microsoft/Phi-1.5`](Microsoft/Phi-1.5) |
| Phi-4 (14B) | ✅ Implemented | [`Microsoft/Phi-4 (14B)`](Microsoft/Phi-4%20%2814B%29) |

</details>

<details>
<summary><b>MiniMax</b> &nbsp;·&nbsp; 3 implemented / 4 total</summary>


_MiniMax M2, M2.5 and M2.7 (one shared file; the configs differ only in unused keys and context length): GQA with QK-norm over the whole query / key projection, partial RoPE (64 of 128 dims), and an MoE in every layer (256 sigmoid-routed experts with a selection bias, top 8, no shared expert). The multi-token-prediction modules are not implemented. M3 is config only._


| Model | Status | Folder |
|---|---|---|
| MiniMax M2 (230B) | ✅ Implemented | [`MiniMax/MiniMax M2 (230B)`](MiniMax/MiniMax%20M2%20%28230B%29) |
| MiniMax M2.7 (230B) | ✅ Implemented | [`MiniMax/MiniMax M2.7 (230B)`](MiniMax/MiniMax%20M2.7%20%28230B%29) |
| MiniMax M3 (428B) | 🚧 Config + tokenizer only | [`MiniMax/MiniMax M3 (428B)`](MiniMax/MiniMax%20M3%20%28428B%29) |
| MiniMax-M2.5 (230B) | ✅ Implemented | [`MiniMax/MiniMax-M2.5 (230B)`](MiniMax/MiniMax-M2.5%20%28230B%29) |

</details>

<details>
<summary><b>Mistral AI</b> &nbsp;·&nbsp; 3 implemented / 3 total</summary>


_Mistral Small 3.1: text decoder only (no vision tower). Mistral Small 4 and Mistral Large 3: multi-head latent attention with YaRN-scaled RoPE and a shared-expert MoE (128 routed experts, top 4; Large 3 adds 3 dense leading layers); text models only._


| Model | Status | Folder |
|---|---|---|
| Mistral Large 3 (673B) | ✅ Implemented | [`Mistral AI/Mistral Large 3 (673B)`](Mistral%20AI/Mistral%20Large%203%20%28673B%29) |
| Mistral Small 3.1 (24B) | ✅ Implemented | [`Mistral AI/Mistral Small 3.1 (24B)`](Mistral%20AI/Mistral%20Small%203.1%20%2824B%29) |
| Mistral Small 4 (119B) | ✅ Implemented | [`Mistral AI/Mistral Small 4 (119B)`](Mistral%20AI/Mistral%20Small%204%20%28119B%29) |

</details>

<details>
<summary><b>Nvidia</b> &nbsp;·&nbsp; 0 implemented / 5 total</summary>


| Model | Status | Folder |
|---|---|---|
| Nemotron 3 Nano (30B-A3B) | 🚧 Config + tokenizer only | [`Nvidia/Nemotron 3 Nano (30B-A3B)`](Nvidia/Nemotron%203%20Nano%20%2830B-A3B%29) |
| Nemotron 3 Nano (4B) | 🚧 Config + tokenizer only | [`Nvidia/Nemotron 3 Nano (4B)`](Nvidia/Nemotron%203%20Nano%20%284B%29) |
| Nemotron 3 Super (120B-A12B) | 🚧 Config + tokenizer only | [`Nvidia/Nemotron 3 Super (120B-A12B)`](Nvidia/Nemotron%203%20Super%20%28120B-A12B%29) |
| Nemotron 3 Ultra (550B-A55B) | 🚧 Config + tokenizer only | [`Nvidia/Nemotron 3 Ultra (550B-A55B)`](Nvidia/Nemotron%203%20Ultra%20%28550B-A55B%29) |
| Nemotron 3.5 Lightning (30B-A3B) | 🚧 Config + tokenizer only | [`Nvidia/Nemotron 3.5 Lightning (30B-A3B)`](Nvidia/Nemotron%203.5%20Lightning%20%2830B-A3B%29) |

</details>

<details>
<summary><b>OpenAI</b> &nbsp;·&nbsp; 2 implemented / 2 total</summary>


_GPT-OSS 20B / 120B: per-head attention sinks, alternating sliding-window and full attention, biased and clamped SwiGLU experts, top-4 routing, YaRN._


| Model | Status | Folder |
|---|---|---|
| GPT-OSS (120B) | ✅ Implemented | [`OpenAI/GPT-OSS (120B)`](OpenAI/GPT-OSS%20%28120B%29) |
| GPT-OSS (20B) | ✅ Implemented | [`OpenAI/GPT-OSS (20B)`](OpenAI/GPT-OSS%20%2820B%29) |

</details>

<details>
<summary><b>Poolside</b> &nbsp;·&nbsp; 3 implemented / 3 total</summary>


_Laguna XS.2, XS 2.1 and S 2.1 (one shared architecture, different configs): per-layer head counts, softplus per-head output gate, sigmoid-routed MoE with a selection bias and a shared expert, YaRN on the full-attention layers only._


| Model | Status | Folder |
|---|---|---|
| Laguna S 2.1 (118B) | ✅ Implemented | [`Poolside/Laguna S 2.1 (118B)`](Poolside/Laguna%20S%202.1%20%28118B%29) |
| Laguna XS 2.1 (33B) | ✅ Implemented | [`Poolside/Laguna XS 2.1 (33B)`](Poolside/Laguna%20XS%202.1%20%2833B%29) |
| Laguna XS.2 (33B) | ✅ Implemented | [`Poolside/Laguna XS.2 (33B)`](Poolside/Laguna%20XS.2%20%2833B%29) |

</details>

<details>
<summary><b>Prime Intellect</b> &nbsp;·&nbsp; 1 implemented / 1 total</summary>


_INTELLECT-3 uses the GLM-4.5-Air architecture._


| Model | Status | Folder |
|---|---|---|
| INTELLECT-3 (106B) | ✅ Implemented | [`Prime Intellect/INTELLECT-3 (106B)`](Prime%20Intellect/INTELLECT-3%20%28106B%29) |

</details>

<details>
<summary><b>Sarvam AI</b> &nbsp;·&nbsp; 1 implemented / 2 total</summary>


_Sarvam 30B: fused QKV projection with QK-norm, a dense first layer, then sigmoid-routed MoE with a shared expert._


| Model | Status | Folder |
|---|---|---|
| Sarvam (105B) | 🚧 Config + tokenizer only | [`Sarvam AI/Sarvam (105B)`](Sarvam%20AI/Sarvam%20%28105B%29) |
| Sarvam (30B) | ✅ Implemented | [`Sarvam AI/Sarvam (30B)`](Sarvam%20AI/Sarvam%20%2830B%29) |

</details>

<details>
<summary><b>StepFun</b> &nbsp;·&nbsp; 1 implemented / 1 total</summary>


_Step 3.5 Flash: 64-head full-attention and 96-head sliding-window layers, sigmoid per-head output gate, zero-centred RMSNorm, per-layer RoPE settings, clamped SwiGLU, sigmoid MoE with a selection bias. The multi-token-prediction layers are not implemented._


| Model | Status | Folder |
|---|---|---|
| Step 3.5 Flash (196B) | ✅ Implemented | [`StepFun/Step 3.5 Flash (196B)`](StepFun/Step%203.5%20Flash%20%28196B%29) |

</details>

<details>
<summary><b>Weibo</b> &nbsp;·&nbsp; 1 implemented / 1 total</summary>


_VibeThinker-3B: a Qwen2 decoder (GQA 16 / 2 heads, bias on the Q / K / V projections only, tied embeddings, no QK-norm)._


| Model | Status | Folder |
|---|---|---|
| VibeThinker-3B (3B) | ✅ Implemented | [`Weibo/VibeThinker-3B (3B)`](Weibo/VibeThinker-3B%20%283B%29) |

</details>

<details>
<summary><b>xAI</b> &nbsp;·&nbsp; 0 implemented / 1 total</summary>


| Model | Status | Folder |
|---|---|---|
| Grok 2.5 (270B) | 📁 Placeholder | [`xAI/Grok 2.5 (270B)`](xAI/Grok%202.5%20%28270B%29) |

</details>

<details>
<summary><b>Xiaomi</b> &nbsp;·&nbsp; 0 implemented / 5 total</summary>


| Model | Status | Folder |
|---|---|---|
| Xiaomi MiMo-V2-Flash (309B) | 🚧 Config + tokenizer only | [`Xiaomi/Xiaomi MiMo-V2-Flash (309B)`](Xiaomi/Xiaomi%20MiMo-V2-Flash%20%28309B%29) |
| Xiaomi MiMo-V2.5 (310B) | 🚧 Config + tokenizer only | [`Xiaomi/Xiaomi MiMo-V2.5 (310B)`](Xiaomi/Xiaomi%20MiMo-V2.5%20%28310B%29) |
| Xiaomi MiMo-V2.5-Pro (1.02T) | 🚧 Config + tokenizer only | [`Xiaomi/Xiaomi MiMo-V2.5-Pro (1.02T)`](Xiaomi/Xiaomi%20MiMo-V2.5-Pro%20%281.02T%29) |
| Xiaomi MiMo-V2.6-Flash-RL (309B) | 🚧 Config + tokenizer only | [`Xiaomi/Xiaomi MiMo-V2.6-Flash-RL (309B)`](Xiaomi/Xiaomi%20MiMo-V2.6-Flash-RL%20%28309B%29) |
| Xiaomi MiMo-V2.6-Pro-RL (1.02T) | 🚧 Config + tokenizer only | [`Xiaomi/Xiaomi MiMo-V2.6-Pro-RL (1.02T)`](Xiaomi/Xiaomi%20MiMo-V2.6-Pro-RL%20%281.02T%29) |

</details>

## Attention mechanisms

The [`attention-mechanisms/`](attention-mechanisms) folder holds small, standalone implementations that are easier to study in isolation than inside a full model. Each file has a runnable demo (`python <file>.py`) that prints the tensor shapes:

| File | Class | Mechanism |
|---|---|---|
| [`additive_attention.py`](attention-mechanisms/additive_attention.py) | `AdditiveAttention` | Additive attention (Bahdanau) |
| [`multiplicative_attention.py`](attention-mechanisms/multiplicative_attention.py) | `MultiplicativeAttention` | Multiplicative attention (Luong) |
| [`self_attention.py`](attention-mechanisms/self_attention.py) | `SelfAttention` | Single-head scaled dot-product self-attention |
| [`masked_multi_head_attention.py`](attention-mechanisms/masked_multi_head_attention.py) | `MaskedMultiHeadAttention` | Causal (masked) multi-head attention |
| [`multi_head_attention.py`](attention-mechanisms/multi_head_attention.py) | `MultiHeadAttention` | Multi-head attention (MHA) |
| [`multi_query_attention.py`](attention-mechanisms/multi_query_attention.py) | `MultiQueryAttention` | Multi-query attention (MQA) |
| [`grouped_query_attention.py`](attention-mechanisms/grouped_query_attention.py) | `GroupedQueryAttention` | Grouped-query attention (GQA) |
| [`multi_head_latent_attention.py`](attention-mechanisms/multi_head_latent_attention.py) | `MultiHeadLatentAttention` | Multi-head latent attention (MLA) — simplified: low-rank K/V only, without query compression or decoupled RoPE |
| [`sliding_window_attention.py`](attention-mechanisms/sliding_window_attention.py) | `SlidingWindowAttention` | Sliding-window attention (SWA), causal or bidirectional |

## Verification status

Being precise about what "implemented" means here:

| Question | Answer today |
|---|---|
| Does each `model.py` build from its `cfg` and run a forward pass? | **Intended, but not exhaustively tested.** Every implemented model ships a `main()` demo that generates tokens with random weights, but the demos for the largest models cannot run on typical hardware (they allocate every parameter in fp32), and some demos have not been executed end to end. See [Known limitations](#known-limitations). |
| Can pretrained weights be loaded? | **Not yet.** There is no weight-loading or checkpoint-conversion code. |
| Have outputs been compared with a reference implementation? | **For the models listed below, yes; for the rest, no.** The comparisons copy identical random weights into a small model of the same structure and into the reference, and compare logits (they agree to roughly 1e-5 or better in float32). Each also checks that the full-size configuration has exactly the same parameter count as the reference. The comparisons were also confirmed to be sensitive by deliberately breaking individual features and checking that the test fails; that was done for every model above except DeepSeek V2 / V3, INTELLECT-3, Mistral Large 3, the three Cohere models and VibeThinker-3B (Granite was checked by switching off each multiplier in turn). |
| Are those comparison scripts in this repository? | **No.** They were run outside the repo and are not committed. There is no automated test suite or CI yet. |
| Does inference use a KV cache or sampling? | **Mostly no.** Generation is greedy and recomputes the full sequence each step. The one exception is `OLMo 2 (7B)/model_with_kv_cache.py`. |

**Checked against `transformers`:** Gemma 3 4B / 12B / 27B, Mistral Small 4, Mistral Large 3 (against `transformers`' Mistral 4 and DeepSeek-V2 code, since Large 3 has no `transformers` implementation), Qwen3 Coder Flash / 30B-A3B / 235B-A22B, INTELLECT-3 (which shares its code with GLM-4.5-Air), DeepSeek V2 and V3, GLM-5 / 5.1 / 5.2 (against the newest `glm_moe_dsa` source of Hugging Face `transformers`, because the released 5.5.x versions predate its indexer sharing and use a different RoPE pairing), GPT-OSS 20B / 120B, Granite 4.1 3B / 8B / 30B, LLaMA 4 Maverick, Qwen1.5 14B-A2.7B (`Qwen2Moe`), SmolLM3, MiniMax M2 / M2.5 / M2.7 (the native `MiniMaxM2`, with the partial-rotary factor set to `rotary_dim / head_dim` as in MiniMax's official config), VibeThinker-3B (`Qwen2`), Tiny Aya (`Cohere2`), and North Mini Code / Command A+ (`Cohere2Moe`, taken from the `transformers` main branch because no released version includes it yet).

**Checked against the model authors' official Hugging Face code:** Step 3.5 Flash, Sarvam 30B, Laguna XS.2 and XS 2.1. Laguna S 2.1 uses the same file as XS 2.1; only its full-size parameter count (117.56 B, equal to a layer-by-layer count) and a shrunken forward pass were checked.

The `main()` demos of the three Cohere models were also run end to end (real `config.json` and `tokenizer.json`, shrunk width and depth); the full-size ones need far more memory than typical hardware.

**Not checked against any reference:** everything else, including OLMo 2 / 3, the other Qwen models, LLaMA 1 / 2 / 3 / 3.1 / 3.2, Gemma 1 / 2 / 3 (1B), Phi, Nanbeige, Antares, Mistral Small 3.1 and the GLM-4.5 / 4.5-Air / 4.7 folders themselves. Treat those as faithful *attempts* at the published architectures until logit-level verification is added.

## Known limitations

Issues already identified, listed so nobody is surprised:

- **Gemma 3 (1B) cannot run a forward pass.** `Model.forward` uses `self.emb_scale`, which is never defined, so it raises `AttributeError`. The 4B / 12B / 27B files define it (the embeddings are scaled by `sqrt(hidden_size)`).
- **Qwen3 (4B), (8B) and (32B) are missing QK-norm.** They do not contain the per-head query/key RMSNorm that the Qwen3 architecture uses. Qwen3 (0.6B) and the three Qwen3 MoE models do.
- **Qwen1.5 and Qwen2.5 are configured with `bias_qkv: False`.** The reference models use a bias on the Q/K/V projections.
- **OLMo 2 (7B) block layout needs confirming.** It is currently written with pre-norm; OLMo 2 is believed to normalise the attention and FFN *outputs* (as the OLMo 3 file does). Not yet checked against the reference code.
- **OLMo 3 applies YaRN to every layer.** Whether the sliding-window layers should use plain RoPE instead is still to be verified.
- **Mistral Small 3.1's demo cannot run as shipped:** `main()` opens `tokenizer_config.json`, which is not in that folder. The model covers the text decoder only (no vision tower).
- **Multi-token-prediction modules are not implemented** (DeepSeek V3, GLM-4.7, GLM-5 / 5.1 / 5.2, Step 3.5 Flash, MiniMax M2 / M2.5 / M2.7). They are only used for speculative decoding and training. **Vision encoders are not implemented** for any model.
- **Demos for large models need far more memory than typical hardware**: a model with hundreds of billions of parameters needs hundreds of GB to several TB in fp32. The parameter counts of these configs were checked on PyTorch's `meta` device instead.
- **Code style is not yet uniform across families.** Older folders (e.g. LLaMA 1/2/3, Qwen dense, Gemma 1, Phi) use a positional-argument attention constructor, an in-file `Rope` module and a `CustomDataset` helper; newer ones use `cfg`-dict constructors, `compute_rope_params` / `apply_rope`, one shared RoPE table per model, and read their `config.json` directly. Models within a family are near-duplicates that differ only in their config values.

## Roadmap

Planned work, roughly in priority order. Nothing below is implemented yet.

- [ ] **Correctness:** add the logit-comparison scripts to the repository, plus weight-loading scripts and comparisons on real small checkpoints (e.g. Qwen2.5-0.5B, LLaMA 3.2 1B, Gemma 3 1B, OLMo 2/3 7B)
- [ ] Verify or fix the older implementations listed under [known limitations](#known-limitations) (the Qwen dense models, OLMo, LLaMA 1 / 2 / 3, Gemma 1 / 2 / 3 1B, Phi)
- [ ] **Implement the remaining 🚧 models**, prioritising new ideas over extra sizes: DeepSeek R1 / V3.2 / V4, Gemma 4, Nemotron 3, MiniMax M3, MiMo, GLM-5.3-Flash, Sarvam 105B, Nanbeige 4.2, Gemma 3 270M, Muse Glimmer
- [ ] Unify code style on the newer `cfg`-dict / `compute_rope_params` structure; share one implementation per family
- [ ] KV cache and sampling (temperature, top-k, top-p) for the other models; chat templates
- [ ] Parameter-count and memory-estimate script (instantiate on the `meta` device to sanity-check configs)
- [ ] Automated smoke tests and CI; `pyproject.toml` packaging
- [ ] Reduce repository size (shared tokenizers, a single `docs/` folder for diagrams)
- [ ] Architecture comparison tables (attention type, norm placement, positional encoding, dense vs. MoE)

## Contributing

Issues and pull requests are welcome — especially:

- verification of an implementation against the reference model (a mismatch report is as valuable as a fix),
- implementations for any 🚧 model,
- corrections to the [known limitations](#known-limitations).

When adding a model, follow the existing layout: a `<Vendor>/<Model Name (Size)>/` folder containing `model.py`, `config.json` and `tokenizer.json`, with `cfg` keys mirroring the official config.

## License

This project's code is released under the [MIT License](LICENSE). The architecture diagrams are third-party material and are not covered by it; see [Acknowledgements](#acknowledgements).

## Acknowledgements

**Architecture visuals** used in this repository are sourced from [Sebastian Raschka's Architecture Gallery](https://sebastianraschka.com/). Special thanks to Sebastian Raschka for creating and sharing these valuable visual resources. The diagrams remain the work of their creator and are not covered by this repository's MIT license; please refer to the source for their terms of use.

Architectures are based on the original papers and on the publicly released model configurations and tokenizers from each model's authors. Model names and trademarks belong to their respective owners; this project is independent and unaffiliated.
