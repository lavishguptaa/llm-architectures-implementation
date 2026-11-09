"""Count the parameters (weights) of any model implemented in this repository, without allocating memory.

The model is built on PyTorch's `meta` device, so even a 671B model is counted in seconds and needs no RAM.
The sizes are the ones the model's own `main()` uses (they mirror its config.json).

Usage (from anywhere):

    python count_parameters.py "Ai2/OLMo 2 (7B)"             # one model: breakdown by component
    python count_parameters.py "Ai2/OLMo 2 (7B)" --compare-hf  # also compare with Hugging Face transformers
    python count_parameters.py --all                          # one summary line for every implemented model
    python count_parameters.py --list                         # list the model folders that can be counted

The model folder is given relative to the repository root (or as any path that exists).
Exit status: 0 on success, 1 if --compare-hf finds a different count, 2 on a usage or loading error.
"""

import argparse
import importlib.util
import os
import re
import sys
import time
from collections import OrderedDict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", "assets", "model_theoretical"}


class _Captured(Exception):
    """Raised by the stand-in Model to hand the constructor arguments back without building anything."""


# --------------------------------------------------------------------------- finding models
def find_models():
    """Every folder in the repository that contains a model.py."""
    folders = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "model.py" in files:
            folders.append(root)
    return sorted(folders)


def resolve_folder(arg):
    for candidate in (arg, os.path.join(REPO, arg)):
        if os.path.isfile(os.path.join(candidate, "model.py")):
            return os.path.abspath(candidate)
    print(f"no model.py found in '{arg}' (relative to the repository root or the current directory). "
          f"Use --list to see the available folders.", file=sys.stderr)
    sys.exit(2)


def short_name(folder):
    return os.path.relpath(folder, REPO).replace(os.sep, "/")


# --------------------------------------------------------------------------- building on the meta device
def build_meta_model(folder):
    """Import the folder's model.py, learn the Model arguments from its main(), and build the model on `meta`."""
    spec = importlib.util.spec_from_file_location(f"_count_{abs(hash(folder))}", os.path.join(folder, "model.py"))
    module = importlib.util.module_from_spec(spec)
    previous = os.getcwd()
    os.chdir(folder)                         # main() reads config.json / tokenizer.json by relative path
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, "Model") or not hasattr(module, "main"):
            raise RuntimeError("model.py has no Model class and main() function")
        real_model_class = module.Model
        captured = {}

        def capture(*args, **kwargs):
            captured["call"] = (args, kwargs)
            raise _Captured

        module.Model = capture                # main() builds the config, then calls Model(cfg): grab that call
        try:
            module.main()
        except _Captured:
            pass
        if "call" not in captured:
            raise RuntimeError("main() did not construct a Model")
        args, kwargs = captured["call"]
        with torch.device("meta"):
            return real_model_class(*args, **kwargs)
    finally:
        os.chdir(previous)


# --------------------------------------------------------------------------- counting
INDEX = re.compile(r"\.\d+(?=\.|$)")


def count_parameters(model):
    """Parameters of the model; a tensor shared by two layers (tied weights) is counted once."""
    return sum(p.numel() for p in model.parameters())


def count_buffers(model):
    return sum(b.numel() for b in model.buffers())


def breakdown(model):
    """Group parameters by name with the layer / expert numbers replaced by '*'."""
    groups = OrderedDict()
    for name, p in model.named_parameters():
        g = groups.setdefault(INDEX.sub(".*", name), {"tensors": 0, "params": 0, "shapes": set()})
        g["tensors"] += 1
        g["params"] += p.numel()
        g["shapes"].add(tuple(p.shape))
    return groups


def human(n):
    return f"{n:,}"


def memory(n_params):
    return (f"{n_params * 4 / 1e9:,.2f} GB in float32, {n_params * 2 / 1e9:,.2f} GB in bfloat16/float16, "
            f"{n_params / 1e9:,.2f} GB in int8")


# --------------------------------------------------------------------------- Hugging Face comparison (optional)
def hf_count(folder):
    """Parameters of the Hugging Face model for this folder's config.json, or (None, reason)."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        return None, "transformers is not installed (pip install transformers)"
    if not os.path.isfile(os.path.join(folder, "config.json")):
        return None, "this folder has no config.json"
    try:
        config = AutoConfig.from_pretrained(folder, trust_remote_code=False)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except Exception as error:               # unsupported architecture, remote-code-only models, ...
        return None, f"the installed transformers cannot build it ({type(error).__name__}: {str(error)[:90]})"
    return sum(p.numel() for p in model.parameters()), None


# --------------------------------------------------------------------------- commands
def show_one(folder, compare_hf):
    started = time.time()
    model = build_meta_model(folder)
    groups = breakdown(model)
    total = count_parameters(model)
    assert sum(g["params"] for g in groups.values()) == total, "breakdown does not add up to the total"

    print(f"{short_name(folder)}  (layer / expert numbers are shown as *)\n")
    name_width = max(len("Parameter group"), max(len(k) for k in groups))
    shape_text = {k: (str(next(iter(g["shapes"]))) if len(g["shapes"]) == 1 else f"{len(g['shapes'])} shapes") for k, g in groups.items()}
    shape_width = max(len("Shape"), max(len(s) for s in shape_text.values()))
    print(f"{'Parameter group':<{name_width}}  {'Tensors':>7}  {'Shape':<{shape_width}}  {'Parameters':>20}  {'Share':>7}")
    print("-" * (name_width + shape_width + 45))
    for key, g in groups.items():
        print(f"{key:<{name_width}}  {g['tensors']:>7}  {shape_text[key]:<{shape_width}}  {human(g['params']):>20}  {100 * g['params'] / total:6.2f}%")
    print("-" * (name_width + shape_width + 45))
    print(f"{'Total':<{name_width}}  {sum(g['tensors'] for g in groups.values()):>7}  {'':<{shape_width}}  {human(total):>20}  {100.0:6.2f}%\n")
    print(f"Memory for the weights: {memory(total)}")
    buffers = count_buffers(model)
    if buffers:
        print(f"Not counted as weights: {human(buffers)} buffer elements (RoPE tables, masks, router biases, ...)")
    print(f"(built and counted in {time.time() - started:.1f} s, nothing was allocated)")

    if compare_hf:
        hf_total, reason = hf_count(folder)
        print()
        if hf_total is None:
            print(f"Hugging Face comparison skipped: {reason}")
        elif hf_total == total:
            print(f"Hugging Face model: {human(hf_total)} parameters -> match")
        else:
            print(f"Hugging Face model: {human(hf_total)} parameters -> DIFFERENT (this repo: {human(total)}, difference {human(total - hf_total)})")
            print("Note: Hugging Face may include parts this repository does not implement (vision / audio towers, "
                  "multi-token-prediction layers).")
            return 1
    return 0


def show_all(compare_hf):
    folders = find_models()
    rows, failures, differences, started = [], [], 0, time.time()
    for i, folder in enumerate(folders, 1):
        name = short_name(folder)
        try:
            n = count_parameters(build_meta_model(folder))
        except Exception as error:
            failures.append((name, f"{type(error).__name__}: {str(error)[:80]}"))
            print(f"[{i}/{len(folders)}] {name}: FAILED", file=sys.stderr)
            continue
        status = ""
        if compare_hf:
            hf_total, reason = hf_count(folder)
            status = "match" if hf_total == n else (f"DIFFERENT ({human(hf_total)})" if hf_total is not None else f"skipped ({reason[:40]})")
            if hf_total is not None and hf_total != n:
                differences += 1
        rows.append((name, n, status))
        print(f"[{i}/{len(folders)}] {name}: {human(n)}", file=sys.stderr)
    width = max(len(r[0]) for r in rows)
    print(f"\n{'Model':<{width}}  {'Parameters':>20}  {'float32':>10}  {'bfloat16':>10}" + ("  Hugging Face" if compare_hf else ""))
    print("-" * (width + 46 + (14 if compare_hf else 0)))
    for name, n, status in rows:
        print(f"{name:<{width}}  {human(n):>20}  {n * 4 / 1e9:>7,.1f} GB  {n * 2 / 1e9:>7,.1f} GB" + (f"  {status}" if compare_hf else ""))
    print(f"\n{len(rows)} models counted in {time.time() - started:.0f} s; {len(failures)} could not be built.")
    for name, why in failures:
        print(f"  FAILED {name}: {why}")
    if compare_hf:
        print(f"{differences} model(s) differ from Hugging Face (a vision tower or other part this repository does not implement is a common reason).")
    return 2 if failures else (1 if differences else 0)


def main():
    parser = argparse.ArgumentParser(description="Count the parameters of a model implemented in this repository (no memory needed).")
    parser.add_argument("model", nargs="?", help='model folder relative to the repository root, e.g. "Ai2/OLMo 2 (7B)"')
    parser.add_argument("--all", action="store_true", help="count every implemented model and print one line each")
    parser.add_argument("--list", action="store_true", help="list the model folders that can be counted")
    parser.add_argument("--compare-hf", action="store_true", help="also build the Hugging Face model from the folder's config.json and compare")
    args = parser.parse_args()

    if args.list:
        print("\n".join(short_name(f) for f in find_models()))
        return 0
    if args.all:
        return show_all(args.compare_hf)
    if not args.model:
        parser.error('give a model folder (for example "Ai2/OLMo 2 (7B)"), or use --all / --list')
    try:
        return show_one(resolve_folder(args.model), args.compare_hf)
    except SystemExit:
        raise
    except Exception as error:
        print(f"could not build the model: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
