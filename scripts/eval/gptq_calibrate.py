# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch==2.13.0",
#     "triton-rocm==3.7.1",
#     "llmcompressor==0.14.0",
#     "compressed-tensors==0.19.0",
#     "transformers==5.17.0",
#     "accelerate==1.15.0",
#     "datasets==5.0.1",
#     "pyarrow",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm7.2"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# triton-rocm = { index = "pytorch-rocm" }
# ///
"""Quantize Llama-3.2-3B-Instruct to GPTQ W4A16 with llm-compressor (Phase 6a Task 18, user decision
2026-09-30 "C then A", option A). Fixture time only, on `novanas`: never in the build or serving path.

Recipe: `GPTQModifier` on every `Linear` except `lm_head`, scheme `W4A16` (4-bit int, symmetric, one
scale per group of 128 input columns, no zero point), `dampening_frac` 0.01, `actorder` None (no act
order), block size 128. Calibration: `--samples` (512) conversations of HuggingFaceH4/ultrachat_200k,
split `train_sft`, from its first parquet shard at a pinned dataset revision (downloaded beforehand
into `--dataset-file`, no network here), shuffled with `--seed` (42), rendered with the model's chat
template and truncated to `--seq-len` (2048) tokens — the llm-compressor W4A16 example's settings.

Output (`--out`): a compressed-tensors `pack-quantized` checkpoint, which Turbine loads as
`ct_pack_int4` (support column `gptq_int4`), plus `turbine_calibration.json` recording the recipe, the
dataset (repo, revision, file, split, seed, samples, sequence length), the source model and the
package versions (torch, its HIP version, llm-compressor, compressed-tensors, transformers, ...).
The checkpoint is written to `<out>.tmp` and renamed over `<out>` only when everything succeeded.

`--check` imports everything, builds the recipe, prints the versions and exits without loading the
model (the environment warm-up; runs with the GPU hidden). A real run refuses to start without a
visible GPU unless `--allow-cpu` (a CPU GPTQ of a 3B model takes hours and slows the whole host).

    uv run scripts/eval/gptq_calibrate.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct \\
        --dataset-file <ultrachat>/data/train_sft-00000-of-00003-a3ecf92756993583.parquet \\
        --out /home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own
"""

import argparse
import json
import os
import shutil
import sys
import time
from importlib import metadata

DATASET_REPO = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DATASET_SPLIT = "train_sft"
SOURCE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
SOURCE_REVISION = "006f5dcd1393c3add266de40994ba96225e9689d"
PACKAGES = [
    "torch",
    "llmcompressor",
    "compressed-tensors",
    "transformers",
    "accelerate",
    "datasets",
    "safetensors",
    "auto-round",
    "numpy",
]


def gptq_modifier_class():
    try:
        from llmcompressor.modifiers.gptq import GPTQModifier
    except ImportError:
        from llmcompressor.modifiers.quantization import GPTQModifier
    return GPTQModifier


def build_recipe(damp):
    return gptq_modifier_class()(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        dampening_frac=damp,
        actorder=None,
        block_size=128,
    )


def versions():
    import torch

    v = {}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = None
    v["torch_hip"] = torch.version.hip
    v["python"] = sys.version.split()[0]
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir")
    ap.add_argument("--dataset-file")
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument(
        "--gptq-backend",
        choices=["auto", "torch", "triton"],
        default="auto",
        help=(
            "GPTQ block-update backend. 'auto' (default) disables llm-compressor's "
            "fused Triton kernel on a ROCm build of torch (its extern-lib lowering "
            "does not compile on ROCm Triton: 'Implicit conversion of CUDA "
            "__nv_fdiv_rn device function has been dropped') and keeps it on CUDA; "
            "'torch' / 'triton' force the eager or fused path via "
            "LLMCOMPRESSOR_DISABLE_GPTQ_TRITON, the switch gptq_quantize.py's own "
            "dispatch (_gptq_block_update_triton_req) already reads."
        ),
    )
    a = ap.parse_args()

    import torch

    is_rocm = getattr(torch.version, "hip", None) is not None
    disable_triton = a.gptq_backend == "torch" or (a.gptq_backend == "auto" and is_rocm)
    gptq_backend_used = "torch_eager" if disable_triton else "triton"
    if disable_triton:
        os.environ["LLMCOMPRESSOR_DISABLE_GPTQ_TRITON"] = "1"

    recipe = build_recipe(a.damp)
    from llmcompressor import oneshot

    v = versions()
    if a.check:
        print(
            json.dumps(
                {
                    "versions": v,
                    "recipe": repr(recipe),
                    "oneshot": oneshot.__module__,
                    "gptq_backend": gptq_backend_used,
                    "torch_hip": torch.version.hip,
                },
                indent=1,
            )
        )
        return 0
    for name in ("model_dir", "dataset_file", "out"):
        if not getattr(a, name):
            ap.error(f"--{name.replace('_', '-')} is required")

    gpu = torch.cuda.is_available()
    if not gpu and not a.allow_cpu:
        print(
            "gptq_calibrate: no visible GPU (pass --allow-cpu to run on the CPU)",
            file=sys.stderr,
        )
        return 1
    device = torch.cuda.get_device_name(0) if gpu else "cpu"
    print(
        f"gptq_calibrate: device {device}; gptq_backend {gptq_backend_used}; "
        f"versions {v}",
        flush=True,
    )

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16)

    ds = load_dataset("parquet", data_files=a.dataset_file, split="train")
    ds = ds.shuffle(seed=a.seed).select(range(a.samples))

    def render(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    def encode(ex):
        return tok(
            ex["text"],
            padding=False,
            max_length=a.seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(render)
    ds = ds.map(encode, remove_columns=ds.column_names)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=a.seq_len,
        num_calibration_samples=a.samples,
    )

    tmp = a.out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, save_compressed=True)
    tok.save_pretrained(tmp)
    prov = {
        "tool": "scripts/eval/gptq_calibrate.py",
        "source_model": {
            "repo": SOURCE_MODEL,
            "revision": SOURCE_REVISION,
            "dir": a.model_dir,
        },
        "recipe": {
            "modifier": "GPTQModifier",
            "targets": "Linear",
            "ignore": ["lm_head"],
            "scheme": "W4A16",
            "bits": 4,
            "symmetric": True,
            "group_size": 128,
            "dampening_frac": a.damp,
            "actorder": None,
            "block_size": 128,
        },
        "calibration": {
            "dataset": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": os.path.basename(a.dataset_file),
            "split": DATASET_SPLIT,
            "shuffle_seed": a.seed,
            "samples": a.samples,
            "max_seq_len": a.seq_len,
            "rendering": "model chat template, add_special_tokens false, truncation",
        },
        "device": device,
        "gptq_backend": gptq_backend_used,
        "versions": v,
        "seconds": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(
        os.path.join(tmp, "turbine_calibration.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    shutil.rmtree(a.out, ignore_errors=True)
    os.rename(tmp, a.out)
    print(f"gptq_calibrate: wrote {a.out} in {prov['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
