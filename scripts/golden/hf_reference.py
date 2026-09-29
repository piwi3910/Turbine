# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==4.57.1",
#     "safetensors==0.6.2",
#     "jinja2==3.1.6",
# ]
# ///
"""Golden reference generator for Turbine (P1 S-11).

Runs every prompt of a prompts.jsonl file through Hugging Face transformers
(`AutoModelForCausalLM`, BF16 weights) with greedy decoding and writes one
reference record per prompt as JSONL:

    {"id", "engine", "model", ["revision",] "captured", "prompt_token_ids",
     "tokens", "top_logprobs": [[[id, logprob], ...], ...]}

`model` is `--model-name` when given, else config.json `_name_or_path` when it
names a Hub repository, else the model directory name. `revision` is the Hub
commit the checkpoint was downloaded at, read from the metadata that
`hf download --local-dir` leaves in `<model-dir>/.cache/huggingface/download/`;
the field is omitted when that metadata is absent (e.g. a synthetic checkpoint).

Rules:
- chat prompts are rendered with the tokenizer's chat template,
  `add_generation_prompt=True` and the prompt's `chat_template_kwargs`, then
  tokenized with `add_special_tokens=False` (the template emits BOS itself);
  completion prompts are tokenized with `add_special_tokens=True`.
- Decoding is greedy and runs for exactly `max_tokens` positions: it does NOT
  stop at EOS. The reference therefore always holds `max_tokens` positions;
  `turbine-golden compare` requests `ignore_eos: true` and treats a candidate
  that stops early as diverging at the position where it stopped.
- Per generated position the script records the chosen token and the top-N
  FP32 `log_softmax` values of FP32 logits: the LM head runs in FP32 on the
  final-norm output (BF16, as the model produces it) and the LM head weight,
  both upcast, so the logits are not rounded to BF16 (whose spacing is 0.125 at
  |logit| in [16, 32)); every other op stays BF16. The engine string ends in
  `-fp32-logits`. This is the golden reference form (user decision 2026-09-26).
- `--bf16-logits` (diagnostics only) instead log-softmaxes the model's own BF16
  logits upcast to FP32; the engine string then has no `-fp32-logits` suffix.
- Output goes to `<out>.tmp`, renamed over `<out>` only when every prompt
  succeeded. Any failure (including usage errors) exits 1 and leaves no file.

Usage:
    uv run scripts/golden/hf_reference.py --model-dir <dir> \
        --prompts tests/golden/prompts.jsonl \
        --out tests/golden/<model-slug>/reference.jsonl \
        [--top-logprobs 20] [--device cpu|cuda] [--model-name <hub-id>] [--bf16-logits] \
        [--config-override <json>]

`--config-override` (Phase 6a S-16) replaces top-level config.json keys with the
given JSON object's before the model loads, e.g.
`{"rope_scaling": {"rope_type": "yarn", "factor": 16.0, ...}}` for the YaRN golden:
the same as loading a copy of the checkpoint whose config.json carries those keys,
without copying the weights. The reference lines do not record it; the fixture's
README does.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors are failures too: exit 1
        self.print_usage(sys.stderr)
        print(f"hf_reference.py: error: {message}", file=sys.stderr)
        sys.exit(1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = _Parser(
        description="Generate golden reference fixtures with HF transformers (BF16, greedy)."
    )
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-logprobs", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--model-name",
        help="Hub id recorded as `model` (default: config.json _name_or_path or the directory name)",
    )
    p.add_argument(
        "--bf16-logits",
        action="store_true",
        help="log-softmax the model's BF16 logits (default: FP32 LM head on the BF16 final-norm output)",
    )
    p.add_argument(
        "--config-override",
        type=json.loads,
        help="JSON object whose keys replace config.json's top-level keys (e.g. rope_scaling)",
    )
    args = p.parse_args(argv)
    if not 1 <= args.top_logprobs <= 20:
        p.error("--top-logprobs must be between 1 and 20")
    if args.config_override is not None and not isinstance(args.config_override, dict):
        p.error("--config-override must be a JSON object")
    return args


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "completion":
                if not isinstance(rec.get("prompt"), str):
                    raise TypeError(
                        f"{path}:{lineno}: completion prompt needs a string 'prompt'"
                    )
            elif kind == "chat":
                if not isinstance(rec.get("messages"), list):
                    raise TypeError(
                        f"{path}:{lineno}: chat prompt needs a 'messages' list"
                    )
            else:
                raise ValueError(f"{path}:{lineno}: unknown kind {kind!r}")
            if not isinstance(rec.get("id"), str) or not isinstance(
                rec.get("max_tokens"), int
            ):
                raise TypeError(
                    f"{path}:{lineno}: 'id' (string) and 'max_tokens' (integer) are required"
                )
            prompts.append(rec)
    return prompts


def model_name(model_dir: Path) -> str:
    """Hub id from config.json `_name_or_path` when it is not a local path, else the directory name."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        name = cfg.get("_name_or_path")
        if (
            isinstance(name, str)
            and name
            and not os.path.isabs(name)
            and not os.path.exists(name)
        ):
            return name
    except (OSError, ValueError):
        pass
    return model_dir.resolve().name


def model_revision(model_dir: Path) -> str | None:
    """Hub commit recorded by `hf download --local-dir`, or None when unknown."""
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        first = meta.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if len(first) == 40 and all(c in "0123456789abcdef" for c in first):
        return first
    return None


def prompt_ids(tokenizer, rec: dict) -> list[int]:
    kwargs = rec.get("chat_template_kwargs") or {}
    if rec["kind"] == "chat":
        text = tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=True, **kwargs
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tokenizer(rec["prompt"], add_special_tokens=True)["input_ids"]
    if not ids:
        raise ValueError(f"prompt {rec['id']} tokenizes to 0 tokens")
    return [int(i) for i in ids]


def generate(
    torch,
    model,
    ids: list[int],
    max_tokens: int,
    top_n: int,
    device: str,
    fp32_logits: bool = True,
):
    tokens: list[int] = []
    tops: list[list[list[float]]] = []
    head_fp32 = model.lm_head.weight.to(torch.float32) if fp32_logits else None

    def forward(input_ids, past):
        if head_fp32 is None:
            out = model(
                input_ids=input_ids,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            return out.logits[0, -1].to(torch.float32), out.past_key_values
        # The decoder output is already final-normed (BF16); only the LM head runs in FP32.
        out = model.model(input_ids=input_ids, past_key_values=past, use_cache=True)
        last = out.last_hidden_state[0, -1].to(torch.float32)
        return head_fp32 @ last, out.past_key_values

    with torch.inference_mode():
        logits, past = forward(torch.tensor([ids], device=device), None)
        for step in range(max_tokens):
            # FP32 log_softmax of the last position's logits (FP32 LM head, or BF16 upcast).
            logprobs = torch.log_softmax(logits, dim=-1)
            chosen = int(torch.argmax(logprobs))
            # torch.topk orders exact ties arbitrarily; argmax (like HF greedy)
            # picks the lowest id. Order ties by ascending id so top-1 is always
            # the chosen token and the fixture is deterministic.
            cutoff = torch.topk(logprobs, top_n).values[-1]
            tied = torch.nonzero(logprobs >= cutoff).flatten().tolist()
            ranked = sorted(tied, key=lambda i: (-float(logprobs[i]), i))[:top_n]
            tokens.append(chosen)
            tops.append([[int(i), float(logprobs[i])] for i in ranked])
            if step + 1 == max_tokens:
                break
            logits, past = forward(torch.tensor([[chosen]], device=device), past)
    return tokens, tops


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is false"
        )
    prompts = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir)
    for key, value in (args.config_override or {}).items():
        setattr(config, key, value)
    if args.config_override:
        print(f"config override {json.dumps(args.config_override)}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, dtype=torch.bfloat16
    )
    model.to(args.device)
    model.eval()
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if args.config_override and rotary is not None:
        print(
            f"rotary {rotary.rope_type}, attention scaling {rotary.attention_scaling}",
            file=sys.stderr,
        )
    engine = f"transformers-{transformers.__version__}-bf16-{args.device}"
    if not args.bf16_logits:
        engine += "-fp32-logits"
    name = args.model_name or model_name(args.model_dir)
    revision = model_revision(args.model_dir)
    print(
        f"engine {engine}, model {name}, revision {revision or 'unknown'}",
        file=sys.stderr,
    )

    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for rec in prompts:
                ids = prompt_ids(tokenizer, rec)
                tokens, tops = generate(
                    torch,
                    model,
                    ids,
                    rec["max_tokens"],
                    args.top_logprobs,
                    args.device,
                    not args.bf16_logits,
                )
                captured = (
                    datetime.datetime.now(datetime.UTC)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
                line = {"id": rec["id"], "engine": engine, "model": name}
                if revision is not None:
                    line["revision"] = revision
                line |= {
                    "captured": captured,
                    "prompt_token_ids": ids,
                    "tokens": tokens,
                    "top_logprobs": tops,
                }
                f.write(
                    json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                print(
                    f"{rec['id']}: {len(ids)} prompt tokens, {len(tokens)} generated",
                    file=sys.stderr,
                )
        os.replace(tmp, args.out)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as e:  # noqa: BLE001 -- any failure: exit 1, no partial file (run() removed the temp file)
        print(f"hf_reference.py: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
