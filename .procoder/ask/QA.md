# Questions procoder cannot answer for you

Written 2026-09-25 19:33 UTC.

Answer each one by writing a line beginning `Answer: ` under it, then
hand the file back with `procoder ask --file .procoder/ask/QA.md`.
Leave the `Key:` lines alone — they are what ties an answer to its question.

## Q1: [decision] decisions.md

Key: a0b3fe55d987
Question: P1: 16 MB Llama tokenizer.json fixture vs the gate's 5 MB file limit

- Commit it gzip-compressed (2.5 MB, `tokenizer.json.gz`), decompress in tests via a `flate2` dev-dependency; sha256 of the decompressed file checked against the pinned revision (recommended)
- Git LFS for large fixtures
- Raise the gate limit to 20 MB in .procoder/config.toml

**Answer (2026-09-25):** raise the gate limit — `.procoder/config.toml` `max_file_mb = 20`; raw tokenizer.json committed. `.prettierignore` keeps downloaded fixtures and golden files byte-identical.

Answer: Raise the gate limit to 20 MB (user).

## Q2: [decision] decisions.md

Key: 0d4ca0aa885e
Question: Phase 1 model source while Meta's gate approval is pending

- unsloth/Llama-3.2-3B-Instruct (ungated re-upload, identical weights/architecture/tokenizer, same Llama 3.2 license) — no plan or spec change beyond the repo id; switch back to meta-llama later only if wanted (recommended)
- A different ungated model family (e.g. Qwen2.5-3B-Instruct, Apache-2.0) — re-plan Phase 1 model code (bias terms, template), redo fixtures
- Wait for Meta's approval

**Answer (2026-09-25):** use unsloth/Llama-3.2-3B-Instruct at the plan-pinned revision 006f5dcd1393c3add266de40994ba96225e9689d (ungated; identical weights/architecture/tokenizer) for Phase 1 weights; meta-llama gate request is pending and may replace it later.

Answer: unsloth/Llama-3.2-3B-Instruct ungated mirror at the pinned revision (user chose the recommended option).

## Q3: [decision] decisions.md

Key: 6ce559b7a0ed
Question: Phase 1: how the Hugging Face token reaches novanas for the gated Llama download

- User writes the token to /home/piwi/.cache/huggingface/token on novanas (chmod 600) after accepting Meta's license; Claude uses it over SSH without seeing it (recommended)
- User pastes the token in chat for this one download

**Answers (2026-09-25):** standing approval for Phase 1 lab Jobs on novanas while its GPUs are free. HF token: the user logs in on novanas themselves (`hf auth login`) after Claude installed uv 0.12.19 + hf CLI (huggingface_hub 2.0.0) for piwi in ~/.local/bin; Claude never sees the token.

Answer: User logged in on novanas with `hf auth login` after Claude installed uv + hf; Claude never sees the token (user).

## Q4: [decision] decisions.md

Key: d91bfbe4dfcf
Question: Phase 1: standing approval for novanas lab Jobs (HIP library build, GPU op tests, golden runs, serve Job)?

- Yes for all Phase 1 lab Jobs on novanas while its GPUs are free; ask again if anything else holds them (recommended)
- Ask before each Job

Answer: Yes — standing approval for Phase 1 lab Jobs on novanas while GPUs are free (user).

## Q5: [decision] decisions.md

Key: 02a4fc4d7fb7
Question: Start pure-Rust parts of later phases now, ahead of phase order?

- Yes: run independent, GPU-free pieces of Phases 3/4/6 (state machines, policies, simulators, transport/protocol) on their own branches in parallel with Phase 1; merged when their phase opens, adjusted to any Phase 1–2 type changes (recommended for throughput)
- No: keep strict phase order

**Answer (2026-09-25):** Yes — run GPU-free parts of later phases ahead on their own branches (runahead/*), merged when their phase opens.

Answer: Yes, run ahead on runahead branches (user chose the recommended option).
