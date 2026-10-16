//! The tiny synthetic checkpoint (P1 S-12): a random-weight `LlamaForCausalLM` (2 layers,
//! hidden 64, GQA with 4 query / 2 KV heads, head_dim 16 (128 via [`TinyOptions::head_dim`]),
//! llama3 rope scaling, tied embeddings)
//! with a byte-level BPE tokenizer and the Llama-3.2 chat template, written in the Hugging Face
//! layout so the loader, executor, server and `hf_reference.py` all run on it without weights.
//! [`write_tiny_olmoe`] writes the mixture-of-experts sibling (`OlmoeForCausalLM`, P2 S-16) with
//! the same tokenizer, and [`write_tiny_family`] the Phase 8 families (Qwen3, Qwen3-MoE,
//! Mistral, Mixtral; Phase 2m S-11) with their own config keys and checkpoint names.
//!
//! Every function here panics on I/O failure: it is a test utility.
use std::path::{Path, PathBuf};

use half::bf16;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde_json::json;

use crate::config::{ModelArchConfig, load_model_config};

/// Byte symbols take ids 0..=255; the specials follow.
pub const BOS: (&str, u32) = ("<|begin_of_text|>", 256);
pub const END_OF_TEXT: (&str, u32) = ("<|end_of_text|>", 257);
pub const START_HEADER: (&str, u32) = ("<|start_header_id|>", 258);
pub const END_HEADER: (&str, u32) = ("<|end_header_id|>", 259);
pub const EOT: (&str, u32) = ("<|eot_id|>", 260);
pub const EOM: (&str, u32) = ("<|eom_id|>", 261);
pub const PYTHON_TAG: (&str, u32) = ("<|python_tag|>", 262);
const SPECIALS: [(&str, u32); 7] = [
    BOS,
    END_OF_TEXT,
    START_HEADER,
    END_HEADER,
    EOT,
    EOM,
    PYTHON_TAG,
];
/// 256 byte symbols + 7 specials; `config.json` `vocab_size` equals the tokenizer size.
pub const TINY_VOCAB: u32 = 263;
/// `generation_config.json` `eos_token_id`.
pub const TINY_EOS: [u32; 2] = [END_OF_TEXT.1, EOT.1];
/// `config.json` `max_position_embeddings`.
pub const TINY_MAX_POSITIONS: u32 = 512;

/// The exact `chat_template` of `meta-llama/Llama-3.2-3B-Instruct` `tokenizer_config.json`
/// (renders tools).
pub const LLAMA32_CHAT_TEMPLATE: &str = r#"{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"#;

/// A Llama-3-style template that never mentions tools (`TinyOptions::template_with_tools:
/// false`), for tests of models whose template cannot render tools.
pub const PLAIN_CHAT_TEMPLATE: &str = "{{- bos_token }}{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";

/// Variations of the tiny checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyOptions {
    /// `tie_word_embeddings`; an untied checkpoint always ships `lm_head.weight`.
    pub tied: bool,
    /// Ship `lm_head.weight` even when tied (the loader must ignore it).
    pub ship_lm_head: bool,
    /// Tensor names left out of `model.safetensors`.
    pub omit: Vec<String>,
    /// Extra BF16 `[hidden]` tensors written under these names.
    pub extra: Vec<String>,
    /// `true`: the exact Llama-3.2 template (renders tools); `false`: [`PLAIN_CHAT_TEMPLATE`].
    pub template_with_tools: bool,
    /// Explicit `config.json` `head_dim` (default 16). Hidden stays 64, so q/o projections are
    /// `[heads * head_dim, hidden]` / `[hidden, heads * head_dim]`; GPU executor tests use 128,
    /// the only head_dim the HIP attention supports.
    pub head_dim: u32,
}

impl Default for TinyOptions {
    fn default() -> TinyOptions {
        TinyOptions {
            tied: true,
            ship_lm_head: false,
            omit: Vec::new(),
            extra: Vec::new(),
            template_with_tools: true,
            head_dim: 16,
        }
    }
}

/// What was written: the directory, its parsed config and the tokenizer size.
#[derive(Clone, Debug)]
pub struct TinySpec {
    pub dir: PathBuf,
    pub config: ModelArchConfig,
    pub vocab: u32,
}

/// Writes the default tiny checkpoint (tied, Llama-3.2 template) into `dir`.
pub fn write_tiny_llama(dir: &Path, seed: u64) -> TinySpec {
    write_tiny_llama_with(dir, seed, &TinyOptions::default())
}

/// Writes `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json` and
/// `tokenizer_config.json` into `dir` (created if needed). Weights are BF16 from ChaCha8 seeded
/// by `seed`: the same seed and options give byte-identical files.
pub fn write_tiny_llama_with(dir: &Path, seed: u64, opts: &TinyOptions) -> TinySpec {
    write_tiny(
        dir,
        seed,
        &llama_config_json(opts.tied, opts.head_dim),
        opts,
    )
}

/// A quantized tiny Llama and the BF16 model it decodes to ([`write_tiny_quantized`]).
#[derive(Clone, Debug)]
pub struct TinyQuantized {
    /// The checkpoint in the packaging `quantization_config` declares.
    pub quantized: TinySpec,
    /// The same model in BF16, every quantized weight replaced by its exact dequantized values.
    pub twin: TinySpec,
}

/// Writes the tiny Llama (tied, [`PLAIN_CHAT_TEMPLATE`], `hidden` and `head_dim` as given; 4
/// query and 2 KV heads, MLP 128) into `dir/quantized`, quantized by the weight format that
/// detects `quantization_config` ([`crate::weights::WeightFormat::write_tiny`]), and its
/// dequantized BF16 twin into `dir/twin` (Phase 6a S-3 fixtures). Block-scaled formats need
/// `hidden` and `head_dim` in multiples of the block (128). Panics when the format refuses the
/// configuration or has no tiny writer.
pub fn write_tiny_quantized(
    dir: &Path,
    seed: u64,
    quantization_config: &serde_json::Value,
    hidden: u32,
    head_dim: u32,
) -> TinyQuantized {
    let opts = TinyOptions {
        template_with_tools: false,
        head_dim,
        ..TinyOptions::default()
    };
    let mut config = llama_config_json(opts.tied, head_dim);
    config["hidden_size"] = json!(hidden);
    let (qdir, tdir) = (dir.join("quantized"), dir.join("twin"));
    write_tiny(&qdir, seed, &config, &opts);
    let format = crate::weights::detect(&json!({ "quantization_config": quantization_config }))
        .unwrap_or_else(|e| panic!("{quantization_config}: {e}"));
    let written = format
        .write_tiny(&qdir, Some(&tdir))
        .unwrap_or_else(|e| panic!("{}: write_tiny: {e}", format.name()));
    assert!(written, "{} has no tiny writer", format.name());
    let spec = |dir: PathBuf| TinySpec {
        config: load_model_config(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
        dir,
        vocab: TINY_VOCAB,
    };
    TinyQuantized {
        quantized: spec(qdir),
        twin: spec(tdir),
    }
}

/// Writes the tiny `OlmoeForCausalLM` checkpoint into `dir`: 2 layers, hidden 64, 4 query and
/// 4 KV heads of dimension 16 with Q/K norm, 8 SwiGLU experts of width 32 per layer with top-2
/// routing and `norm_topk_prob: false`, rope theta 10000 without scaling, untied `lm_head`,
/// the tiny tokenizer and [`PLAIN_CHAT_TEMPLATE`] (OLMoE's template renders no tools). Same
/// files and determinism as [`write_tiny_llama_with`].
pub fn write_tiny_olmoe(dir: &Path, seed: u64) -> TinySpec {
    write_tiny_olmoe_with_head_dim(dir, seed, 16)
}

/// [`write_tiny_olmoe`] with heads of dimension `head_dim` (an explicit `config.json`
/// `head_dim` unless it is the derived 16; hidden stays 64): GPU executor tests use 128, the
/// only head_dim the HIP attention supports.
pub fn write_tiny_olmoe_with_head_dim(dir: &Path, seed: u64, head_dim: u32) -> TinySpec {
    let opts = TinyOptions {
        tied: false,
        template_with_tools: false,
        head_dim,
        ..TinyOptions::default()
    };
    let mut config = olmoe_config_json();
    if head_dim != 16 {
        config["head_dim"] = json!(head_dim);
    }
    write_tiny(dir, seed, &config, &opts)
}

/// [`write_tiny_quantized`] for the tiny OLMoE (8 experts, top-2) with the given `hidden` size
/// and expert width `inter` (both multiples of 128 for the block-scaled formats; heads of
/// dimension 128): the checkpoint quantized as `quantization_config` says, in `dir/quantized`,
/// and its dequantized BF16 twin in `dir/twin` (Phase 7 S-4 fixtures).
pub fn write_tiny_olmoe_quantized(
    dir: &Path,
    seed: u64,
    quantization_config: &serde_json::Value,
    hidden: u32,
    inter: u32,
) -> TinyQuantized {
    let opts = TinyOptions {
        tied: false,
        template_with_tools: false,
        head_dim: 128,
        ..TinyOptions::default()
    };
    let mut config = olmoe_config_json();
    config["head_dim"] = json!(128);
    config["hidden_size"] = json!(hidden);
    config["intermediate_size"] = json!(inter);
    let (qdir, tdir) = (dir.join("quantized"), dir.join("twin"));
    write_tiny(&qdir, seed, &config, &opts);
    let format = crate::weights::detect(&json!({ "quantization_config": quantization_config }))
        .unwrap_or_else(|e| panic!("{quantization_config}: {e}"));
    let written = format
        .write_tiny(&qdir, Some(&tdir))
        .unwrap_or_else(|e| panic!("{}: write_tiny: {e}", format.name()));
    assert!(written, "{} has no tiny writer", format.name());
    let spec = |dir: PathBuf| TinySpec {
        config: load_model_config(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
        dir,
        vocab: TINY_VOCAB,
    };
    TinyQuantized {
        quantized: spec(qdir),
        twin: spec(tdir),
    }
}

/// [`write_tiny_olmoe_quantized`] for the tiny Qwen3-MoE (per-head Q/K norm, 8 experts top-2,
/// `norm_topk_prob: true`; 4 query / 2 KV heads of dimension 128) with the given `hidden` size
/// and `moe_intermediate_size` `inter` (multiples of 128 for the block-scaled formats; `inter`
/// 256 lets a tensor-parallel pair of ranks hold whole 128 blocks): the checkpoint quantized as
/// `quantization_config` says, in `dir/quantized`, and its dequantized BF16 twin in `dir/twin`
/// (Phase 7 S-5 fixtures).
pub fn write_tiny_qwen3moe_quantized(
    dir: &Path,
    seed: u64,
    quantization_config: &serde_json::Value,
    hidden: u32,
    inter: u32,
) -> TinyQuantized {
    let opts = TinyOptions {
        tied: false,
        template_with_tools: false,
        head_dim: 128,
        ..TinyOptions::default()
    };
    let mut config = family_config_json("qwen3_moe");
    config["head_dim"] = json!(128);
    config["hidden_size"] = json!(hidden);
    config["moe_intermediate_size"] = json!(inter);
    let (qdir, tdir) = (dir.join("quantized"), dir.join("twin"));
    write_tiny(&qdir, seed, &config, &opts);
    let format = crate::weights::detect(&json!({ "quantization_config": quantization_config }))
        .unwrap_or_else(|e| panic!("{quantization_config}: {e}"));
    let written = format
        .write_tiny(&qdir, Some(&tdir))
        .unwrap_or_else(|e| panic!("{}: write_tiny: {e}", format.name()));
    assert!(written, "{} has no tiny writer", format.name());
    let spec = |dir: PathBuf| TinySpec {
        config: load_model_config(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
        dir,
        vocab: TINY_VOCAB,
    };
    TinyQuantized {
        quantized: spec(qdir),
        twin: spec(tdir),
    }
}

/// [`write_tiny_quantized`] for the tiny checkpoint of the Phase 7 family `family` (see
/// [`write_tiny_family`]; hidden 64, expert width 32): the checkpoint quantized as
/// `quantization_config` says in `dir/quantized` and its dequantized BF16 twin in `dir/twin`
/// (Phase 7 S-7 fixtures: the tiny Mixtral with FP8 tensor experts, static W8A8). Formats
/// without block scales only: the tiny dims are not multiples of 128. Panics when the format
/// refuses the checkpoint or has no tiny writer.
pub fn write_tiny_family_quantized(
    dir: &Path,
    family: &str,
    seed: u64,
    quantization_config: &serde_json::Value,
) -> TinyQuantized {
    let opts = TinyOptions {
        tied: family == "qwen3",
        template_with_tools: false,
        ..TinyOptions::default()
    };
    let (qdir, tdir) = (dir.join("quantized"), dir.join("twin"));
    write_tiny(&qdir, seed, &family_config_json(family), &opts);
    let format = crate::weights::detect(&json!({ "quantization_config": quantization_config }))
        .unwrap_or_else(|e| panic!("{quantization_config}: {e}"));
    let written = format
        .write_tiny(&qdir, Some(&tdir))
        .unwrap_or_else(|e| panic!("{}: write_tiny: {e}", format.name()));
    assert!(written, "{} has no tiny writer", format.name());
    let spec = |dir: PathBuf| TinySpec {
        config: load_model_config(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
        dir,
        vocab: TINY_VOCAB,
    };
    TinyQuantized {
        quantized: spec(qdir),
        twin: spec(tdir),
    }
}

/// Writes the tiny `Qwen3_5ForConditionalGeneration` checkpoint into `dir` (Phase 7 S-11): 8
/// layers with `full_attention_interval` 4 (layers 3 and 7 full attention, the rest Gated
/// DeltaNet), hidden 64, 4 query / 2 KV heads of dimension 64 (`rotary_dim` 16 via
/// `partial_rotary_factor` 0.25), the full layers' `q_proj` per-head interleaved `[q_j | g_j]`
/// (`attn_output_gate`) and per-head Q/K norm, the linear layers' Gated DeltaNet projections
/// (2 key / 4 value heads of dimension 8, conv window 4), untied `lm_head`, the tiny tokenizer
/// and [`PLAIN_CHAT_TEMPLATE`]. The tensors are named as `Qwen/Qwen3.5-9B` stores them, under
/// `model.language_model`. Same files and determinism as [`write_tiny_llama_with`].
pub fn write_tiny_qwen3_5(dir: &Path, seed: u64) -> TinySpec {
    write_tiny_qwen3_5_with(dir, seed, &TinyOptions::default())
}

/// [`write_tiny_qwen3_5`] with extra files and tensors ([`TinyOptions::extra`], e.g. the
/// unused `model.visual.*` and `mtp.*` tensors the family's `skipped_prefixes` count).
pub fn write_tiny_qwen3_5_with(dir: &Path, seed: u64, extra: &TinyOptions) -> TinySpec {
    let opts = TinyOptions {
        tied: false,
        template_with_tools: false,
        head_dim: 64,
        extra: extra.extra.clone(),
        omit: extra.omit.clone(),
        ship_lm_head: extra.ship_lm_head,
    };
    write_tiny_slots(
        dir,
        seed,
        &qwen3_5_config_json(),
        &opts,
        Some(crate::families::qwen3_5::qwen3_5_checkpoint_slots),
    )
}

/// `config.json` of [`write_tiny_qwen3_5`]: the keys of `Qwen/Qwen3.5-9B` at tiny size, under
/// the multimodal wrapper's `text_config`.
fn qwen3_5_config_json() -> serde_json::Value {
    let text = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5_text",
        "attention_bias": false,
        "attn_output_gate": true,
        "bos_token_id": BOS.1,
        "eos_token_id": TINY_EOS,
        "full_attention_interval": 4,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 8,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": TINY_MAX_POSITIONS,
        "num_attention_heads": 4,
        "num_hidden_layers": 8,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": TINY_VOCAB,
    });
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text,
        "tie_word_embeddings": false,
    })
}

/// Writes the tiny `GptOssForCausalLM` checkpoint into `dir` (Phase 7 S-8): 4 layers alternating
/// a sliding window of `sliding_window` positions (layers 0 and 2) and full attention, hidden
/// 64, 4 query / 2 KV heads of dimension 64 with biased projections and one sink logit per
/// query head, YaRN (factor 4 over 128 positions), 4 MXFP4 experts of width 32 per layer with
/// top-2 routing, biases and the clamped activation, a router bias, untied `lm_head`, the tiny
/// tokenizer and [`PLAIN_CHAT_TEMPLATE`]. The tensors are the OpenAI checkpoint's, named and
/// shaped as `openai/gpt-oss-20b` stores them (`mlp.experts.gate_up_proj_blocks` and so on).
/// Same files and determinism as [`write_tiny_llama_with`].
pub fn write_tiny_gpt_oss(dir: &Path, seed: u64, sliding_window: u32) -> TinySpec {
    write_tiny_gpt_oss_with(dir, seed, sliding_window, 32)
}

/// [`write_tiny_gpt_oss`] with an explicit `intermediate_size`. The default 32 keeps the tiny
/// gpt-oss checkpoints of the transformers trace fixtures; the HIP executor tests need 64
/// instead: the expert MXFP4 qgemm kernel takes k in steps of 64, so the down projection
/// (k = `inter`) refuses an expert width of 32 (`NoProvider` at registry build).
pub fn write_tiny_gpt_oss_with(
    dir: &Path,
    seed: u64,
    sliding_window: u32,
    expert_inter: i64,
) -> TinySpec {
    let opts = TinyOptions {
        tied: false,
        template_with_tools: false,
        head_dim: 64,
        ..TinyOptions::default()
    };
    write_tiny(
        dir,
        seed,
        &gpt_oss_config_json(sliding_window, expert_inter),
        &opts,
    )
}

/// `config.json` of [`write_tiny_gpt_oss`]: the keys of `openai/gpt-oss-20b` at tiny size.
fn gpt_oss_config_json(sliding_window: u32, inter: i64) -> serde_json::Value {
    json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "attention_bias": true,
        "attention_dropout": 0.0,
        "bos_token_id": BOS.1,
        "eos_token_id": TINY_EOS,
        "experts_per_token": 2,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initial_context_length": 128,
        "initializer_range": 0.02,
        "intermediate_size": inter,
        "layer_types": [
            "sliding_attention", "full_attention", "sliding_attention", "full_attention"
        ],
        "max_position_embeddings": TINY_MAX_POSITIONS,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "output_router_logits": false,
        "pad_token_id": END_OF_TEXT.1,
        "quantization_config": {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head"
            ],
            "quant_method": "mxfp4"
        },
        "rms_norm_eps": 1e-5,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "rope_type": "yarn",
            "truncate": false
        },
        "rope_theta": 150000,
        "router_aux_loss_coef": 0.9,
        "sliding_window": sliding_window,
        "swiglu_limit": 7.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": TINY_VOCAB,
    })
}

/// The families [`write_tiny_family`] writes, by registry name.
pub const TINY_PHASE8_FAMILIES: [&str; 4] = ["qwen3", "qwen3_moe", "mistral", "mixtral"];

/// Writes the tiny checkpoint of the Phase 8 family `family` (a registry name of
/// [`TINY_PHASE8_FAMILIES`]) into `dir`, with the tiny tokenizer and [`PLAIN_CHAT_TEMPLATE`]:
/// 2 layers, hidden 64, 4 query / 2 KV heads of dimension 16, rope theta 1e6 without scaling,
/// keys as in the family's `config.json` (`tests/fixtures/<slug>/config.json`):
///
/// - `qwen3`: per-head Q/K norm, explicit `head_dim`, tied embeddings, MLP 128.
/// - `qwen3_moe`: per-head Q/K norm, 8 experts of `moe_intermediate_size` 32, top-2 with
///   `norm_topk_prob: true`, untied.
/// - `mistral`: the Llama layer, `sliding_window: null`, untied, MLP 128.
/// - `mixtral`: 8 `num_local_experts` of `intermediate_size` 32, top-2, untied,
///   `block_sparse_moe` names.
///
/// Same files and determinism as [`write_tiny_llama_with`]. Panics for another family.
pub fn write_tiny_family(dir: &Path, family: &str, seed: u64) -> TinySpec {
    write_tiny_family_with_head_dim(dir, family, seed, 16)
}

/// [`write_tiny_family`] with heads of dimension `head_dim` (the explicit `config.json`
/// `head_dim`; hidden stays 64): GPU executor tests use 128, the only head_dim the HIP
/// attention supports.
pub fn write_tiny_family_with_head_dim(
    dir: &Path,
    family: &str,
    seed: u64,
    head_dim: u32,
) -> TinySpec {
    let opts = TinyOptions {
        tied: family == "qwen3",
        template_with_tools: false,
        head_dim,
        ..TinyOptions::default()
    };
    let mut config = family_config_json(family);
    if head_dim != 16 {
        config["head_dim"] = json!(head_dim);
    }
    write_tiny(dir, seed, &config, &opts)
}

/// `config.json` of [`write_tiny_family`].
fn family_config_json(family: &str) -> serde_json::Value {
    let (architecture, extra) = match family {
        "qwen3" => (
            "Qwen3ForCausalLM",
            json!({
                "attention_bias": false, "head_dim": 16, "rope_scaling": null,
                "tie_word_embeddings": true, "use_sliding_window": false, "max_window_layers": 2,
            }),
        ),
        "qwen3_moe" => (
            "Qwen3MoeForCausalLM",
            json!({
                "attention_bias": false, "head_dim": 16, "rope_scaling": null,
                "use_sliding_window": false, "max_window_layers": 2, "decoder_sparse_step": 1,
                "mlp_only_layers": [], "moe_intermediate_size": 32, "norm_topk_prob": true,
                "num_experts": 8, "num_experts_per_tok": 2, "output_router_logits": false,
                "router_aux_loss_coef": 0.001,
            }),
        ),
        "mistral" => ("MistralForCausalLM", json!({})),
        "mixtral" => (
            "MixtralForCausalLM",
            json!({
                "intermediate_size": 32, "num_local_experts": 8, "num_experts_per_tok": 2,
                "output_router_logits": false, "router_aux_loss_coef": 0.02,
            }),
        ),
        other => panic!("no tiny checkpoint for family {other}"),
    };
    let mut cfg = json!({
        "architectures": [architecture],
        "model_type": family,
        "attention_dropout": 0.0,
        "bos_token_id": BOS.1,
        "eos_token_id": TINY_EOS,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "max_position_embeddings": TINY_MAX_POSITIONS,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": TINY_VOCAB,
    });
    for (key, value) in extra.as_object().expect("object") {
        cfg[key] = value.clone();
    }
    cfg
}

fn write_tiny(
    dir: &Path,
    seed: u64,
    config_json: &serde_json::Value,
    opts: &TinyOptions,
) -> TinySpec {
    write_tiny_slots(dir, seed, config_json, opts, None)
}

/// [`write_tiny`] with an explicit checkpoint tensor layout builder (the family's
/// [`crate::loader::WeightSlot`]s when its load layout differs, e.g. Qwen3.5's de-interleaved
/// `q_proj`): `checkpoint_slots` builds the tensors to write, `None` the family's own
/// [`crate::families::ModelFamily::weight_slots`].
fn write_tiny_slots(
    dir: &Path,
    seed: u64,
    config_json: &serde_json::Value,
    opts: &TinyOptions,
    checkpoint_slots: Option<fn(&ModelArchConfig) -> Vec<crate::loader::WeightSlot>>,
) -> TinySpec {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    write_json(&dir.join("config.json"), config_json);
    write_json(
        &dir.join("generation_config.json"),
        &json!({
            "bos_token_id": BOS.1,
            "eos_token_id": TINY_EOS,
            "do_sample": true,
            "temperature": 0.6,
            "top_p": 0.9,
        }),
    );
    write_json(&dir.join("tokenizer.json"), &tokenizer_json());
    write_json(
        &dir.join("tokenizer_config.json"),
        &tokenizer_config_json(opts.template_with_tools),
    );

    let config =
        load_model_config(dir).unwrap_or_else(|e| panic!("tiny config does not parse: {e}"));
    let slots = match checkpoint_slots {
        Some(build) => build(&config),
        None => config.family.0.weight_slots(&config),
    };
    write_weights(&dir.join("model.safetensors"), &config, seed, opts, &slots);
    TinySpec {
        dir: dir.to_path_buf(),
        config,
        vocab: TINY_VOCAB,
    }
}

fn write_json(path: &Path, value: &serde_json::Value) {
    let mut text = serde_json::to_string_pretty(value).expect("serialize JSON");
    text.push('\n');
    std::fs::write(path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Keys as in `allenai/OLMoE-1B-7B-0125-Instruct` `config.json` (no `head_dim`: 64 / 4 = 16).
fn olmoe_config_json() -> serde_json::Value {
    json!({
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": BOS.1,
        "clip_qkv": null,
        "eos_token_id": TINY_EOS,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": TINY_MAX_POSITIONS,
        "norm_topk_prob": false,
        "num_attention_heads": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "output_router_logits": false,
        "pad_token_id": END_OF_TEXT.1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": null,
        "rope_theta": 10000.0,
        "router_aux_loss_coef": 0.01,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": TINY_VOCAB,
    })
}

fn llama_config_json(tied: bool, head_dim: u32) -> serde_json::Value {
    json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": BOS.1,
        "eos_token_id": TINY_EOS,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "max_position_embeddings": TINY_MAX_POSITIONS,
        "mlp_bias": false,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 32,
            "rope_type": "llama3"
        },
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": TINY_VOCAB,
    })
}

/// GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to U+0100 onwards.
fn byte_symbols() -> Vec<char> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let mut next = 256u32;
    (0u32..256)
        .map(|b| {
            let code = if printable(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(code).expect("valid code point")
        })
        .collect()
}

fn tokenizer_json() -> serde_json::Value {
    let mut vocab = serde_json::Map::new();
    for (id, symbol) in byte_symbols().into_iter().enumerate() {
        vocab.insert(symbol.to_string(), json!(id));
    }
    let added: Vec<serde_json::Value> = SPECIALS
        .iter()
        .map(|(content, id)| {
            json!({
                "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect();
    let bos = |type_id: u32| json!({"SpecialToken": {"id": BOS.0, "type_id": type_id}});
    let seq = |id: &str, type_id: u32| json!({"Sequence": {"id": id, "type_id": type_id}});
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [bos(0), seq("A", 0)],
            "pair": [bos(0), seq("A", 0), bos(1), seq("B", 1)],
            "special_tokens": {
                BOS.0: {"id": BOS.0, "ids": [BOS.1], "tokens": [BOS.0]}
            }
        },
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": []
        }
    })
}

fn tokenizer_config_json(template_with_tools: bool) -> serde_json::Value {
    let mut decoder = serde_json::Map::new();
    for (content, id) in SPECIALS {
        decoder.insert(
            id.to_string(),
            json!({
                "content": content, "lstrip": false, "normalized": false, "rstrip": false,
                "single_word": false, "special": true
            }),
        );
    }
    let template = if template_with_tools {
        LLAMA32_CHAT_TEMPLATE
    } else {
        PLAIN_CHAT_TEMPLATE
    };
    // `clean_up_tokenization_spaces: false` keeps transformers' decoded text identical to the
    // tokenizers crate's (which has no clean-up step).
    json!({
        "added_tokens_decoder": decoder,
        "bos_token": BOS.0,
        "chat_template": template,
        "clean_up_tokenization_spaces": false,
        "eos_token": EOT.0,
        "model_input_names": ["input_ids", "attention_mask"],
        "model_max_length": TINY_MAX_POSITIONS,
        "tokenizer_class": "PreTrainedTokenizerFast"
    })
}

/// Uniform in `[-1, 1)` from 24 random bits.
fn unit(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
}

/// BF16 bytes of a tensor: norms are `1 ± 0.1`, the embedding `U(-1, 1)`, projections
/// `U(-1, 1) · sqrt(3 / fan_in)` (unit output variance), so logits are decisive rather than
/// near-ties.
fn random_bf16(rng: &mut ChaCha8Rng, name: &str, shape: &[usize]) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    let (offset, scale) = if shape.len() == 1 {
        (1.0, 0.1)
    } else if name == "model.embed_tokens.weight" {
        (0.0, 1.0)
    } else {
        (0.0, (3.0 / shape[1] as f32).sqrt())
    };
    let mut bytes = Vec::with_capacity(numel * 2);
    for _ in 0..numel {
        let v = bf16::from_f32(offset + scale * unit(rng));
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Random bytes of a U8 slot (the OpenAI MXFP4 expert tensors, [`gpt_oss_config_json`]): the
/// E2M1 code bytes of a `_blocks` tensor are uniform; a `_scales` tensor holds E8M0 exponents
/// 120..=122 (`2^-7 … 2^-5`), so the dequantized expert weights are about as large as the
/// BF16 projections' (`≈ 0.02 … 0.09` rms).
fn random_u8(rng: &mut ChaCha8Rng, name: &str, shape: &[usize]) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    (0..numel)
        .map(|_| {
            let r = rng.next_u32();
            if name.ends_with("_scales") {
                120 + (r >> 8) as u8 % 3
            } else {
                (r >> 8) as u8
            }
        })
        .collect()
}

/// The BF16 bytes of a tensor with a name of its own (gpt-oss): a bias is `U(-0.1, 0.1)` (the
/// gate/up bias has a few outliers past the activation's clamp of 7: every 16th gate column
/// `+7.5`, every 16th up column `-7.5`, so the clamp binds), the attention sinks `U(-1, 1)`,
/// and the router bias pairs the experts 40 apart (the top-k set is bias-decided, the two
/// gates logit-driven), so a numerics difference between implementations cannot flip the
/// selected experts.
/// `None` for every other tensor.
fn random_named_bf16(rng: &mut ChaCha8Rng, name: &str, shape: &[usize]) -> Option<Vec<u8>> {
    let is_bias = name.ends_with(".bias") || name.ends_with("_bias");
    let is_router = name.ends_with("mlp.router.weight");
    if !is_bias && !name.ends_with(".sinks") && !is_router {
        return None;
    }
    let numel: usize = shape.iter().product();
    if name.ends_with("mlp.router.bias") {
        // Experts paired by bias (`40·(e/2) + e%2`): the top-k selection is decided by the
        // bias (a pair gap of 40 dwarfs any logit), so a numerics difference between
        // implementations cannot flip the selected set, while the within-pair gap of 1 keeps
        // the two gates comparable and logit-driven.
        let bytes: Vec<u8> = (0..numel)
            .flat_map(|i| bf16::from_f32(40.0 * (i / 2) as f32 + (i % 2) as f32).to_le_bytes())
            .collect();
        return Some(bytes);
    }
    let scale = if is_bias { 0.1 } else { 1.0 };
    let width = shape.last().copied().unwrap_or(1);
    let mut bytes = Vec::with_capacity(numel * 2);
    for i in 0..numel {
        let mut v = scale * unit(rng);
        if name.ends_with("gate_up_proj_bias") {
            // Interleaved columns: even = gate, odd = up.
            match (i % width, (i % width) / 2 % 16) {
                (c, 3) if c % 2 == 0 => v = 7.5,
                (c, 11) if c % 2 == 1 => v = -7.5,
                _ => {}
            }
        }
        bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
    }
    Some(bytes)
}

fn write_weights(
    path: &Path,
    config: &ModelArchConfig,
    seed: u64,
    opts: &TinyOptions,
    slots: &[crate::loader::WeightSlot],
) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let format = config.weight_format.get();
    let mut tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for slot in slots {
        // A family whose checkpoint stores some tensors packed (gpt-oss's MXFP4 experts) names
        // them U8 in its weight format.
        let (dtype, data) = if format.slot_dtype(&slot) == turbine_core::types::DType::U8 {
            (Dtype::U8, random_u8(&mut rng, &slot.name, &slot.shape))
        } else {
            let data = random_named_bf16(&mut rng, &slot.name, &slot.shape)
                .unwrap_or_else(|| random_bf16(&mut rng, &slot.name, &slot.shape));
            (Dtype::BF16, data)
        };
        tensors.push((slot.name.clone(), dtype, slot.shape.clone(), data));
    }
    if config.tie_word_embeddings && opts.ship_lm_head {
        let shape = vec![config.vocab_size as usize, config.hidden as usize];
        let data = random_bf16(&mut rng, "lm_head.weight", &shape);
        tensors.push(("lm_head.weight".to_string(), Dtype::BF16, shape, data));
    }
    for name in &opts.extra {
        let shape = vec![config.hidden as usize];
        let data = random_bf16(&mut rng, name, &shape);
        tensors.push((name.clone(), Dtype::BF16, shape, data));
    }
    tensors.retain(|(name, _, _, _)| !opts.omit.contains(name));
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            let view = TensorView::new(*dtype, shape.clone(), data)
                .unwrap_or_else(|e| panic!("tensor view {name}: {e}"));
            (name.clone(), view)
        })
        .collect();
    safetensors::serialize_to_file(views, None, path)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MoeConfig, RopeScaling};
    use crate::testing::TempDir;

    #[test]
    fn tiny_checkpoint_is_deterministic_and_parses() {
        let a = TempDir::new("tiny-a");
        let b = TempDir::new("tiny-b");
        let c = TempDir::new("tiny-c");
        let spec = write_tiny_llama(a.path(), 7);
        write_tiny_llama(b.path(), 7);
        write_tiny_llama(c.path(), 8);
        let weights = |d: &TempDir| std::fs::read(d.path().join("model.safetensors")).unwrap();
        assert_eq!(weights(&a), weights(&b));
        assert_ne!(weights(&a), weights(&c));

        let cfg = &spec.config;
        assert_eq!(cfg.family.0.name(), "llama");
        assert_eq!(
            (
                cfg.num_layers,
                cfg.hidden,
                cfg.num_attention_heads,
                cfg.num_kv_heads
            ),
            (2, 64, 4, 2)
        );
        assert_eq!((cfg.head_dim, cfg.intermediate), (16, 128));
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(
            cfg.rope_scaling,
            Some(RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 32,
            })
        );
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.vocab_size, TINY_VOCAB);
        assert_eq!(spec.vocab, TINY_VOCAB);
        assert_eq!(cfg.max_position_embeddings, TINY_MAX_POSITIONS);
        assert_eq!(cfg.eos_token_ids.as_slice(), &TINY_EOS);
    }

    #[test]
    fn head_dim_option_decouples_attention_width_from_hidden() {
        let dir = TempDir::new("tiny-hd128");
        let opts = TinyOptions {
            head_dim: 128,
            ..TinyOptions::default()
        };
        let spec = write_tiny_llama_with(dir.path(), 7, &opts);
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(raw["head_dim"], 128);
        let cfg = &spec.config;
        assert_eq!(
            (
                cfg.hidden,
                cfg.num_attention_heads,
                cfg.num_kv_heads,
                cfg.head_dim
            ),
            (64, 4, 2, 128)
        );
        let bytes = std::fs::read(dir.path().join("model.safetensors")).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let shape = |name: &str| st.tensor(name).unwrap().shape().to_vec();
        let p = "model.layers.0.self_attn";
        assert_eq!(shape(&format!("{p}.q_proj.weight")), [512, 64]);
        assert_eq!(shape(&format!("{p}.k_proj.weight")), [256, 64]);
        assert_eq!(shape(&format!("{p}.v_proj.weight")), [256, 64]);
        assert_eq!(shape(&format!("{p}.o_proj.weight")), [64, 512]);
    }

    #[test]
    fn tiny_olmoe_is_deterministic_and_parses() {
        let a = TempDir::new("tiny-olmoe-a");
        let b = TempDir::new("tiny-olmoe-b");
        let spec = write_tiny_olmoe(a.path(), 7);
        write_tiny_olmoe(b.path(), 7);
        for file in ["model.safetensors", "config.json", "tokenizer_config.json"] {
            let read = |d: &TempDir| std::fs::read(d.path().join(file)).unwrap();
            assert_eq!(read(&a), read(&b), "{file}");
        }

        let cfg = &spec.config;
        assert_eq!(cfg.family.0.name(), "olmoe");
        assert_eq!(
            (
                cfg.num_layers,
                cfg.hidden,
                cfg.num_attention_heads,
                cfg.num_kv_heads,
                cfg.head_dim
            ),
            (2, 64, 4, 4, 16)
        );
        assert_eq!(
            cfg.moe,
            Some(MoeConfig {
                num_experts: 8,
                experts_per_token: 2,
                expert_intermediate: 32,
                norm_topk_prob: false,
            })
        );
        assert!(cfg.qk_norm);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.rope_scaling, None);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.vocab_size, TINY_VOCAB);
        assert_eq!(cfg.eos_token_ids.as_slice(), &TINY_EOS);

        // Every OLMoE slot is in the file with its shape, BF16, and nothing else is.
        let index = crate::SafetensorsIndex::open(a.path()).unwrap();
        let slots = cfg.family.0.weight_slots(cfg);
        assert_eq!(slots.len(), 1 + 2 * (9 + 8 * 3) + 2);
        assert_eq!(index.entries().count(), slots.len());
        for slot in &slots {
            let entry = index.get(&slot.name).unwrap();
            assert_eq!(entry.shape, slot.shape, "{}", slot.name);
            assert_eq!(entry.dtype, safetensors::Dtype::BF16);
        }
        cfg.check_supported_weights(&index).unwrap();

        // Transformers loads it with the fast tokenizer class and a template without tools.
        let tok_config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(a.path().join("tokenizer_config.json")).unwrap())
                .unwrap();
        assert_eq!(tok_config["tokenizer_class"], "PreTrainedTokenizerFast");
        assert_eq!(tok_config["chat_template"], PLAIN_CHAT_TEMPLATE);
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(a.path().join("config.json")).unwrap()).unwrap();
        assert_eq!(config["model_type"], "olmoe");
    }

    #[test]
    fn tiny_tokenizer_is_byte_level_with_llama_specials() {
        let dir = TempDir::new("tiny-tok");
        write_tiny_llama(dir.path(), 1);
        let tok = tokenizers::Tokenizer::from_file(dir.path().join("tokenizer.json"))
            .expect("tiny tokenizer.json loads");
        assert_eq!(tok.get_vocab_size(true), TINY_VOCAB as usize);
        for (content, id) in SPECIALS {
            assert_eq!(tok.token_to_id(content), Some(id), "{content}");
        }
        // One token per byte, BOS prepended by the post-processor.
        let enc = tok.encode("ab é", true).expect("encode");
        assert_eq!(enc.get_ids(), &[256, 97, 98, 32, 0xC3, 0xA9]);
        let enc = tok
            .encode("<|start_header_id|>user<|end_header_id|>", false)
            .expect("encode specials");
        assert_eq!(
            enc.get_ids(),
            &[258, b'u' as u32, b's' as u32, b'e' as u32, b'r' as u32, 259]
        );
        let text = tok
            .decode(&[256, 104, 105, 32, 0xC3, 0xA9, 260], true)
            .expect("decode");
        assert_eq!(text, "hi é");

        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.path().join("tokenizer_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config["chat_template"], LLAMA32_CHAT_TEMPLATE);
        assert_eq!(config["bos_token"], "<|begin_of_text|>");
        assert_eq!(config["eos_token"], "<|eot_id|>");

        let plain = TempDir::new("tiny-plain");
        let opts = TinyOptions {
            template_with_tools: false,
            ..TinyOptions::default()
        };
        write_tiny_llama_with(plain.path(), 1, &opts);
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(plain.path().join("tokenizer_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config["chat_template"], PLAIN_CHAT_TEMPLATE);
        assert!(!PLAIN_CHAT_TEMPLATE.contains("tools"));
    }
}
