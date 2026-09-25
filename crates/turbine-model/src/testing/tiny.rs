//! The tiny synthetic checkpoint (P1 S-12): a random-weight `LlamaForCausalLM` (2 layers,
//! hidden 64, GQA with 4 query / 2 KV heads, head_dim 16 (128 via [`TinyOptions::head_dim`]),
//! llama3 rope scaling, tied embeddings)
//! with a byte-level BPE tokenizer and the Llama-3.2 chat template, written in the Hugging Face
//! layout so the loader, executor, server and `hf_reference.py` all run on it without weights.
//! [`write_tiny_olmoe`] writes the mixture-of-experts sibling (`OlmoeForCausalLM`, P2 S-16) with
//! the same tokenizer.
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
use crate::loader::weight_slots;

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

/// Writes the tiny `OlmoeForCausalLM` checkpoint into `dir`: 2 layers, hidden 64, 4 query and
/// 4 KV heads of dimension 16 with Q/K norm, 8 SwiGLU experts of width 32 per layer with top-2
/// routing and `norm_topk_prob: false`, rope theta 10000 without scaling, untied `lm_head`,
/// the tiny tokenizer and [`PLAIN_CHAT_TEMPLATE`] (OLMoE's template renders no tools). Same
/// files and determinism as [`write_tiny_llama_with`].
pub fn write_tiny_olmoe(dir: &Path, seed: u64) -> TinySpec {
    let opts = TinyOptions {
        tied: false,
        template_with_tools: false,
        ..TinyOptions::default()
    };
    write_tiny(dir, seed, &olmoe_config_json(), &opts)
}

fn write_tiny(
    dir: &Path,
    seed: u64,
    config_json: &serde_json::Value,
    opts: &TinyOptions,
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
    write_weights(&dir.join("model.safetensors"), &config, seed, opts);
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

fn write_weights(path: &Path, config: &ModelArchConfig, seed: u64, opts: &TinyOptions) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for slot in weight_slots(config) {
        let data = random_bf16(&mut rng, &slot.name, &slot.shape);
        tensors.push((slot.name, slot.shape, data));
    }
    if config.tie_word_embeddings && opts.ship_lm_head {
        let shape = vec![config.vocab_size as usize, config.hidden as usize];
        let data = random_bf16(&mut rng, "lm_head.weight", &shape);
        tensors.push(("lm_head.weight".to_string(), shape, data));
    }
    for name in &opts.extra {
        let shape = vec![config.hidden as usize];
        let data = random_bf16(&mut rng, name, &shape);
        tensors.push((name.clone(), shape, data));
    }
    tensors.retain(|(name, _, _)| !opts.omit.contains(name));
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, data)| {
            let view = TensorView::new(Dtype::BF16, shape.clone(), data)
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
    use crate::config::{Architecture, MoeConfig, RopeScaling};
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
        assert_eq!(cfg.architecture, Architecture::Llama);
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
        assert_eq!(cfg.architecture, Architecture::Olmoe);
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
        let slots = crate::loader::olmoe_slots(cfg);
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
