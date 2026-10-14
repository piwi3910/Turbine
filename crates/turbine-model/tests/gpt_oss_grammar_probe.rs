//! Diagnosis (dropped with the fix): the harmony required grammar's mask on the real gpt-oss
//! tokenizer. Run on novanas with
//! `TURBINE_GPT_OSS_TOKENIZER=/home/piwi/turbine-models/gpt-oss-20b/tokenizer.json`.

use turbine_model::formats::Harmony;
use turbine_model::structured::{GrammarCompiler, GrammarLimits, TokenMask, step_mask};
use turbine_model::tokenizer::Tokenizer;
use turbine_model::{ToolChoice, ToolFormat};

#[test]
#[ignore = "needs the gpt-oss tokenizer.json (TURBINE_GPT_OSS_TOKENIZER, \
            TURBINE_TEST_GPT_OSS_MODEL_DIR or the lab paths)"]
fn required_mask_probe() {
    let path = std::env::var("TURBINE_GPT_OSS_TOKENIZER")
        .or_else(|_| {
            std::env::var("TURBINE_TEST_GPT_OSS_MODEL_DIR")
                .map(|dir| format!("{dir}/tokenizer.json"))
        })
        .unwrap_or_else(|_| "/models/gpt-oss-20b/tokenizer.json".to_string());
    let tokenizer = Tokenizer::from_file(std::path::Path::new(&path))
        .or_else(|e| {
            Tokenizer::from_file(std::path::Path::new(
                "/home/piwi/turbine-models/gpt-oss-20b/tokenizer.json",
            ))
            .map_err(|_| e)
        })
        .expect("tokenizer");
    let eos = [200002u32, 200012u32, 199999u32];
    let compiler = GrammarCompiler::new(&tokenizer, &eos).expect("trie");
    let limits = GrammarLimits {
        max_schema_bytes: 64 * 1024,
    };
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    })];
    let spec = Harmony
        .grammar(&tools, &ToolChoice::Required, false)
        .expect("grammar");
    println!(
        "grammar source:\n{}",
        match &spec {
            turbine_core::request::ConstraintSpec::ToolCall { grammar_source } =>
                grammar_source.clone(),
            _ => String::new(),
        }
    );
    let mut matcher = compiler.compile(&spec, &limits).expect("compile");
    // Walk the whole grammar the way the engine does: forced runs feed without the model,
    // every sampled step takes a masked token. The first empty mask is the failure the
    // server reports as "the constraint allows no token".
    let v = tokenizer.vocab_size() as usize;
    for step in 0..200 {
        let mut mask = TokenMask::new_none(v);
        if let Err(e) = step_mask(matcher.as_mut(), &eos, &mut mask) {
            println!("step {step}: MASK ERROR: {e}");
            return;
        }
        let allowed: Vec<u32> = (0..v as u32).filter(|&id| mask.is_allowed(id)).collect();
        let forced = matcher.forced_tokens(usize::MAX);
        println!(
            "step {step}: allowed {} {:?} forced {forced:?}",
            allowed.len(),
            &allowed[..allowed.len().min(3)],
        );
        let tok = if forced.is_empty() {
            allowed[0]
        } else {
            forced[0]
        };
        if matches!(tok, 200012 | 200002 | 199999) {
            println!("step {step}: end token {tok}");
            return;
        }
        if let Err(e) = matcher.commit(tok) {
            println!("step {step}: COMMIT ERROR at {tok}: {e}");
            return;
        }
    }
}
