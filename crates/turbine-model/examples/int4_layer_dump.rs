//! Dequantizes one INT4 linear layer of a real AWQ / GPTQ / compressed-tensors checkpoint the
//! way Turbine serves it: the checkpoint's own `quantization_config` configures the registered
//! weight format, the loader's slots of the layer are read and repacked (`WeightFormat::repack`,
//! which also validates `g_idx` and the symmetric zero points), and the CPU reference provider
//! dequantizes the repacked bytes (`turbine_kernels::cpu::quant::dequantize`). The `[n, k]` F32
//! result is written little-endian to `<out>`, for a cross-check against an independent
//! dequantization (Phase 6a GPTQ numerics investigation, `.procoder/handoff/p6a-gptq-numerics.md`).
//!
//! usage: cargo run --release -p turbine-model --example int4_layer_dump --
//!        <model-dir> <format: gptq|awq|ct_pack_int4> <layer, e.g. model.layers.0.self_attn.q_proj> <n> <k> <out.f32>

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use turbine_kernels::cpu::quant::dequantize;
use turbine_kernels::quant::QuantSchemeDesc;
use turbine_model::SafetensorsIndex;
use turbine_model::WeightSlot;
use turbine_model::weights::WeightFormat;

fn read(entry: &turbine_model::TensorEntry) -> Vec<u8> {
    let mut f = std::fs::File::open(&entry.file).expect("open shard");
    f.seek(SeekFrom::Start(entry.range.start)).expect("seek");
    let mut buf = vec![0u8; entry.byte_len() as usize];
    f.read_exact(&mut buf).expect("read tensor");
    buf
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir, format, layer, n, k, out] = args.as_slice() else {
        eprintln!(
            "usage: int4_layer_dump <model-dir> <gptq|awq|ct_pack_int4> <layer> <n> <k> <out.f32>"
        );
        std::process::exit(2);
    };
    let (n, k): (usize, usize) = (n.parse().expect("n"), k.parse().expect("k"));
    let dir = Path::new(dir);
    let top: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("config.json is JSON");
    let entry: &dyn WeightFormat = match format.as_str() {
        "gptq" => &turbine_model::weights::gptq::GPTQ,
        "awq" => &turbine_model::weights::awq::AWQ,
        "ct_pack_int4" => &turbine_model::weights::ct_pack_int4::CT_PACK_INT4,
        other => panic!("unknown format {other}"),
    };
    entry
        .check_config(&top)
        .expect("format claims the checkpoint");
    let fmt = entry.configure(&top).expect("configure");
    eprintln!("format: {}", fmt.describe());
    let index = SafetensorsIndex::open(dir).expect("open checkpoint");
    let base = WeightSlot {
        name: format!("{layer}.weight"),
        shape: vec![n, k],
        stack: None,
        source: None,
    };
    let mut stored = Vec::new();
    for slot in fmt.slots(&base) {
        let Some(e) = index.get(&slot.name) else {
            assert!(fmt.optional(&slot), "{} not in the checkpoint", slot.name);
            continue;
        };
        fmt.check_tensor(e).expect("check_tensor");
        let bytes = fmt.repack(&slot, e, read(e)).expect("repack");
        eprintln!(
            "slot {} {:?} -> {} bytes",
            slot.name,
            slot.shape,
            bytes.len()
        );
        if slot.shape.iter().product::<usize>() != 0 {
            stored.push(bytes);
        }
    }
    let scales: Vec<f32> = stored[1]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let group = (k / (scales.len() / n)) as u32;
    let (scheme, zeros) = match stored.get(2) {
        Some(z) => (QuantSchemeDesc::Int4GroupZp { group }, Some(z.as_slice())),
        None => (QuantSchemeDesc::Int4GroupSym { group }, None),
    };
    eprintln!("scheme {scheme:?}");
    let w = dequantize(scheme, &stored[0], &scales, zeros, n, k);
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes).expect("write out");
    eprintln!("wrote {out} ({n} x {k} f32)");
}
