//! Validates real checkpoints the way the loader does, without uploading a weight or touching a
//! GPU: `config.json` (format detection and parsing), the dtype allowlist
//! (`check_supported_weights`), the loader's planning (`WeightLoader::plan`: tensors present,
//! dtypes, shapes, as-stored byte lengths, stacks) and every repacked slot's
//! `WeightFormat::repack_with` (scales, zero points, check-only `g_idx` / `weight_shape` /
//! `qzeros`, packed data), reading only those tensors. Prints one line per directory:
//! `ok <dir> …` or `FAIL <dir>: <error>`; exits 1 when any fails.
//!
//! usage: cargo run --release -p turbine-model --example check_checkpoint -- <model-dir>...

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use turbine_model::config::load_model_config;
use turbine_model::{ModelError, SafetensorsIndex, TensorEntry, WeightLoader, WeightSlot};

fn read(entry: &TensorEntry) -> Result<Vec<u8>, ModelError> {
    let io = |e: std::io::Error| ModelError::Safetensors {
        file: entry.file.clone(),
        tensor: entry.name.clone(),
        rule: e.to_string(),
    };
    let mut f = std::fs::File::open(&entry.file).map_err(io)?;
    f.seek(SeekFrom::Start(entry.range.start)).map_err(io)?;
    let mut buf = vec![0u8; entry.byte_len() as usize];
    f.read_exact(&mut buf).map_err(io)?;
    Ok(buf)
}

fn check(dir: &Path) -> Result<String, ModelError> {
    let cfg = load_model_config(dir)?;
    let index = SafetensorsIndex::open(dir)?;
    cfg.check_supported_weights(&index)?;
    let format = cfg.weight_format.get();
    let slots: Vec<WeightSlot> = cfg
        .family
        .0
        .weight_slots(&cfg)
        .iter()
        .flat_map(|s| format.slots(s))
        .collect();
    let planned = WeightLoader::plan(format, &index, &slots)?;
    let (mut repacked, mut checked) = (0usize, 0usize);
    for (slot, entry) in &planned {
        if !format.repacks(slot) {
            continue;
        }
        let mut companions = Vec::new();
        for name in format.companions(slot) {
            let e = index
                .get(&name)
                .ok_or_else(|| ModelError::MissingTensor(name.clone()))?;
            companions.push((e, read(e)?));
        }
        let bytes = format.repack_with(slot, entry, read(entry)?, &companions)?;
        let want = slot.shape.iter().product::<usize>() * format.slot_dtype(slot).size_bytes();
        if bytes.len() != want {
            return Err(ModelError::Safetensors {
                file: entry.file.clone(),
                tensor: entry.name.clone(),
                rule: format!("repacked to {} bytes, slot needs {want}", bytes.len()),
            });
        }
        if want == 0 {
            checked += 1;
        } else {
            repacked += 1;
        }
    }
    Ok(format!(
        "format={} slots={} planned={} repacked={repacked} check_only={checked}",
        format.name(),
        slots.len(),
        planned.len()
    ))
}

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    if dirs.is_empty() {
        eprintln!("usage: check_checkpoint <model-dir>...");
        std::process::exit(2);
    }
    let mut failed = false;
    for dir in &dirs {
        match check(Path::new(dir)) {
            Ok(summary) => println!("ok {dir} {summary}"),
            Err(e) => {
                failed = true;
                println!("FAIL {dir}: {e}");
            }
        }
    }
    std::process::exit(i32::from(failed));
}
