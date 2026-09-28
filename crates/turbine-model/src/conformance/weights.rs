//! The `weight_format` suite: the tiny checkpoint of at least one registered model family is in
//! the format, loads through it, and uploads exactly the bytes the format's `weight_bytes`
//! (packed data plus scales) predicts for the memory budget.

use std::sync::Arc;

use turbine_core::registry::Registry;
use turbine_core::types::DeviceId;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

use super::{ConformanceFailure, Report, ensure};
use crate::testing::TempDir;
use crate::weights::WeightFormat;
use crate::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, families};

const SEED: u64 = 11;

/// Runs every check over every format of `reg`; `Err` lists each broken check.
///
/// Per format: `tiny` (the `config.json` of at least one registered family's tiny checkpoint
/// declares the format, and every such checkpoint loads through [`WeightLoader::load_format`]
/// with every tensor accepted and mapped to a slot) and `bytes` (for each of those checkpoints
/// the uploaded bytes equal [`WeightFormat::weight_bytes`], packed data and scales included).
pub fn weights_suite(reg: &Registry<dyn WeightFormat>) -> Result<(), Vec<ConformanceFailure>> {
    let mut report = Report::new(reg);
    let tmp = TempDir::new("turbine-conformance-weights");
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    for format in reg.iter() {
        let name = format.name();
        let mut byte_counts: Vec<(&'static str, u64, u64)> = Vec::new();
        report.check(name, "tiny", || {
            let mut loaded = Vec::new();
            for family in families::registry().iter() {
                let dir = tmp.path().join(format!("{name}-{}", family.name()));
                let spec = family.write_tiny(&dir, SEED);
                let text = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
                let top: serde_json::Value =
                    serde_json::from_slice(&text).map_err(|e| e.to_string())?;
                if format.check_config(&top).is_err() {
                    continue;
                }
                let index = SafetensorsIndex::open(&dir).map_err(|e| e.to_string())?;
                let slots = family.weight_slots(&spec.config);
                let weights =
                    WeightLoader::load_format(format, &index, &slots, &mem, MAX_STAGING_BYTES)
                        .map_err(|e| format!("{}: {e}", family.name()))?;
                ensure(weights.unexpected.is_empty(), || {
                    format!("{}: unexpected {:?}", family.name(), weights.unexpected)
                })?;
                byte_counts.push((
                    family.name(),
                    format.weight_bytes(&spec.config),
                    weights.weight_bytes,
                ));
                loaded.push(family.name());
            }
            ensure(!loaded.is_empty(), || {
                "no registered family's tiny checkpoint declares this format".into()
            })
        });
        report.check(name, "bytes", || {
            for (family, predicted, uploaded) in &byte_counts {
                ensure(predicted == uploaded, || {
                    format!("{family}: weight_bytes {predicted} but {uploaded} bytes uploaded")
                })?;
            }
            Ok(())
        });
    }
    report.finish()
}
