//! Safetensors checkpoint index (P1 S-2): reads `model.safetensors.index.json` and its shards, or
//! a single `model.safetensors`, and validates every header before any tensor byte is read —
//! header size bound, known dtype, shape × dtype size equal to the byte range, ranges inside the
//! file and non-overlapping, every tensor listed once. Headers are read with positioned reads (no
//! `mmap`); tensor data is read later by the weight loader. Pickle formats (`*.bin`, `*.pt`,
//! `*.pth`, `*.ckpt`) are never opened (TS §16).
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub use ::safetensors::Dtype;
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};

use crate::ModelError;

/// Single-file checkpoint name.
pub const SINGLE_FILE: &str = "model.safetensors";
/// Sharded checkpoint index name.
pub const INDEX_FILE: &str = "model.safetensors.index.json";
/// Headers larger than this are rejected from the 8-byte length prefix, before allocating.
pub const MAX_HEADER_BYTES: u64 = 100 << 20;
/// Largest index JSON read into memory (real indexes are tens of KiB).
pub const MAX_INDEX_BYTES: u64 = 16 << 20;
/// The `tensor` of errors that concern a whole header rather than one tensor.
pub const HEADER_TENSOR: &str = "<header>";
/// Weight-file extensions refused without opening the file.
pub const PICKLE_EXTENSIONS: [&str; 4] = ["bin", "pt", "pth", "ckpt"];

/// One validated tensor; `range` is its absolute byte range inside `file`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub file: PathBuf,
    pub range: Range<u64>,
}

impl TensorEntry {
    pub fn byte_len(&self) -> u64 {
        self.range.end - self.range.start
    }
}

/// Every tensor of a checkpoint, sorted by name.
#[derive(Debug)]
pub struct SafetensorsIndex {
    dir: PathBuf,
    entries: Vec<TensorEntry>,
    by_name: HashMap<String, usize>,
}

impl SafetensorsIndex {
    /// Opens `<dir>/model.safetensors.index.json` (and every shard it lists), else
    /// `<dir>/model.safetensors`. A directory without either that holds a pickle file yields
    /// `ModelError::Pickle` naming it; pickle files are never opened.
    pub fn open(dir: &Path) -> Result<SafetensorsIndex, ModelError> {
        let meta = std::fs::metadata(dir).map_err(|e| io_err(dir, e))?;
        if !meta.is_dir() {
            return Err(ModelError::Io {
                path: dir.to_path_buf(),
                detail: "not a directory".into(),
            });
        }
        let index_path = dir.join(INDEX_FILE);
        let single_path = dir.join(SINGLE_FILE);
        let entries = if exists(&index_path) {
            open_sharded(dir, &index_path)?
        } else if exists(&single_path) {
            let mut all = BTreeMap::new();
            read_header(&single_path, &mut all)?;
            all.into_values().collect()
        } else {
            return Err(match find_pickle(dir)? {
                Some(path) => ModelError::Pickle { path },
                None => ModelError::Io {
                    path: dir.to_path_buf(),
                    detail: format!("no {SINGLE_FILE} or {INDEX_FILE}"),
                },
            });
        };
        let by_name = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.clone(), i))
            .collect();
        Ok(SafetensorsIndex {
            dir: dir.to_path_buf(),
            entries,
            by_name,
        })
    }

    /// The checkpoint directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// Entries sorted by name.
    pub fn entries(&self) -> impl Iterator<Item = &TensorEntry> {
        self.entries.iter()
    }

    /// Sum of every tensor's byte length.
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(TensorEntry::byte_len).sum()
    }
}

fn exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

pub(crate) fn io_err(path: &Path, e: std::io::Error) -> ModelError {
    ModelError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    }
}

fn st_err(file: &Path, tensor: &str, rule: impl Into<String>) -> ModelError {
    ModelError::Safetensors {
        file: file.to_path_buf(),
        tensor: tensor.to_string(),
        rule: rule.into(),
    }
}

/// Opens `path` for reading when it, or the target of a symlink, is a regular file. Checked
/// again on the open handle so a swap between the two checks is caught.
pub(crate) fn open_regular(path: &Path) -> Result<File, ModelError> {
    let not_regular = || ModelError::Io {
        path: path.to_path_buf(),
        detail: "not a regular file".into(),
    };
    let meta = std::fs::metadata(path).map_err(|e| io_err(path, e))?;
    if !meta.is_file() {
        return Err(not_regular());
    }
    let file = File::open(path).map_err(|e| io_err(path, e))?;
    let meta = file.metadata().map_err(|e| io_err(path, e))?;
    if !meta.is_file() {
        return Err(not_regular());
    }
    Ok(file)
}

/// Reads a whole regular file of at most `limit` bytes.
fn read_small_file(path: &Path, limit: u64) -> Result<Vec<u8>, ModelError> {
    let file = open_regular(path)?;
    let len = file.metadata().map_err(|e| io_err(path, e))?.len();
    if len > limit {
        return Err(ModelError::Io {
            path: path.to_path_buf(),
            detail: format!("file of {len} B exceeds the {limit} B limit"),
        });
    }
    let mut buf = vec![0u8; len as usize];
    file.read_exact_at(&mut buf, 0)
        .map_err(|e| io_err(path, e))?;
    Ok(buf)
}

fn is_pickle(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| PICKLE_EXTENSIONS.contains(&e))
}

/// The first pickle file of `dir` by name, found from the directory listing alone.
fn find_pickle(dir: &Path) -> Result<Option<PathBuf>, ModelError> {
    let mut found: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| io_err(dir, e))? {
        let entry = entry.map_err(|e| io_err(dir, e))?;
        if entry.file_name().to_str().is_some_and(is_pickle) {
            found.push(entry.path());
        }
    }
    found.sort();
    Ok(found.into_iter().next())
}

#[derive(Deserialize)]
struct IndexJson {
    weight_map: BTreeMap<String, String>,
}

fn open_sharded(dir: &Path, index_path: &Path) -> Result<Vec<TensorEntry>, ModelError> {
    let raw = read_small_file(index_path, MAX_INDEX_BYTES)?;
    let index: IndexJson = serde_json::from_slice(&raw).map_err(|e| ModelError::Io {
        path: index_path.to_path_buf(),
        detail: format!("malformed index: {e}"),
    })?;
    // Shard file name → the tensors the index maps to it.
    let mut shards: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (tensor, shard) in &index.weight_map {
        if is_pickle(shard) {
            return Err(ModelError::Pickle {
                path: dir.join(shard),
            });
        }
        if shard.contains('/') || shard.contains('\\') || !shard.ends_with(".safetensors") {
            return Err(st_err(
                index_path,
                tensor,
                format!("shard {shard:?} is not a .safetensors file in the checkpoint directory"),
            ));
        }
        shards
            .entry(shard.as_str())
            .or_default()
            .push(tensor.as_str());
    }
    for (shard, tensors) in &shards {
        let path = dir.join(shard);
        if !exists(&path) {
            return Err(st_err(
                &path,
                tensors[0],
                "shard listed in index is missing",
            ));
        }
    }
    let mut all = BTreeMap::new();
    for shard in shards.keys() {
        read_header(&dir.join(shard), &mut all)?;
    }
    for (tensor, shard) in &index.weight_map {
        let path = dir.join(shard);
        match all.get(tensor.as_str()) {
            Some(entry) if entry.file == path => {}
            Some(entry) => {
                return Err(st_err(
                    &entry.file,
                    tensor,
                    format!("{INDEX_FILE} maps it to {shard} but it is stored in another shard"),
                ));
            }
            None => {
                return Err(st_err(
                    &path,
                    tensor,
                    format!("listed in {INDEX_FILE} but absent from the shard"),
                ));
            }
        }
    }
    Ok(all.into_values().collect())
}

/// Every key/value pair of the header object in file order, duplicates kept (a map type would
/// silently keep only one of two equal keys).
struct HeaderPairs(Vec<(String, serde_json::Value)>);

impl<'de> Deserialize<'de> for HeaderPairs {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<HeaderPairs, D::Error> {
        struct PairsVisitor;
        impl<'de> Visitor<'de> for PairsVisitor {
            type Value = HeaderPairs;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<HeaderPairs, A::Error> {
                let mut pairs = Vec::new();
                while let Some(pair) = map.next_entry::<String, serde_json::Value>()? {
                    pairs.push(pair);
                }
                Ok(HeaderPairs(pairs))
            }
        }
        d.deserialize_map(PairsVisitor)
    }
}

/// One header entry, shaped like `safetensors::tensor::TensorInfo` but with the dtype kept as
/// text so an unknown one is reported by name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

/// Parses and validates one file's header, adding its tensors to `all` with absolute ranges.
fn read_header(path: &Path, all: &mut BTreeMap<String, TensorEntry>) -> Result<(), ModelError> {
    let file = open_regular(path)?;
    let file_len = file.metadata().map_err(|e| io_err(path, e))?.len();
    if file_len < 8 {
        return Err(st_err(
            path,
            HEADER_TENSOR,
            format!("file of {file_len} B is shorter than the 8-byte header length"),
        ));
    }
    let mut prefix = [0u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| io_err(path, e))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES {
        return Err(st_err(
            path,
            HEADER_TENSOR,
            format!("header of {header_len} bytes exceeds 100 MiB"),
        ));
    }
    if header_len > file_len - 8 {
        return Err(st_err(
            path,
            HEADER_TENSOR,
            format!("range outside file: header of {header_len} bytes in a file of {file_len} B"),
        ));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact_at(&mut header, 8)
        .map_err(|e| io_err(path, e))?;
    let pairs: HeaderPairs = serde_json::from_slice(&header).map_err(|e| {
        st_err(
            path,
            HEADER_TENSOR,
            format!("header is not a JSON object: {e}"),
        )
    })?;
    let data_start = 8 + header_len;
    let data_len = file_len - data_start;
    let mut local: Vec<TensorEntry> = Vec::with_capacity(pairs.0.len());
    let mut local_names: HashSet<String> = HashSet::new();
    for (name, value) in pairs.0 {
        if name == "__metadata__" {
            continue;
        }
        if !local_names.insert(name.clone()) {
            return Err(st_err(path, &name, "tensor listed twice in the header"));
        }
        local.push(parse_entry(path, name, value, data_start, data_len)?);
    }
    local.sort_by_key(|e| (e.range.start, e.range.end));
    for pair in local.windows(2) {
        let (prev, next) = (&pair[0], &pair[1]);
        if next.range.start < prev.range.end {
            return Err(st_err(
                path,
                &next.name,
                format!("overlapping ranges with {}", prev.name),
            ));
        }
    }
    for entry in local {
        if all.contains_key(&entry.name) {
            return Err(st_err(
                path,
                &entry.name,
                "tensor listed twice across shards",
            ));
        }
        all.insert(entry.name.clone(), entry);
    }
    Ok(())
}

/// Validates one header entry against the data section `[data_start, data_start + data_len)`.
fn parse_entry(
    path: &Path,
    name: String,
    value: serde_json::Value,
    data_start: u64,
    data_len: u64,
) -> Result<TensorEntry, ModelError> {
    let raw: RawTensor = serde_json::from_value(value)
        .map_err(|e| st_err(path, &name, format!("malformed entry: {e}")))?;
    let dtype: Dtype = serde_json::from_value(serde_json::Value::String(raw.dtype.clone()))
        .map_err(|_| st_err(path, &name, format!("unknown dtype {}", raw.dtype)))?;
    let (begin, end) = raw.data_offsets;
    if begin > end || end > data_len {
        return Err(st_err(
            path,
            &name,
            format!(
                "range outside file: data_offsets [{begin}, {end}] in a data section of \
                 {data_len} B"
            ),
        ));
    }
    let expected = raw
        .shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
        .and_then(|n| n.checked_mul(dtype.bitsize() as u64))
        .filter(|bits| bits % 8 == 0)
        .map(|bits| bits / 8);
    if expected != Some(end - begin) {
        let size = expected.map_or_else(|| "invalid".to_string(), |b| format!("{b} B"));
        return Err(st_err(
            path,
            &name,
            format!(
                "shape {:?} × {dtype} size {size} != byte range {} B",
                raw.shape,
                end - begin
            ),
        ));
    }
    Ok(TensorEntry {
        name,
        dtype,
        shape: raw.shape,
        file: path.to_path_buf(),
        range: data_start + begin..data_start + end,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::ModelError;
    use crate::testing::TempDir;

    /// Writes `<8-byte LE header length><header><data>` verbatim.
    fn write_raw(path: &Path, header: &str, data: &[u8]) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes).expect("write safetensors file");
    }

    /// Asserts `result` is a `Safetensors` error for `file` / `tensor` whose rule contains
    /// `rule`, and that the rendered message names all three.
    #[track_caller]
    fn expect_rule(
        result: Result<SafetensorsIndex, ModelError>,
        file: &Path,
        tensor: &str,
        rule: &str,
    ) {
        let err = result.expect_err("malformed checkpoint must be rejected");
        let text = err.to_string();
        assert!(text.contains(&file.display().to_string()), "{text}");
        assert!(text.contains(tensor), "{text}");
        assert!(text.contains(rule), "{text}");
        match err {
            ModelError::Safetensors {
                file: f,
                tensor: t,
                rule: r,
            } => {
                assert_eq!(f, file, "{text}");
                assert_eq!(t, tensor, "{text}");
                assert!(r.contains(rule), "rule {r:?} should contain {rule:?}");
            }
            other => panic!("expected a safetensors error, got {other:?}"),
        }
    }

    fn single(prefix: &str) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new(prefix);
        let file = dir.path().join(SINGLE_FILE);
        (dir, file)
    }

    #[test]
    fn rejects_malformed_headers() {
        // Out-of-file range: 8 bytes claimed, 4 present.
        let (dir, file) = single("st-range");
        write_raw(
            &file,
            r#"{"w":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#,
            &[0; 4],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            "w",
            "range outside file",
        );

        // Overlapping ranges.
        let (dir, file) = single("st-overlap");
        write_raw(
            &file,
            r#"{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[4],"data_offsets":[4,12]}}"#,
            &[0; 12],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            "b",
            "overlapping ranges with a",
        );

        // A [2, 2] BF16 tensor (8 bytes) spanning 6 bytes.
        let (dir, file) = single("st-size");
        write_raw(
            &file,
            r#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,6]}}"#,
            &[0; 6],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            "w",
            "shape [2, 2] × BF16 size 8 B != byte range 6 B",
        );

        // Unknown dtype.
        let (dir, file) = single("st-dtype");
        write_raw(
            &file,
            r#"{"w":{"dtype":"Q4","shape":[4],"data_offsets":[0,2]}}"#,
            &[0; 2],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            "w",
            "unknown dtype Q4",
        );

        // A length prefix claiming 101 MiB: rejected from the prefix alone (the file is 10 bytes,
        // so nothing of that size is ever allocated or read).
        let (dir, file) = single("st-header");
        let mut bytes = (101u64 << 20).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&file, bytes).expect("write");
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            HEADER_TENSOR,
            "header of 105906176 bytes exceeds 100 MiB",
        );

        // The same tensor twice in one header.
        let (dir, file) = single("st-dup");
        write_raw(
            &file,
            r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"w":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}"#,
            &[0; 8],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            "w",
            "tensor listed twice",
        );

        // Index naming a missing shard.
        let dir = TempDir::new("st-shard");
        let s1 = dir.path().join("model-00001-of-00002.safetensors");
        let s2 = dir.path().join("model-00002-of-00002.safetensors");
        std::fs::write(
            dir.path().join(INDEX_FILE),
            r#"{"metadata":{"total_size":8},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .expect("write index");
        write_raw(
            &s1,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &s2,
            "b",
            "shard listed in index is missing",
        );

        // A tensor stored in two shards.
        write_raw(
            &s2,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}"#,
            &[0; 8],
        );
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &s2,
            "a",
            "tensor listed twice",
        );

        // Listed in weight_map but absent from its shard.
        let c_header = r#"{"c":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        write_raw(&s2, c_header, &[0; 4]);
        std::fs::write(
            dir.path().join(INDEX_FILE),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}}"#,
        )
        .expect("write index");
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &s2,
            "b",
            "absent from the shard",
        );

        // A well-formed sharded checkpoint opens with absolute ranges.
        let s2_header = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"c":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        write_raw(&s2, s2_header, &[0; 8]);
        let index = SafetensorsIndex::open(dir.path()).expect("valid sharded checkpoint");
        assert_eq!(index.entries().count(), 3);
        assert_eq!(index.total_bytes(), 12);
        let c = index.get("c").expect("c");
        let data_start = 8 + s2_header.len() as u64;
        assert_eq!(c.file, s2);
        assert_eq!(c.dtype, Dtype::F32);
        assert_eq!(c.shape, vec![1]);
        assert_eq!(c.range, data_start + 4..data_start + 8);
        assert_eq!(index.get("a").expect("a").file, s1);
        assert!(index.get("missing").is_none());
    }

    #[test]
    fn single_file_and_directory_errors() {
        // A valid single file.
        let (dir, file) = single("st-single");
        let header = r#"{"x":{"dtype":"BF16","shape":[1,3],"data_offsets":[0,6]}}"#;
        write_raw(&file, header, &[1, 2, 3, 4, 5, 6]);
        let index = SafetensorsIndex::open(dir.path()).expect("valid single file");
        let x = index.get("x").expect("x");
        assert_eq!(x.shape, vec![1, 3]);
        assert_eq!(
            x.range,
            8 + header.len() as u64..8 + header.len() as u64 + 6
        );
        assert_eq!(index.total_bytes(), 6);

        // A missing directory and an empty directory are Io errors naming the directory.
        let empty = TempDir::new("st-empty");
        let missing = empty.path().join("nope");
        match SafetensorsIndex::open(&missing).expect_err("missing dir") {
            ModelError::Io { path, .. } => assert_eq!(path, missing),
            other => panic!("expected Io, got {other:?}"),
        }
        match SafetensorsIndex::open(empty.path()).expect_err("empty dir") {
            ModelError::Io { path, .. } => assert_eq!(path, empty.path()),
            other => panic!("expected Io, got {other:?}"),
        }

        // A pickle-only directory is refused without opening the file (mode 0o000 would make
        // any open fail with a permission error instead).
        let pickle = TempDir::new("st-pickle");
        let bin = pickle.path().join("pytorch_model.bin");
        std::fs::write(&bin, b"not really a pickle").expect("write");
        match SafetensorsIndex::open(pickle.path()).expect_err("pickle-only dir") {
            ModelError::Pickle { path } => assert_eq!(path, bin),
            other => panic!("expected Pickle, got {other:?}"),
        }

        // A symlink to a directory is not followed as a checkpoint file.
        let link = TempDir::new("st-link");
        std::fs::create_dir(link.path().join("sub")).expect("mkdir");
        std::os::unix::fs::symlink(link.path().join("sub"), link.path().join(SINGLE_FILE))
            .expect("symlink");
        let err = SafetensorsIndex::open(link.path()).expect_err("symlink to a directory");
        assert!(err.to_string().contains("not a regular file"), "{err}");

        // A header that is not JSON.
        let (dir, file) = single("st-json");
        write_raw(&file, "not json", &[]);
        expect_rule(
            SafetensorsIndex::open(dir.path()),
            &file,
            HEADER_TENSOR,
            "header is not a JSON object",
        );
    }
}
