//! `--set <dotted.key>=<yaml value>` overrides and the YAML value-tree helpers used to
//! apply them and to name offending keys.

use std::str::FromStr;

use serde_norway::{Mapping, Value};

use super::ConfigError;

/// One `--set <dotted.key>=<yaml scalar>` override.
#[derive(Clone, Debug, PartialEq)]
pub struct Override {
    pub key: String,
    pub value: Value,
}

impl FromStr for Override {
    type Err = ConfigError;

    fn from_str(arg: &str) -> Result<Self, Self::Err> {
        let bad = |reason: &str| ConfigError::BadOverride {
            arg: arg.to_string(),
            reason: reason.to_string(),
        };
        let (key, raw) = arg
            .split_once('=')
            .ok_or_else(|| bad("expected <dotted.key>=<yaml value>"))?;
        if key.is_empty() || key.split('.').any(str::is_empty) {
            return Err(bad("key must be a dotted path such as kv.cpu.enabled"));
        }
        let value: Value = serde_norway::from_str(raw)
            .map_err(|e| bad(&format!("value is not valid YAML: {e}")))?;
        Ok(Override {
            key: key.to_string(),
            value,
        })
    }
}

/// Set `dotted` in `root` to `value`, creating (or replacing non-mapping) intermediate levels.
pub(crate) fn set_path(root: &mut Value, dotted: &str, value: Value) {
    let mut node = root;
    let mut segments = dotted.split('.').peekable();
    while let Some(segment) = segments.next() {
        if !node.is_mapping() {
            *node = Value::Mapping(Mapping::new());
        }
        let Value::Mapping(map) = node else {
            unreachable!("node was just made a mapping")
        };
        let key = Value::String(segment.to_string());
        if segments.peek().is_none() {
            map.insert(key, value);
            return;
        }
        node = map
            .entry(key)
            .or_insert_with(|| Value::Mapping(Mapping::new()));
    }
}

/// First key in `user` (document order) that does not exist in `known`, as a dotted path.
pub(crate) fn first_unknown_key(user: &Value, known: &Value, prefix: &str) -> Option<String> {
    let (Value::Mapping(user_map), Value::Mapping(known_map)) = (user, known) else {
        return None;
    };
    for (k, v) in user_map {
        let name = match k {
            Value::String(s) => s.clone(),
            other => serde_norway::to_string(other)
                .map(|s| s.trim_end().to_string())
                .unwrap_or_else(|_| "<non-string key>".to_string()),
        };
        let path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}.{name}")
        };
        match known_map.get(Value::String(name)) {
            None => return Some(path),
            Some(known_child) => {
                if let Some(found) = first_unknown_key(v, known_child, &path) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Every leaf of `user` as `(dotted path, value)`, in document order. A leaf is any value
/// that is not a mapping, or a mapping whose counterpart in `known` is not a mapping.
pub(crate) fn leaves(user: &Value, known: &Value, prefix: &str, out: &mut Vec<(String, Value)>) {
    match (user, known) {
        (Value::Mapping(user_map), Value::Mapping(known_map)) => {
            for (k, v) in user_map {
                let Value::String(name) = k else { continue };
                let path = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                match known_map.get(Value::String(name.clone())) {
                    Some(known_child) => leaves(v, known_child, &path, out),
                    None => out.push((path, v.clone())),
                }
            }
        }
        _ => out.push((prefix.to_string(), user.clone())),
    }
}
