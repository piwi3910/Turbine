//! `ByteSize`: a non-negative integer (bytes) or `"<integer><unit>"`, no space, case-sensitive.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A size in bytes, written in YAML as an integer or `<integer><B|KB|MB|GB|TB|KiB|MiB|GiB|TiB>`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ByteSize(pub u64);

const UNITS: [(&str, u64); 9] = [
    ("B", 1),
    ("KB", 1_000),
    ("MB", 1_000_000),
    ("GB", 1_000_000_000),
    ("TB", 1_000_000_000_000),
    ("KiB", 1 << 10),
    ("MiB", 1 << 20),
    ("GiB", 1 << 30),
    ("TiB", 1 << 40),
];

impl ByteSize {
    pub const fn kib(n: u64) -> Self {
        ByteSize(n << 10)
    }
    pub const fn mib(n: u64) -> Self {
        ByteSize(n << 20)
    }
    pub const fn gib(n: u64) -> Self {
        ByteSize(n << 30)
    }
}

impl FromStr for ByteSize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        let (digits, unit) = s.split_at(split);
        if digits.is_empty() {
            return Err(format!(
                "invalid byte size {s:?}: expected a non-negative integer optionally followed by B, KB, MB, GB, TB, KiB, MiB, GiB or TiB"
            ));
        }
        let n: u64 = digits
            .parse()
            .map_err(|_| format!("invalid byte size {s:?}: number overflows u64"))?;
        let multiplier = if unit.is_empty() {
            1
        } else {
            UNITS
                .iter()
                .find(|(name, _)| *name == unit)
                .map(|(_, m)| *m)
                .ok_or_else(|| {
                    format!(
                        "invalid byte size {s:?}: unknown unit {unit:?} (units: B, KB, MB, GB, TB, KiB, MiB, GiB, TiB; no space, case-sensitive)"
                    )
                })?
        };
        n.checked_mul(multiplier)
            .map(ByteSize)
            .ok_or_else(|| format!("invalid byte size {s:?}: value overflows u64"))
    }
}

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, m) in [
            ("TiB", 1u64 << 40),
            ("GiB", 1 << 30),
            ("MiB", 1 << 20),
            ("KiB", 1 << 10),
        ] {
            if self.0 >= m && self.0.is_multiple_of(m) {
                return write!(f, "{}{name}", self.0 / m);
            }
        }
        write!(f, "{}B", self.0)
    }
}

impl Serialize for ByteSize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ByteSizeVisitor;

        impl Visitor<'_> for ByteSizeVisitor {
            type Value = ByteSize;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a non-negative integer or a string like \"64GiB\"")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<ByteSize, E> {
                Ok(ByteSize(v))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<ByteSize, E> {
                u64::try_from(v)
                    .map(ByteSize)
                    .map_err(|_| E::custom(format!("invalid byte size {v}: must be non-negative")))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<ByteSize, E> {
                Err(E::custom(format!(
                    "invalid byte size {v}: must be an integer"
                )))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<ByteSize, E> {
                v.parse().map_err(E::custom)
            }
        }

        deserializer.deserialize_any(ByteSizeVisitor)
    }
}
