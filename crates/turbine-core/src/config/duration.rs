//! `HumanDuration`: `"<integer><ms|s|m|h>"`, no space, exact case — one parser for every
//! duration key of every phase (CONFLICT C-14).

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A duration written in YAML as `<integer><unit>` with unit `ms`, `s`, `m` or `h`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct HumanDuration(pub Duration);

/// Units, longest name first where names share a prefix (`ms` before `m`).
const UNITS: [(&str, u64); 4] = [("ms", 1), ("s", 1_000), ("m", 60_000), ("h", 3_600_000)];

impl HumanDuration {
    pub const fn from_secs(s: u64) -> Self {
        HumanDuration(Duration::from_secs(s))
    }
    pub const fn from_millis(ms: u64) -> Self {
        HumanDuration(Duration::from_millis(ms))
    }
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl FromStr for HumanDuration {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let expected = "expected <integer><unit> with unit ms, s, m or h (no space, lower case)";
        let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        let (digits, unit) = s.split_at(split);
        if digits.is_empty() || unit.is_empty() {
            return Err(format!("invalid duration {s:?}: {expected}"));
        }
        let millis_per = UNITS
            .iter()
            .find(|(name, _)| *name == unit)
            .map(|(_, m)| *m)
            .ok_or_else(|| format!("invalid duration {s:?}: unknown unit {unit:?}; {expected}"))?;
        let n: u64 = digits
            .parse()
            .map_err(|_| format!("invalid duration {s:?}: number overflows u64"))?;
        n.checked_mul(millis_per)
            .map(|ms| HumanDuration(Duration::from_millis(ms)))
            .ok_or_else(|| format!("invalid duration {s:?}: value overflows"))
    }
}

impl fmt::Display for HumanDuration {
    /// The largest unit that represents the value exactly (`"10m"`, `"1500ms"`); `"0s"` for zero.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = self.0.as_millis();
        if ms == 0 {
            return f.write_str("0s");
        }
        for (name, per) in UNITS.iter().rev() {
            let per = u128::from(*per);
            if ms.is_multiple_of(per) {
                return write!(f, "{}{name}", ms / per);
            }
        }
        write!(f, "{ms}ms")
    }
}

impl Serialize for HumanDuration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for HumanDuration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DurationVisitor;

        impl Visitor<'_> for DurationVisitor {
            type Value = HumanDuration;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a duration string like \"30s\", \"250ms\", \"10m\" or \"1h\"")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<HumanDuration, E> {
                Err(E::custom(format!(
                    "invalid duration {v}: a unit is required (ms, s, m or h), e.g. \"{v}s\""
                )))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<HumanDuration, E> {
                Err(E::custom(format!(
                    "invalid duration {v}: must be a non-negative integer with a unit (ms, s, m or h)"
                )))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<HumanDuration, E> {
                Err(E::custom(format!(
                    "invalid duration {v}: must be an integer with a unit (ms, s, m or h)"
                )))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<HumanDuration, E> {
                v.parse().map_err(E::custom)
            }
        }

        deserializer.deserialize_any(DurationVisitor)
    }
}
