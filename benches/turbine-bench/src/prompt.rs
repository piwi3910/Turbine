//! Seeded synthetic prompts: `--prompt-words` words from a built-in 1,000-word list, drawn by a
//! SplitMix64 PRNG seeded with `seed + request index`, so the same seed gives identical bytes.

/// Size of the built-in word list.
pub const WORD_COUNT: usize = 1000;

const ONSETS: [&str; 10] = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "t"];
const NUCLEI: [&str; 10] = ["a", "e", "i", "o", "u", "ai", "ea", "io", "ou", "ue"];
const CODAS: [&str; 10] = ["", "n", "r", "s", "t", "l", "m", "x", "nd", "st"];

/// Word `i` (taken modulo 1,000) of the built-in list: onset × nucleus × coda, all distinct.
pub fn word(i: usize) -> String {
    let i = i % WORD_COUNT;
    format!(
        "{}{}{}",
        ONSETS[i / 100],
        NUCLEI[(i / 10) % 10],
        CODAS[i % 10]
    )
}

/// SplitMix64 (Steele, Lea, Flood 2014): tiny, deterministic, no dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// The prompt for request `index`: `words` words joined by single spaces.
pub fn prompt(seed: u64, index: u64, words: u32) -> String {
    let mut rng = SplitMix64(seed.wrapping_add(index));
    let mut out = String::new();
    for n in 0..words {
        if n > 0 {
            out.push(' ');
        }
        // WORD_COUNT fits in u64 and the remainder is < WORD_COUNT, so both casts are lossless.
        out.push_str(&word((rng.next() % WORD_COUNT as u64) as usize));
    }
    out
}

/// Prompts for requests `0..count`.
pub fn prompts(seed: u64, count: u32, words: u32) -> Vec<String> {
    (0..u64::from(count))
        .map(|i| prompt(seed, i, words))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_are_deterministic() {
        let a = prompts(7, 10, 256);
        let b = prompts(7, 10, 256);
        let c = prompts(8, 10, 256);
        assert_eq!(a, b, "same seed must give byte-identical prompts");
        assert_ne!(a, c, "a different seed must give different prompts");
        assert_eq!(a[0].split(' ').count(), 256);
        let distinct: std::collections::HashSet<String> = (0..WORD_COUNT).map(word).collect();
        assert_eq!(distinct.len(), WORD_COUNT);
    }
}
