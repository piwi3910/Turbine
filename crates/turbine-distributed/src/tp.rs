//! Tensor-parallel sharding rules (P5 S-6, contract §15.2): which contiguous slice of heads,
//! KV heads, intermediate columns/rows and vocabulary rows a rank owns. Pure functions; the
//! planner has already rejected tensor-parallel sizes these splits cannot serve, so an uneven
//! split here is a caller bug and panics.
//!
//! How the ranges compose (verified against an unsharded reference in the tests):
//! q/k/v and gate/up (and every MoE expert) are column-parallel, o_proj and down_proj are
//! row-parallel followed by an all-reduce, the router is replicated, a QK-norm over the full
//! projection uses all-reduced partial sums of squares, the embedding is vocab-parallel with
//! an all-reduce and the LM head vocab-parallel with an all-gather whose padded rows are −inf.

use std::ops::Range;

/// This rank's position in its tensor-parallel group.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShardSpec {
    pub rank: u32,
    pub world: u32,
}

impl ShardSpec {
    fn check(self) {
        assert!(
            self.world > 0 && self.rank < self.world,
            "rank {} outside a group of {}",
            self.rank,
            self.world
        );
    }
}

/// `dim / world` elements starting at `rank × dim / world`.
fn even_split(dim: u32, s: ShardSpec, what: &str) -> Range<u32> {
    s.check();
    assert!(
        dim.is_multiple_of(s.world),
        "{what} {dim} is not divisible by tensor-parallel size {}",
        s.world
    );
    let per = dim / s.world;
    s.rank * per..(s.rank + 1) * per
}

/// Attention heads of this rank.
pub fn head_range(num_heads: u32, s: ShardSpec) -> Range<u32> {
    even_split(num_heads, s, "attention heads")
}

/// KV heads of this rank: an even split while `world ≤ num_kv_heads`; beyond that each KV head
/// is replicated on `world / num_kv_heads` consecutive ranks (the ranks whose query heads use it).
pub fn kv_head_range(num_kv_heads: u32, s: ShardSpec) -> Range<u32> {
    s.check();
    if s.world <= num_kv_heads {
        return even_split(num_kv_heads, s, "KV heads");
    }
    assert!(
        num_kv_heads > 0 && s.world.is_multiple_of(num_kv_heads),
        "tensor-parallel size {} is not divisible by {num_kv_heads} KV heads",
        s.world
    );
    let head = s.rank / (s.world / num_kv_heads);
    head..head + 1
}

/// Output columns of a column-parallel projection (q/k/v, gate/up, every expert along the
/// intermediate dimension).
pub fn column_range(dim: u32, s: ShardSpec) -> Range<u32> {
    even_split(dim, s, "column dimension")
}

/// Input rows of a row-parallel projection (o_proj, down_proj), followed by an all-reduce.
pub fn row_range(dim: u32, s: ShardSpec) -> Range<u32> {
    even_split(dim, s, "row dimension")
}

/// `(offset, rows, padded_rows)`: this rank holds vocabulary rows `offset .. offset + rows`
/// of shards `padded_rows = ceil(vocab / world)` long; the last shard's missing rows are
/// padding, masked to −inf before the LM-head all-gather.
pub fn vocab_shard(vocab: u32, s: ShardSpec) -> (u32, u32, u32) {
    s.check();
    let padded = vocab.div_ceil(s.world);
    let offset = s.rank * padded;
    let rows = vocab.saturating_sub(offset).min(padded);
    (offset, rows, padded)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;
    use std::time::Duration;

    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceBuffer, DeviceId, DeviceMemory, StreamRef};

    use super::*;
    use crate::collective::{Collective, HostCollective, ReduceOp};

    /// splitmix64 values uniform in [-scale, scale).
    fn values(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0xA24B_AED4_963E_E407);
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    struct Cfg {
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        /// Dense MLP intermediate, or per-expert intermediate for MoE.
        inter: usize,
        vocab: usize,
        experts: usize,
        top_k: usize,
        qk_norm: bool,
        tied: bool,
    }

    const EPS: f32 = 1e-6;

    struct Expert {
        gate: Vec<f32>, // hidden × inter
        up: Vec<f32>,   // hidden × inter
        down: Vec<f32>, // inter × hidden
    }

    struct Weights {
        embed: Vec<f32>,           // vocab × hidden
        lm_head: Option<Vec<f32>>, // vocab × hidden (None = tied)
        norm_in: Vec<f32>,
        wq: Vec<f32>, // hidden × heads·dim
        wk: Vec<f32>, // hidden × kv·dim
        wv: Vec<f32>,
        q_norm: Vec<f32>, // heads·dim
        k_norm: Vec<f32>, // kv·dim
        wo: Vec<f32>,     // heads·dim × hidden
        norm_post: Vec<f32>,
        router: Vec<f32>,     // hidden × experts (MoE)
        experts: Vec<Expert>, // one for dense
        norm_final: Vec<f32>,
    }

    impl Weights {
        fn new(c: Cfg, seed: u64) -> Weights {
            let (h, qd, kd) = (c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim);
            let mut k = seed;
            let mut next = |n: usize, scale: f32| {
                k += 1;
                values(k, n, scale)
            };
            let norm = |v: Vec<f32>| v.into_iter().map(|x| 1.0 + x).collect::<Vec<f32>>();
            let n_experts = c.experts.max(1);
            Weights {
                embed: next(c.vocab * h, 0.5),
                lm_head: (!c.tied).then(|| next(c.vocab * h, 0.3)),
                norm_in: norm(next(h, 0.2)),
                wq: next(h * qd, 0.3),
                wk: next(h * kd, 0.3),
                wv: next(h * kd, 0.3),
                q_norm: norm(next(qd, 0.2)),
                k_norm: norm(next(kd, 0.2)),
                wo: next(qd * h, 0.3),
                norm_post: norm(next(h, 0.2)),
                router: next(h * c.experts, 0.8),
                experts: (0..n_experts)
                    .map(|_| Expert {
                        gate: next(h * c.inter, 0.3),
                        up: next(h * c.inter, 0.3),
                        down: next(c.inter * h, 0.3),
                    })
                    .collect(),
                norm_final: norm(next(h, 0.2)),
            }
        }
    }

    /// `x` (t × k) times the columns `cols` of `w` (k × n, row-major).
    fn matmul_cols(
        x: &[f32],
        t: usize,
        k: usize,
        w: &[f32],
        n: usize,
        cols: Range<usize>,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; t * cols.len()];
        for i in 0..t {
            for (o, j) in cols.clone().enumerate() {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += x[i * k + p] * w[p * n + j];
                }
                out[i * cols.len() + o] = acc;
            }
        }
        out
    }

    /// `x` (t × rows.len()) times the rows `rows` of `w` (k × n).
    fn matmul_rows(x: &[f32], t: usize, w: &[f32], n: usize, rows: Range<usize>) -> Vec<f32> {
        let kl = rows.len();
        let mut out = vec![0.0f32; t * n];
        for i in 0..t {
            for j in 0..n {
                let mut acc = 0.0f32;
                for (p, r) in rows.clone().enumerate() {
                    acc += x[i * kl + p] * w[r * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn rmsnorm(x: &[f32], t: usize, n: usize, w: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; t * n];
        for i in 0..t {
            let row = &x[i * n..(i + 1) * n];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / n as f32 + EPS).sqrt();
            for j in 0..n {
                out[i * n + j] = row[j] * inv * w[j];
            }
        }
        out
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Causal attention for the heads in `heads` whose K/V columns hold the KV heads `kv`.
    fn attention(
        c: Cfg,
        t: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: Range<usize>,
        kv: Range<usize>,
    ) -> Vec<f32> {
        let d = c.head_dim;
        let (nq, nk) = (heads.len(), kv.len());
        let group = c.heads / c.kv_heads;
        let mut out = vec![0.0f32; t * nq * d];
        for (hl, hg) in heads.clone().enumerate() {
            let kl = hg / group - kv.start;
            for i in 0..t {
                let qi = &q[i * nq * d + hl * d..i * nq * d + (hl + 1) * d];
                let scores: Vec<f32> = (0..=i)
                    .map(|s| {
                        let ks = &k[s * nk * d + kl * d..s * nk * d + (kl + 1) * d];
                        qi.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for s in 0..=i {
                    for j in 0..d {
                        out[i * nq * d + hl * d + j] += e[s] / z * v[s * nk * d + kl * d + j];
                    }
                }
            }
        }
        out
    }

    /// Top-k experts and their softmax probabilities (not renormalised), per token.
    fn route(c: Cfg, t: usize, h: &[f32], router: &[f32]) -> Vec<Vec<(usize, f32)>> {
        let logits = matmul_cols(h, t, c.hidden, router, c.experts, 0..c.experts);
        (0..t)
            .map(|i| {
                let row = &logits[i * c.experts..(i + 1) * c.experts];
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                let mut idx: Vec<usize> = (0..c.experts).collect();
                idx.sort_by(|a, b| e[*b].total_cmp(&e[*a]));
                idx[..c.top_k].iter().map(|&j| (j, e[j] / z)).collect()
            })
            .collect()
    }

    /// The whole model on one device, without any range function (the oracle).
    fn reference(c: Cfg, w: &Weights, tokens: &[usize]) -> Vec<f32> {
        let (t, h) = (tokens.len(), c.hidden);
        let (qd, kd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&tok| w.embed[tok * h..(tok + 1) * h].to_vec())
            .collect();
        let hn = rmsnorm(&x, t, h, &w.norm_in);
        let mut q = matmul_cols(&hn, t, h, &w.wq, qd, 0..qd);
        let mut k = matmul_cols(&hn, t, h, &w.wk, kd, 0..kd);
        let v = matmul_cols(&hn, t, h, &w.wv, kd, 0..kd);
        if c.qk_norm {
            q = rmsnorm(&q, t, qd, &w.q_norm);
            k = rmsnorm(&k, t, kd, &w.k_norm);
        }
        let a = attention(c, t, &q, &k, &v, 0..c.heads, 0..c.kv_heads);
        let o = matmul_rows(&a, t, &w.wo, h, 0..qd);
        x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);
        let h2 = rmsnorm(&x, t, h, &w.norm_post);
        let mlp = |e: &Expert| {
            let g = matmul_cols(&h2, t, h, &e.gate, c.inter, 0..c.inter);
            let u = matmul_cols(&h2, t, h, &e.up, c.inter, 0..c.inter);
            let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            matmul_rows(&act, t, &e.down, h, 0..c.inter)
        };
        let m = if c.experts == 0 {
            mlp(&w.experts[0])
        } else {
            let routes = route(c, t, &h2, &w.router);
            let mut m = vec![0.0f32; t * h];
            for (e, expert) in w.experts.iter().enumerate() {
                let out = mlp(expert);
                for (i, r) in routes.iter().enumerate() {
                    if let Some(&(_, p)) = r.iter().find(|(j, _)| *j == e) {
                        for j in 0..h {
                            m[i * h + j] += p * out[i * h + j];
                        }
                    }
                }
            }
            m
        };
        x.iter_mut().zip(&m).for_each(|(x, m)| *x += m);
        let f = rmsnorm(&x, t, h, &w.norm_final);
        let head = w.lm_head.as_ref().unwrap_or(&w.embed);
        let mut logits = vec![0.0f32; t * c.vocab];
        for i in 0..t {
            for r in 0..c.vocab {
                logits[i * c.vocab + r] = (0..h).map(|j| f[i * h + j] * head[r * h + j]).sum();
            }
        }
        logits
    }

    /// A rank's view of its collective, over its own host "device".
    struct Comm<'a> {
        c: &'a HostCollective,
        mem: Arc<dyn DeviceMemory>,
        stream: StreamRef,
    }

    impl Comm<'_> {
        fn all_reduce(&self, v: &mut [f32]) {
            let buf = DeviceBuffer::alloc(&self.mem, v.len() * 4).expect("alloc");
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            buf.whole().write_bytes(&bytes).expect("write");
            self.c
                .all_reduce(
                    &mut buf.whole(),
                    turbine_tensor::DType::F32,
                    ReduceOp::Sum,
                    &self.stream,
                )
                .expect("all_reduce");
            let out = buf.whole().read_bytes().expect("read");
            for (x, c) in v.iter_mut().zip(out.chunks_exact(4)) {
                *x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }

        fn all_gather(&self, v: &[f32]) -> Vec<f32> {
            let world = self.c.world_size();
            let send = DeviceBuffer::alloc(&self.mem, v.len() * 4).expect("alloc");
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            send.whole().write_bytes(&bytes).expect("write");
            let recv = DeviceBuffer::alloc(&self.mem, v.len() * 4 * world).expect("alloc");
            self.c
                .all_gather(&send.whole(), &mut recv.whole(), &self.stream)
                .expect("all_gather");
            recv.whole()
                .read_bytes()
                .expect("read")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    }

    /// The same model on one rank of `comm`, using only the range functions to pick this
    /// rank's slices. Every rank returns the full logits (t × vocab).
    fn sharded(c: Cfg, w: &Weights, tokens: &[usize], comm: &Comm<'_>) -> Vec<f32> {
        let s = ShardSpec {
            rank: comm.c.rank() as u32,
            world: comm.c.world_size() as u32,
        };
        let u = |r: Range<u32>| r.start as usize..r.end as usize;
        let (t, h, d) = (tokens.len(), c.hidden, c.head_dim);
        let (qd, kd) = (c.heads * d, c.kv_heads * d);

        // Vocab-parallel embedding: this rank's rows only, then all-reduce.
        let (off, rows, padded) = vocab_shard(c.vocab as u32, s);
        let vocab_rows = off as usize..(off + rows) as usize;
        let mut x = vec![0.0f32; t * h];
        for (i, &tok) in tokens.iter().enumerate() {
            if vocab_rows.contains(&tok) {
                x[i * h..(i + 1) * h].copy_from_slice(&w.embed[tok * h..(tok + 1) * h]);
            }
        }
        comm.all_reduce(&mut x);

        // Attention: heads split, KV heads split or replicated.
        let heads = u(head_range(c.heads as u32, s));
        let kv = u(kv_head_range(c.kv_heads as u32, s));
        let qcols = heads.start * d..heads.end * d;
        let kcols = kv.start * d..kv.end * d;
        let hn = rmsnorm(&x, t, h, &w.norm_in);
        let mut q = matmul_cols(&hn, t, h, &w.wq, qd, qcols.clone());
        let mut k = matmul_cols(&hn, t, h, &w.wk, kd, kcols.clone());
        let v = matmul_cols(&hn, t, h, &w.wv, kd, kcols.clone());
        if c.qk_norm {
            // Full-projection norm from all-reduced partial sums of squares.
            for (proj, cols, full, weight) in [
                (&mut q, &qcols, qd, &w.q_norm),
                (&mut k, &kcols, kd, &w.k_norm),
            ] {
                let n = cols.len();
                let mut ss: Vec<f32> = (0..t)
                    .map(|i| proj[i * n..(i + 1) * n].iter().map(|v| v * v).sum())
                    .collect();
                comm.all_reduce(&mut ss);
                for i in 0..t {
                    let inv = 1.0 / (ss[i] / full as f32 + EPS).sqrt();
                    for (j, col) in cols.clone().enumerate() {
                        proj[i * n + j] *= inv * weight[col];
                    }
                }
            }
        }
        let a = attention(c, t, &q, &k, &v, heads.clone(), kv);
        // Row-parallel o_proj + all-reduce.
        let orows = u(row_range(qd as u32, s));
        assert_eq!(orows, qcols, "o_proj rows follow the head split");
        let mut o = matmul_rows(&a, t, &w.wo, h, orows);
        comm.all_reduce(&mut o);
        x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);

        // MLP / experts: gate/up column-parallel, down row-parallel, one all-reduce.
        let h2 = rmsnorm(&x, t, h, &w.norm_post);
        let icols = u(column_range(c.inter as u32, s));
        let irows = u(row_range(c.inter as u32, s));
        let mlp = |e: &Expert| {
            let g = matmul_cols(&h2, t, h, &e.gate, c.inter, icols.clone());
            let up = matmul_cols(&h2, t, h, &e.up, c.inter, icols.clone());
            let act: Vec<f32> = g.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            matmul_rows(&act, t, &e.down, h, irows.clone())
        };
        let mut m = if c.experts == 0 {
            mlp(&w.experts[0])
        } else {
            // The router is replicated, so every rank picks the same experts.
            let routes = route(c, t, &h2, &w.router);
            let mut m = vec![0.0f32; t * h];
            for (e, expert) in w.experts.iter().enumerate() {
                let out = mlp(expert);
                for (i, r) in routes.iter().enumerate() {
                    if let Some(&(_, p)) = r.iter().find(|(j, _)| *j == e) {
                        for j in 0..h {
                            m[i * h + j] += p * out[i * h + j];
                        }
                    }
                }
            }
            m
        };
        comm.all_reduce(&mut m);
        x.iter_mut().zip(&m).for_each(|(x, m)| *x += m);

        // Vocab-parallel LM head (tied: the embedding shard), padding −inf, all-gather.
        let f = rmsnorm(&x, t, h, &w.norm_final);
        let head = w.lm_head.as_ref().unwrap_or(&w.embed);
        let padded = padded as usize;
        let mut local = vec![f32::NEG_INFINITY; t * padded];
        for i in 0..t {
            for (p, r) in vocab_rows.clone().enumerate() {
                local[i * padded + p] = (0..h).map(|j| f[i * h + j] * head[r * h + j]).sum();
            }
        }
        // Gathered layout: rank-major, each rank t × padded.
        let gathered = comm.all_gather(&local);
        let world = s.world as usize;
        let mut logits = vec![0.0f32; t * c.vocab];
        for i in 0..t {
            for vid in 0..c.vocab {
                let (rank, p) = (vid / padded, vid % padded);
                logits[i * c.vocab + vid] = gathered[rank * t * padded + i * padded + p];
            }
        }
        // Every padded slot really is −inf.
        for rank in 0..world {
            for i in 0..t {
                for p in 0..padded {
                    if rank * padded + p >= c.vocab {
                        assert_eq!(
                            gathered[rank * t * padded + i * padded + p],
                            f32::NEG_INFINITY
                        );
                    }
                }
            }
        }
        logits
    }

    fn run_tp(c: Cfg, w: &Weights, tokens: &[usize], world: usize) -> Vec<Vec<f32>> {
        let group = HostCollective::group(world, Duration::from_secs(5));
        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .iter()
                .map(|col| {
                    scope.spawn(move || {
                        let mem: Arc<dyn DeviceMemory> =
                            HostMemory::new(DeviceId(col.rank() as u32), 1 << 26);
                        let stream = mem.compute_stream();
                        let comm = Comm {
                            c: col,
                            mem,
                            stream,
                        };
                        sharded(c, w, tokens, &comm)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        })
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn sharded_layers_match_unsharded() {
        let dense = Cfg {
            hidden: 32,
            heads: 8,
            kv_heads: 2,
            head_dim: 4,
            inter: 48,
            vocab: 1003,
            experts: 0,
            top_k: 0,
            qk_norm: false,
            tied: true,
        };
        let moe = Cfg {
            hidden: 32,
            heads: 4,
            kv_heads: 4,
            head_dim: 8,
            inter: 16,
            vocab: 1003,
            experts: 8,
            top_k: 2,
            qk_norm: true,
            tied: false,
        };
        for (name, c, seed) in [("dense", dense, 11u64), ("moe", moe, 23u64)] {
            let w = Weights::new(c, seed);
            let tokens = [5usize, 1002, 500, 17];
            let oracle = reference(c, &w, &tokens);
            assert!(oracle.iter().all(|x| x.is_finite()), "{name}");
            for world in [1usize, 2, 4] {
                let per_rank = run_tp(c, &w, &tokens, world);
                for (rank, logits) in per_rank.iter().enumerate() {
                    let diff = max_abs_diff(logits, &oracle);
                    assert!(
                        diff <= 1e-5,
                        "{name} tp {world} rank {rank}: max |Δlogit| {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn range_rules() {
        let s = |rank, world| ShardSpec { rank, world };
        assert_eq!(head_range(24, s(1, 2)), 12..24);
        assert_eq!(head_range(16, s(3, 8)), 6..8);
        // KV heads split while tp ≤ kv heads, replicated one per rank group beyond.
        assert_eq!(kv_head_range(8, s(1, 2)), 4..8);
        assert_eq!(kv_head_range(2, s(0, 4)), 0..1);
        assert_eq!(kv_head_range(2, s(1, 4)), 0..1);
        assert_eq!(kv_head_range(2, s(2, 4)), 1..2);
        assert_eq!(kv_head_range(2, s(3, 4)), 1..2);
        assert_eq!(column_range(8192, s(1, 4)), 2048..4096);
        assert_eq!(row_range(3072, s(0, 2)), 0..1536);
        // 1003 over 4: shards of 251, the last holds 250 rows and one padded slot.
        assert_eq!(vocab_shard(1003, s(0, 4)), (0, 251, 251));
        assert_eq!(vocab_shard(1003, s(3, 4)), (753, 250, 251));
        assert_eq!(vocab_shard(128_256, s(1, 2)), (64_128, 64_128, 64_128));
        assert_eq!(vocab_shard(5, s(3, 4)), (6, 0, 2));
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn uneven_split_is_a_bug() {
        let _ = column_range(10, ShardSpec { rank: 0, world: 4 });
    }
}
