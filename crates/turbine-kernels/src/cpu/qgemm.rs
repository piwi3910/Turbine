//! `qgemm` and `quantize_act` on the cpu-reference provider (ABI v2.9 semantics, Phase 6a S-5):
//! the weight is dequantized to F32 by [`quant::dequantize`], FP8 activations are decoded with
//! their scales, the product accumulates in F32 and is rounded to `c`'s dtype. The reference
//! every GPU implementation is compared against.
use super::quant;
use super::{
    CpuReference, expect_rank, invalid, is_float, load, load_bytes, math, store, store_bytes,
};
use crate::KernelError;
use crate::ops::{
    QGemmConfig, QGemmContext, QGemmKernel, QuantizeActConfig, QuantizeActContext,
    QuantizeActKernel,
};
use crate::quant::{ActQuantDesc, QuantSchemeDesc};
use turbine_core::types::DType;

fn weight_bytes_per_row(scheme: QuantSchemeDesc, k: usize) -> usize {
    scheme.data_bytes(1, k)
}

/// The activation matrix as F32: BF16 / F32 as stored, FP8 decoded with its scales.
fn activations(ctx: &QGemmContext<'_>, m: usize, k: usize) -> Result<Vec<f32>, KernelError> {
    let mode = ctx.cfg.act_quant;
    if !mode.is_fp8() {
        return load(&ctx.a);
    }
    let bytes = load_bytes(&ctx.a)?;
    let scales_view = ctx
        .a_scales
        .as_ref()
        .ok_or_else(|| invalid("qgemm: FP8 activations need a_scales".into()))?;
    let scales = load(scales_view)?;
    if scales.len() != mode.scale_count(m, k) {
        return Err(invalid(format!(
            "qgemm: {} activation scales for {m} × {k} in mode {}",
            scales.len(),
            mode.as_str()
        )));
    }
    let mut a = vec![0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            let s = match mode {
                ActQuantDesc::Fp8Tensor => scales[0],
                ActQuantDesc::Fp8Token => scales[r],
                ActQuantDesc::Fp8Group { group } => {
                    scales[r * k.div_ceil(group as usize) + c / group as usize]
                }
                _ => unreachable!("is_fp8"),
            };
            a[r * k + c] = quant::fp8_e4m3_value(bytes[r * k + c]) * s;
        }
    }
    Ok(a)
}

impl QGemmKernel for CpuReference {
    fn supports(&self, cfg: &QGemmConfig) -> bool {
        let a_ok = if cfg.act_quant.is_fp8() {
            cfg.a_dtype == DType::F8E4M3
        } else {
            is_float(cfg.a_dtype)
        };
        a_ok && is_float(cfg.c_dtype)
    }

    fn implementation(&self, _cfg: &QGemmConfig) -> String {
        "cpu_qgemm_ref".into()
    }

    fn execute(&self, ctx: &mut QGemmContext<'_>) -> Result<(), KernelError> {
        expect_rank("a", &ctx.a, 2)?;
        expect_rank("c", &ctx.c, 2)?;
        let (m, k) = (ctx.a.shape[0], ctx.a.shape[1]);
        let n = ctx.c.shape[1];
        let cfg = ctx.cfg;
        if (n, k) != (cfg.n as usize, cfg.k as usize) || ctx.c.shape[0] != m {
            return Err(invalid(format!(
                "qgemm: a {:?} and c {:?} do not match the config {cfg}",
                ctx.a.shape.as_slice(),
                ctx.c.shape.as_slice()
            )));
        }
        let data = load_bytes(&ctx.b)?;
        if data.len() != n * weight_bytes_per_row(cfg.scheme, k) {
            return Err(invalid(format!(
                "qgemm: {} weight bytes for {n} × {k} in scheme {}",
                data.len(),
                cfg.scheme.as_str()
            )));
        }
        let scales = match cfg.scheme {
            QuantSchemeDesc::Mxfp4 => quant::mxfp4_scales_as_f32(&load_bytes(&ctx.b_scales)?),
            _ => load(&ctx.b_scales)?,
        };
        let zeros = ctx.b_zeros.as_ref().map(load_bytes).transpose()?;
        let w = quant::dequantize(cfg.scheme, &data, &scales, zeros.as_deref(), n, k);
        let a = activations(ctx, m, k)?;
        let shape = math::GemmShape {
            m,
            n,
            k,
            trans_b: true,
        };
        let c = math::gemm(&a, &w, &[], &shape, ctx.alpha, 0.0);
        store(&ctx.c, &c)
    }
}

impl QuantizeActKernel for CpuReference {
    fn supports(&self, cfg: &QuantizeActConfig) -> bool {
        let out_ok = match cfg.mode {
            ActQuantDesc::None => false,
            ActQuantDesc::Mxfp4Emulated => is_float(cfg.out_dtype),
            _ => cfg.out_dtype == DType::F8E4M3,
        };
        out_ok && is_float(cfg.x_dtype)
    }

    fn implementation(&self, _cfg: &QuantizeActConfig) -> String {
        "cpu_quantize_act_ref".into()
    }

    fn execute(&self, ctx: &mut QuantizeActContext<'_>) -> Result<(), KernelError> {
        expect_rank("x", &ctx.x, 2)?;
        let (rows, cols) = (ctx.x.shape[0], ctx.x.shape[1]);
        let x = load(&ctx.x)?;
        let mode = ctx.cfg.mode;
        if mode.is_fp8() {
            let (bytes, scales) =
                quant::quantize_activations_fp8(&x, rows, cols, mode, ctx.static_scale);
            store_bytes(&ctx.out, &bytes)?;
            store(&ctx.scales, &scales)
        } else {
            let mut y = x;
            let scales =
                quant::quantize_dequantize_activations(&mut y, rows, cols, mode, ctx.static_scale);
            store(&ctx.out, &y)?;
            store(&ctx.scales, &scales)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use turbine_core::types::DeviceId;
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceMemory, Tensor};

    use super::*;
    use crate::cpu::quant::{
        dequantize, fp8_e4m3_round, mxfp4_quantize_group, mxfp4_scales_as_f32,
        quantize_dequantize_activations,
    };
    use crate::ops::KernelProvider;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn mem() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(0), 1 << 26)
    }

    fn bytes_tensor(
        mem: &Arc<dyn DeviceMemory>,
        shape: &[usize],
        dtype: DType,
        b: &[u8],
    ) -> Tensor {
        let mut t = Tensor::empty(mem, shape, dtype).expect("alloc");
        t.storage.copy_from_host(0, b).expect("upload");
        t
    }

    fn f32_tensor(mem: &Arc<dyn DeviceMemory>, shape: &[usize], dtype: DType, v: &[f32]) -> Tensor {
        let t = Tensor::empty(mem, shape, dtype).expect("alloc");
        store(&t.view(), v).expect("store");
        t
    }

    /// A random `n × k` weight quantized in `scheme`: (data, F32 scales or E8M0 bytes as f32,
    /// zero points).
    fn quantized_weight(
        rng: &mut Rng,
        scheme: QuantSchemeDesc,
        n: usize,
        k: usize,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>) {
        let count = scheme.scale_count(n, k);
        match scheme {
            QuantSchemeDesc::Fp8Tensor
            | QuantSchemeDesc::Fp8Channel
            | QuantSchemeDesc::Fp8Block { .. } => {
                let data = (0..n * k)
                    .map(|_| fp8_e4m3_round(rng.next() * 400.0))
                    .collect();
                let scales = (0..count)
                    .map(|_| 0.001 + rng.next().abs() * 0.01)
                    .collect();
                (data, scales, None)
            }
            QuantSchemeDesc::Int4GroupZp { .. } | QuantSchemeDesc::Int4GroupSym { .. } => {
                let data = (0..n * k / 2)
                    .map(|_| ((rng.next().abs() * 255.0) as u32 & 0xff) as u8)
                    .collect();
                let scales = (0..count).map(|_| 0.01 + rng.next().abs() * 0.05).collect();
                let zeros = matches!(scheme, QuantSchemeDesc::Int4GroupZp { .. }).then(|| {
                    (0..count)
                        .map(|_| ((rng.next().abs() * 15.0) as u32) as u8)
                        .collect()
                });
                (data, scales, zeros)
            }
            QuantSchemeDesc::Mxfp4 => {
                let mut data = Vec::new();
                let mut exps = Vec::new();
                for _ in 0..n * k / 32 {
                    let mut g = [0f32; 32];
                    for v in g.iter_mut() {
                        *v = rng.next() * 3.0;
                    }
                    let (p, e) = mxfp4_quantize_group(&g);
                    data.extend_from_slice(&p);
                    exps.push(e);
                }
                (data, mxfp4_scales_as_f32(&exps), None)
            }
        }
    }

    /// For every scheme and activation mode, the CPU `qgemm` equals `gemm(dequantize(W),
    /// qdq(A))` bit for bit, reading every scale layout from device views. Breaks if a layout,
    /// a scale index or the FP8 activation decode is wrong.
    #[test]
    fn matches_dequantized_gemm() {
        let mem = mem();
        let cpu = CpuReference;
        let (m, n, k) = (5, 16, 256);
        let schemes = [
            QuantSchemeDesc::Fp8Tensor,
            QuantSchemeDesc::Fp8Channel,
            QuantSchemeDesc::Fp8Block {
                block_n: 8,
                block_k: 128,
            },
            QuantSchemeDesc::Int4GroupZp { group: 64 },
            QuantSchemeDesc::Int4GroupSym { group: 128 },
            QuantSchemeDesc::Mxfp4,
        ];
        let modes = [
            ActQuantDesc::None,
            ActQuantDesc::Fp8Tensor,
            ActQuantDesc::Fp8Token,
            ActQuantDesc::Fp8Group { group: 128 },
            ActQuantDesc::Mxfp4Emulated,
        ];
        let mut rng = Rng(7);
        for scheme in schemes {
            for mode in modes {
                let (data, scales, zeros) = quantized_weight(&mut rng, scheme, n, k);
                let x: Vec<f32> = (0..m * k)
                    .map(|_| half::bf16::from_f32(rng.next() * 3.0).to_f32())
                    .collect();
                // Reference: dequantized weight, quantize-dequantized activations.
                let mut xq = x.clone();
                quantize_dequantize_activations(&mut xq, m, k, mode, 0.02);
                let w = dequantize(scheme, &data, &scales, zeros.as_deref(), n, k);
                let shape = math::GemmShape {
                    m,
                    n,
                    k,
                    trans_b: true,
                };
                let want: Vec<f32> = math::gemm(&xq, &w, &[], &shape, 1.0, 0.0)
                    .into_iter()
                    .map(|v| crate::round_to(DType::F32, v))
                    .collect();

                // Through the ops: quantize_act, then qgemm.
                let x_t = f32_tensor(&mem, &[m, k], DType::BF16, &x);
                let (a, a_scales, a_dtype) = if mode == ActQuantDesc::None {
                    (x_t, None, DType::BF16)
                } else {
                    let out_dtype = if mode.is_fp8() {
                        DType::F8E4M3
                    } else {
                        DType::BF16
                    };
                    let qcfg = QuantizeActConfig {
                        cols: k as u32,
                        mode,
                        x_dtype: DType::BF16,
                        out_dtype,
                    };
                    assert!(QuantizeActKernel::supports(&cpu, &qcfg), "{qcfg}");
                    let out = Tensor::empty(&mem, &[m, k], out_dtype).unwrap();
                    let s = Tensor::empty(&mem, &[mode.scale_count(m, k)], DType::F32).unwrap();
                    cpu.quantize_act()
                        .unwrap()
                        .execute(&mut QuantizeActContext {
                            cfg: qcfg,
                            x: x_t.view(),
                            out: out.view(),
                            scales: s.view(),
                            static_scale: 0.02,
                        })
                        .unwrap();
                    let keep = mode.is_fp8().then_some(s);
                    (out, keep, out_dtype)
                };
                let cfg = QGemmConfig {
                    n: n as u32,
                    k: k as u32,
                    scheme,
                    act_quant: if mode.is_fp8() {
                        mode
                    } else {
                        ActQuantDesc::None
                    },
                    a_dtype,
                    c_dtype: DType::F32,
                };
                assert!(QGemmKernel::supports(&cpu, &cfg), "{cfg}");
                let row_bytes = scheme.data_bytes(1, k);
                let b = bytes_tensor(&mem, &[n, row_bytes], DType::U8, &data);
                let b_scales = match scheme {
                    QuantSchemeDesc::Mxfp4 => {
                        let e: Vec<u8> = scales.iter().map(|&v| v as u8).collect();
                        bytes_tensor(&mem, &[e.len()], DType::U8, &e)
                    }
                    _ => f32_tensor(&mem, &[scales.len()], DType::F32, &scales),
                };
                let z = zeros
                    .as_ref()
                    .map(|z| bytes_tensor(&mem, &[z.len()], DType::U8, z));
                let c = Tensor::empty(&mem, &[m, n], DType::F32).unwrap();
                cpu.qgemm()
                    .unwrap()
                    .execute(&mut QGemmContext {
                        cfg,
                        a: a.view(),
                        a_scales: a_scales.as_ref().map(Tensor::view),
                        b: b.view(),
                        b_scales: b_scales.view(),
                        b_zeros: z.as_ref().map(Tensor::view),
                        c: c.view(),
                        alpha: 1.0,
                        prefill: false,
                    })
                    .unwrap();
                let got = load(&c.view()).unwrap();
                assert_eq!(got, want, "{} / {}", scheme.as_str(), mode.as_str());
            }
        }
    }

    /// A mismatched weight size is refused, not read out of bounds.
    #[test]
    fn rejects_wrong_weight_bytes() {
        let mem = mem();
        let cpu = CpuReference;
        let cfg = QGemmConfig {
            n: 4,
            k: 64,
            scheme: QuantSchemeDesc::Int4GroupSym { group: 64 },
            act_quant: ActQuantDesc::None,
            a_dtype: DType::BF16,
            c_dtype: DType::BF16,
        };
        let a = f32_tensor(&mem, &[1, 64], DType::BF16, &[0.5; 64]);
        let b = bytes_tensor(&mem, &[4, 16], DType::U8, &[0; 64]); // needs 4 × 32
        let s = f32_tensor(&mem, &[4], DType::F32, &[1.0; 4]);
        let c = Tensor::empty(&mem, &[1, 4], DType::BF16).unwrap();
        let err = cpu
            .qgemm()
            .unwrap()
            .execute(&mut QGemmContext {
                cfg,
                a: a.view(),
                a_scales: None,
                b: b.view(),
                b_scales: s.view(),
                b_zeros: None,
                c: c.view(),
                alpha: 1.0,
                prefill: true,
            })
            .unwrap_err();
        assert!(err.to_string().contains("weight bytes"), "{err}");
    }
}
