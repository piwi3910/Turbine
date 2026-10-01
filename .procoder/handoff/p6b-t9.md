# Handoff: p6b-t9 (Task 9 TurboQuant tier proof)

Branch `p6b-t9`. Builder rotated after step 1 (MSE-only K). The second builder ran and judged the Task 9 lab gates (below); no row flipped.

## Done

- 521816e `tq_loss_on_real_kv` lab diagnostic; user decision "6b Task 9: TurboQuant K quantizer" A (K becomes MSE-only).
- 2c3c1f0 `feat(kv): TurboQuant K is MSE-only (tq4 K/V 4 bits, tq2 K/V 2 bits)`. It replaces the earlier `wip:` commit. It covers the
  codec (`encode_vec` / `decode_vec`, `qjl.rs` removed), `cpu::tq_attention`, the HIP `tq_device.hpp`, `kv_transcode_tq.hip` and
  `paged_attention_mixed.hip` (no S·Rq prep, no nibble tables), the table upload (`[layers][heads][2·128]`: 229 KB for Llama, 262 KB for
  OLMoE, down from 14.9 / 17.0 MB), the ABI v2.11 header docs (still minor 11, unshipped), `tq_loss.rs` (shipped formats only), spec S-4 /
  S-5 and their acceptance line, the contract, plan Task 7 and `docs/extending/kv-format.md`. `kernels/rocm/tools/attn_eval.cpp` (off by
  default) is marked as frozen at its evaluation commit.
- `nmse_bound` comes from measurements. The conformance suite's largest block nmse is tq4 0.0098 and tq2 0.119 (Gaussian and outlier
  blocks, BF16 and FP8 L0). Real K/V reach 0.0094 and 0.118. The bounds stay at 0.0125 and 0.15, about a 25 % margin.
- Gate: `gate: ok crates=all passed=896`.
- Host mutation: masking the low code bit of K fails `k_mse_bound_records` and `layout_round_trip`.
- GPU mutation, run 1001145445-17ecfcb0. Swapping the K/V signs in `decode_chunk` and dropping the K norm in `tq_score` fails both
  `kv_transcode_matches_cpu` and `paged_mixed_matches_cpu`. The tree was restored.
- Lab `--tier quick`, run 1001141634-0bbd1fbc. `kv_transcode_matches_cpu`, `paged_mixed_matches_cpu`, `tq_kv_matches_reference`,
  `lossy_tier_reuse_tq4` and `tq_kv_serves_on_the_device` all pass. One failure, `turbine-server engine::tp::tests::static_tiers_flood_then_resume`
  (static resumed `[0,0,0,0]` against `[32,…]`), is unrelated: it is a BF16 L2 test on the CPU, not owned by this branch, and passed 5 out
  of 5 reruns on novanas and in the gate. It is a flake under lab load and goes to the lead.
- Lab `tq_loss_on_real_kv`, run 1001145040-2cb1960f, exit 0. The shipped formats now equal the old `k4mse` / `k2mse` columns exactly:

| mean      | K nmse | V nmse | score e_sd | out_rel | TV     |
| --------- | ------ | ------ | ---------- | ------- | ------ |
| Llama tq4 | 0.0089 | 0.0092 | 0.277      | 0.0803  | 0.0341 |
| Llama tq2 | 0.1115 | 0.1152 | 0.956      | 0.5477  | 0.2052 |
| OLMoE tq4 | 0.0089 | 0.0093 | 0.128      | 0.0428  | 0.0297 |
| OLMoE tq2 | 0.1101 | 0.1152 | 0.512      | 0.2219  | 0.1328 |

## Task 9 lab gates (2026-10-01, second builder): run and judged, nothing flipped

Numbers and commands: `.procoder/perf-log.md`, Phase 6b, "TurboQuant lower-tier gates". Commits: cb7dd38 (Llama eval reports and pairs),
e88c333 (`kv_gpu lossy_tier_reuse_tq4` bound), 4a8469c (OLMoE baselines and candidates), then the perf log, the `TIER_FORMAT_REFUSALS` doc
comment and the AGENTS.md support line. Labbook set `phase-6b-kv-compression`: 4 lab-bench, 18 multi-turn and 1 kv_gpu run. No serve Job of
this branch is left.

| Model × format | golden c1 / c16 | GSM8K-sp median (BF16) | McNemar (9 pairs) | multi-turn ratio vs `l0` | L1 blocks/GiB vs `l0` (tail share) |
| -------------- | --------------- | ---------------------- | ----------------- | ------------------------ | ---------------------------------- |
| Llama `tq4`    | PASS / PASS     | 0.775 (0.780) PASS     | n.s. (p ≥ 0.23)   | 0.885 vs 0.905, miss     | 2.96× (0.079)                      |
| Llama `tq2`    | PASS / PASS     | 0.720 (0.780) FAIL     | 3 of 9 p < 0.05   | 0.905 vs 0.905, −0.0004  | 5.21× (0.043)                      |
| OLMoE `tq4`    | PASS / PASS     | 0.665 (0.635) PASS     | n.s. (p 0.24)     | 0.858 vs 0.861, miss     | 2.90× (0.088)                      |
| OLMoE `tq2`    | PASS / PASS     | 0.610 (0.635) FAIL     | n.s. (p 0.53)     | 0.858 vs 0.861, miss     | 4.73× (0.065)                      |

- The Llama BF16 baselines (`turbine-bf16-sp-r{1,2,3}.json`, f9a4c91) were reused: since then only TurboQuant code changed, and the recipe and
  runner are the same. OLMoE got its own three; its runs are bit-identical across repeats and its servers count 762,875 prompt tokens.
- OLMoE multi-turn could not run the spec workload: its 4,096-token context overflows at turn 1. It ran a 600-word prefix, 128-word turns, 64
  tokens and a 4 GiB L0 instead (perf log). Record this as a spec amendment if the lead accepts it.
- `kv_gpu` bound `TQ4_TIER_BOUNDS` (0.25, 0.75, 0.9) = golden's Llama batched bounds (likely 0.25, tail 0.75), which the tq4 golden16 run
  passed, with 90 % of positions as for FP8. Measured 0.242 / 0.98, a thin margin on the first-8 bound. The test is deterministic (one
  request, greedy).
- `TIER_FORMAT_REFUSALS` has one row per format, so a per-model status cannot be expressed. Both formats stay `experimental`.

## Open: decisions for the lead / user

1. `tq4` misses only the multi-turn cached-ratio criterion. On Llama the cause is promotion speed, not quality: L1→L0 promotions of `tq4`
   blocks averaged 245–289 ms in runs 1–2 (`l0` 41–61 ms, `tq2` about 100 ms), so the planner chose `recompute_cheaper` (17 and 11 plans) and
   later-turn p99 rose to 1.8–2.1 s. On OLMoE the shortfall is −0.003 with no recomputes.
   - A) Profile the `tq4` promotion path (device decode transcode plus copy) and rerun the multi-turn A/B; flip `tq4` if it then passes.
     Recommended: it is a real latency defect, and `tq2` with half the bytes promotes faster, which points at a fixed or per-bit cost in
     the decode.
   - B) Treat the ratio criterion as met within run-to-run noise (Llama r3 0.905; OLMoE −0.003) and flip `tq4` now.
   - C) Amend the criterion (for example ratio within 0.01 of `l0`, or later-turn TTFT not worse) and judge again.
2. `tq2` fails the eval on both models (drops 0.060 / 0.025). It stays `experimental`. Options: A) keep it as an experimental capacity tier
   (5.2× / 4.7× per GiB); B) mixed widths (e.g. K 4-bit, V 2-bit) as a new format, a spec change; C) drop it from the lower tiers.
3. OLMoE multi-turn workload amendment (above): accept it, or define another OLMoE workload.

## Former QJL K measurements (pre-decision; for the record)

```
layer  rot2   kurt  mean  s_sd   | per format: K nmse  V nmse  e_bias  e_sd  prod*d  out_rel  tv
    0 1.000  2.83 0.492  1.792 | tq4: 0.0504 0.0090 +0.0079 0.535 0.049 0.1058 0.0785 | tq4/k4mse: 0.0089 0.0090 +0.0309 0.230 0.009 0.0666 0.0317 | tq2: 0.5442 0.1131 +0.1045 1.781 0.547 0.6928 0.3453 | tq2/k2mse: 0.1109 0.1131 +0.4380 0.817 0.148 0.3676 0.1743 |
    1 1.000  2.86 0.497  1.784 | tq4: 0.0521 0.0091 -0.0192 0.665 0.053 0.1353 0.0432 | tq4/k4mse: 0.0090 0.0091 +0.0633 0.281 0.010 0.0831 0.0188 | tq2: 0.5637 0.1148 -0.1203 2.186 0.577 1.1796 0.2866 | tq2/k2mse: 0.1129 0.1148 +0.9056 0.960 0.210 0.5131 0.1485 |
    2 1.000  2.83 0.488  1.916 | tq4: 0.0518 0.0093 -0.0055 0.628 0.050 0.1617 0.0435 | tq4/k4mse: 0.0090 0.0093 +0.0725 0.277 0.010 0.0964 0.0172 | tq2: 0.5548 0.1160 -0.0580 1.998 0.512 1.4972 0.2256 | tq2/k2mse: 0.1121 0.1160 +1.0179 0.956 0.250 0.8867 0.1404 |
    3 1.000  2.79 0.528  1.863 | tq4: 0.0497 0.0092 -0.0035 0.619 0.051 0.1319 0.0378 | tq4/k4mse: 0.0088 0.0092 +0.0546 0.269 0.010 0.0817 0.0161 | tq2: 0.5442 0.1152 -0.1079 2.045 0.561 0.8874 0.2173 | tq2/k2mse: 0.1103 0.1152 +0.8933 0.943 0.226 0.7357 0.1801 |
    4 1.000  2.80 0.544  1.883 | tq4: 0.0508 0.0093 -0.0200 0.645 0.051 0.1062 0.0450 | tq4/k4mse: 0.0089 0.0093 +0.0612 0.275 0.010 0.0842 0.0238 | tq2: 0.5509 0.1148 -0.3249 2.065 0.535 0.8023 0.2396 | tq2/k2mse: 0.1112 0.1148 +0.8830 0.944 0.205 0.5858 0.1723 |
    5 1.000  2.89 0.492  1.944 | tq4: 0.0517 0.0093 +0.0187 0.628 0.054 0.1292 0.0775 | tq4/k4mse: 0.0090 0.0093 +0.0594 0.277 0.011 0.0882 0.0372 | tq2: 0.5620 0.1152 +0.0911 2.012 0.556 0.8902 0.3704 | tq2/k2mse: 0.1139 0.1152 +0.8172 0.998 0.228 0.5503 0.2140 |
    6 1.000  2.82 0.512  1.902 | tq4: 0.0513 0.0093 +0.0179 0.647 0.052 0.1225 0.0890 | tq4/k4mse: 0.0089 0.0093 +0.0593 0.279 0.010 0.0753 0.0397 | tq2: 0.5563 0.1149 +0.2540 2.089 0.551 0.9868 0.4558 | tq2/k2mse: 0.1116 0.1149 +0.8406 0.972 0.205 0.4259 0.2295 |
    7 1.000  2.76 0.517  2.029 | tq4: 0.0500 0.0092 -0.0135 0.655 0.051 0.1089 0.0792 | tq4/k4mse: 0.0088 0.0092 +0.0442 0.277 0.009 0.0671 0.0353 | tq2: 0.5419 0.1155 -0.2159 2.103 0.528 0.6693 0.3863 | tq2/k2mse: 0.1099 0.1155 +0.6698 0.959 0.161 0.2967 0.1697 |
    8 1.000  2.78 0.525  1.961 | tq4: 0.0500 0.0092 -0.0066 0.631 0.050 0.1034 0.1194 | tq4/k4mse: 0.0088 0.0092 +0.0402 0.270 0.009 0.0588 0.0510 | tq2: 0.5380 0.1157 -0.0232 2.030 0.519 0.6239 0.4815 | tq2/k2mse: 0.1105 0.1157 +0.6340 0.928 0.159 0.2937 0.2468 |
    9 1.000  2.74 0.523  1.959 | tq4: 0.0502 0.0092 +0.0162 0.610 0.051 0.1261 0.1212 | tq4/k4mse: 0.0088 0.0092 +0.0381 0.262 0.010 0.0644 0.0484 | tq2: 0.5409 0.1156 +0.2546 2.005 0.561 0.6832 0.5004 | tq2/k2mse: 0.1091 0.1156 +0.5375 0.908 0.153 0.2763 0.2105 |
   10 1.000  2.82 0.484  2.005 | tq4: 0.0507 0.0093 -0.0384 0.566 0.054 0.1023 0.1228 | tq4/k4mse: 0.0089 0.0093 +0.0423 0.236 0.010 0.0607 0.0532 | tq2: 0.5504 0.1152 -0.4713 1.867 0.620 0.4807 0.4131 | tq2/k2mse: 0.1117 0.1152 +0.6243 0.830 0.180 0.2808 0.2398 |
   11 1.000  2.74 0.565  2.039 | tq4: 0.0503 0.0093 -0.0132 0.647 0.050 0.0976 0.0859 | tq4/k4mse: 0.0088 0.0093 +0.0589 0.286 0.010 0.0603 0.0385 | tq2: 0.5451 0.1152 -0.2683 2.067 0.519 0.5887 0.3966 | tq2/k2mse: 0.1091 0.1152 +0.8579 0.970 0.200 0.2847 0.2105 |
   12 1.000  2.99 0.502  2.058 | tq4: 0.0537 0.0092 -0.0027 0.598 0.053 0.1136 0.1436 | tq4/k4mse: 0.0094 0.0092 +0.0590 0.261 0.011 0.0595 0.0610 | tq2: 0.5739 0.1147 -0.0425 1.898 0.535 0.5857 0.5064 | tq2/k2mse: 0.1175 0.1147 +0.7072 0.904 0.196 0.3299 0.2906 |
   13 1.000  2.92 0.527  2.141 | tq4: 0.0508 0.0092 +0.0217 0.649 0.050 0.1138 0.1187 | tq4/k4mse: 0.0091 0.0092 +0.0574 0.283 0.010 0.0588 0.0484 | tq2: 0.5491 0.1148 +0.3112 2.055 0.510 0.7086 0.5293 | tq2/k2mse: 0.1143 0.1148 +0.7948 0.993 0.191 0.3693 0.2726 |
   14 1.000  2.80 0.538  2.054 | tq4: 0.0505 0.0094 +0.0229 0.698 0.051 0.1243 0.1220 | tq4/k4mse: 0.0088 0.0094 +0.0482 0.291 0.009 0.0599 0.0498 | tq2: 0.5527 0.1166 +0.2825 2.344 0.586 0.8597 0.5535 | tq2/k2mse: 0.1105 0.1166 +0.7033 1.014 0.160 0.3340 0.2476 |
   15 1.000  2.86 0.535  1.869 | tq4: 0.0514 0.0092 +0.0067 0.632 0.052 0.1330 0.0902 | tq4/k4mse: 0.0090 0.0092 +0.0622 0.272 0.010 0.0726 0.0399 | tq2: 0.5517 0.1157 -0.1636 2.019 0.538 0.7912 0.3868 | tq2/k2mse: 0.1130 0.1157 +0.8109 0.925 0.198 0.4764 0.2422 |
   16 1.000  2.87 0.540  1.795 | tq4: 0.0509 0.0092 -0.0294 0.647 0.055 0.1543 0.0567 | tq4/k4mse: 0.0090 0.0092 +0.0644 0.274 0.010 0.0952 0.0252 | tq2: 0.5505 0.1150 -0.3861 2.101 0.600 0.9430 0.2722 | tq2/k2mse: 0.1130 0.1150 +0.9488 0.945 0.236 0.7605 0.2051 |
   17 1.000  2.85 0.557  1.779 | tq4: 0.0516 0.0093 +0.0106 0.588 0.051 0.1457 0.0696 | tq4/k4mse: 0.0090 0.0093 +0.0617 0.255 0.010 0.0960 0.0293 | tq2: 0.5495 0.1158 +0.1051 1.831 0.499 0.9639 0.3023 | tq2/k2mse: 0.1123 0.1158 +0.8528 0.875 0.221 0.6261 0.1916 |
   18 1.000  2.85 0.556  1.782 | tq4: 0.0514 0.0093 +0.0305 0.585 0.048 0.1764 0.0782 | tq4/k4mse: 0.0090 0.0093 +0.0539 0.260 0.010 0.0873 0.0310 | tq2: 0.5578 0.1161 +0.2947 1.867 0.504 1.2315 0.3854 | tq2/k2mse: 0.1127 0.1161 +0.8184 0.883 0.204 0.7268 0.2052 |
   19 1.000  2.81 0.545  1.962 | tq4: 0.0506 0.0092 +0.0118 0.686 0.050 0.1704 0.0664 | tq4/k4mse: 0.0089 0.0092 +0.0570 0.297 0.010 0.0815 0.0251 | tq2: 0.5456 0.1143 +0.0847 2.343 0.589 1.7887 0.4252 | tq2/k2mse: 0.1112 0.1143 +0.8872 1.005 0.193 0.5591 0.1727 |
   20 1.000  2.77 0.556  1.909 | tq4: 0.0497 0.0093 +0.0036 0.630 0.051 0.2111 0.0616 | tq4/k4mse: 0.0088 0.0093 +0.0491 0.269 0.010 0.1000 0.0244 | tq2: 0.5455 0.1148 +0.0795 2.091 0.565 1.6700 0.3353 | tq2/k2mse: 0.1096 0.1148 +0.7547 0.948 0.190 0.6539 0.1675 |
   21 1.000  2.83 0.563  2.071 | tq4: 0.0504 0.0094 +0.0332 0.656 0.048 0.2161 0.0536 | tq4/k4mse: 0.0089 0.0094 +0.0691 0.288 0.010 0.0973 0.0201 | tq2: 0.5433 0.1163 +0.3054 2.192 0.550 1.8656 0.3309 | tq2/k2mse: 0.1119 0.1163 +0.9597 0.980 0.211 0.7482 0.1564 |
   22 1.000  2.74 0.567  2.138 | tq4: 0.0504 0.0092 -0.0263 0.726 0.051 0.2054 0.0338 | tq4/k4mse: 0.0088 0.0092 +0.0791 0.326 0.011 0.1127 0.0151 | tq2: 0.5453 0.1148 -0.1282 2.303 0.512 2.5955 0.2869 | tq2/k2mse: 0.1091 0.1148 +1.1614 1.087 0.244 0.9248 0.1365 |
   23 1.000  2.81 0.567  1.930 | tq4: 0.0510 0.0092 -0.0229 0.647 0.052 0.1606 0.0767 | tq4/k4mse: 0.0090 0.0092 +0.0701 0.292 0.011 0.0901 0.0318 | tq2: 0.5486 0.1150 -0.2644 2.202 0.609 1.0524 0.3429 | tq2/k2mse: 0.1112 0.1150 +0.9954 0.973 0.240 0.7698 0.2331 |
   24 1.000  2.79 0.563  1.980 | tq4: 0.0507 0.0093 +0.0210 0.642 0.048 0.1914 0.0772 | tq4/k4mse: 0.0089 0.0093 +0.0659 0.289 0.010 0.0989 0.0314 | tq2: 0.5487 0.1162 +0.3147 2.077 0.513 1.8059 0.4396 | tq2/k2mse: 0.1105 0.1162 +0.9374 1.015 0.222 0.8894 0.2320 |
   25 1.000  2.68 0.585  1.974 | tq4: 0.0495 0.0092 +0.0242 0.636 0.048 0.1768 0.0617 | tq4/k4mse: 0.0086 0.0092 +0.0514 0.274 0.009 0.0826 0.0220 | tq2: 0.5327 0.1154 +0.2541 2.145 0.557 1.6595 0.3682 | tq2/k2mse: 0.1064 0.1154 +0.7923 0.944 0.181 0.6188 0.1474 |
   26 1.000  2.90 0.528  2.046 | tq4: 0.0518 0.0092 +0.0113 0.703 0.053 0.2059 0.1244 | tq4/k4mse: 0.0091 0.0092 +0.0608 0.298 0.010 0.0919 0.0512 | tq2: 0.5559 0.1149 +0.0199 2.332 0.584 1.0954 0.4942 | tq2/k2mse: 0.1142 0.1149 +0.8393 1.032 0.190 0.5737 0.2774 |
   27 1.000  2.76 0.575  1.933 | tq4: 0.0497 0.0091 +0.0053 0.700 0.050 0.1363 0.0841 | tq4/k4mse: 0.0088 0.0091 +0.0570 0.302 0.010 0.0761 0.0369 | tq2: 0.5389 0.1135 -0.0927 2.276 0.525 0.9394 0.4320 | tq2/k2mse: 0.1102 0.1135 +0.8910 1.051 0.192 0.4776 0.2304 |
mean tq4        K nmse 0.0508  V nmse 0.0092  e_bias +0.0022  e_sd 0.639  prod*d 0.051  out_rel 0.1452  tv 0.0808
mean tq4/k4mse  K nmse 0.0089  V nmse 0.0092  e_bias +0.0568  e_sd 0.277  prod*d 0.010  out_rel 0.0803  tv 0.0341
mean tq2        K nmse 0.5494  V nmse 0.1152  e_bias +0.0032  e_sd 2.083  prod*d 0.549  out_rel 1.0906  tv 0.3825
mean tq2/k2mse  K nmse 0.1115  V nmse 0.1152  e_bias +0.8205  e_sd 0.956  prod*d 0.200  out_rel 0.5477  tv 0.2052
layer  rot2   kurt  mean  s_sd   | per format: K nmse  V nmse  e_bias  e_sd  prod*d  out_rel  tv
    0 1.000  2.26 0.203  1.393 | tq4: 0.0466 0.0091 -0.0028 0.258 0.050 0.0631 0.0159 | tq4/k4mse: 0.0083 0.0091 +0.0074 0.112 0.009 0.0569 0.0092 | tq2: 0.4870 0.1070 +0.0203 0.808 0.493 0.2990 0.0781 | tq2/k2mse: 0.0978 0.1070 +0.0430 0.556 0.235 0.2479 0.0424 |
    1 1.000  2.48 0.271  1.372 | tq4: 0.0492 0.0091 +0.0017 0.219 0.046 0.0490 0.0324 | tq4/k4mse: 0.0087 0.0091 +0.0113 0.095 0.009 0.0397 0.0139 | tq2: 0.5341 0.1144 -0.0182 0.731 0.513 0.2825 0.1231 | tq2/k2mse: 0.1071 0.1144 +0.1582 0.391 0.170 0.2304 0.0644 |
    2 1.000  2.64 0.347  1.797 | tq4: 0.0493 0.0092 +0.0004 0.304 0.050 0.0437 0.0453 | tq4/k4mse: 0.0086 0.0092 +0.0189 0.126 0.009 0.0276 0.0201 | tq2: 0.5289 0.1157 +0.0167 0.976 0.521 0.2492 0.1791 | tq2/k2mse: 0.1062 0.1157 +0.2790 0.504 0.181 0.1880 0.1018 |
    3 1.000  2.74 0.189  1.458 | tq4: 0.0505 0.0092 -0.0023 0.267 0.051 0.0589 0.0555 | tq4/k4mse: 0.0089 0.0092 +0.0115 0.111 0.009 0.0402 0.0265 | tq2: 0.5458 0.1152 -0.0395 0.860 0.524 0.2606 0.1837 | tq2/k2mse: 0.1098 0.1152 +0.1626 0.436 0.153 0.2014 0.1044 |
    4 1.000  2.73 0.261  1.604 | tq4: 0.0498 0.0093 -0.0023 0.297 0.050 0.0588 0.0560 | tq4/k4mse: 0.0088 0.0093 +0.0167 0.125 0.009 0.0361 0.0247 | tq2: 0.5353 0.1155 -0.0135 0.948 0.507 0.2823 0.1978 | tq2/k2mse: 0.1092 0.1155 +0.2527 0.492 0.172 0.1713 0.1010 |
    5 1.000  2.78 0.273  1.496 | tq4: 0.0510 0.0094 +0.0012 0.273 0.052 0.0586 0.0703 | tq4/k4mse: 0.0090 0.0094 +0.0159 0.116 0.010 0.0383 0.0306 | tq2: 0.5507 0.1164 +0.0261 0.886 0.549 0.2603 0.2397 | tq2/k2mse: 0.1120 0.1164 +0.2150 0.454 0.176 0.1897 0.1317 |
    6 1.000  2.77 0.255  1.698 | tq4: 0.0509 0.0092 +0.0010 0.318 0.050 0.0594 0.0572 | tq4/k4mse: 0.0089 0.0092 +0.0190 0.137 0.010 0.0395 0.0265 | tq2: 0.5518 0.1154 +0.0222 1.027 0.526 0.2407 0.1943 | tq2/k2mse: 0.1112 0.1154 +0.2838 0.576 0.206 0.1775 0.1062 |
    7 1.000  2.77 0.262  1.874 | tq4: 0.0508 0.0093 -0.0028 0.318 0.050 0.0607 0.0673 | tq4/k4mse: 0.0089 0.0093 +0.0167 0.137 0.009 0.0434 0.0334 | tq2: 0.5447 0.1161 +0.0105 1.025 0.517 0.2423 0.2268 | tq2/k2mse: 0.1104 0.1161 +0.2186 0.536 0.165 0.1907 0.1398 |
    8 1.000  2.82 0.270  1.957 | tq4: 0.0500 0.0093 -0.0047 0.359 0.049 0.0614 0.0627 | tq4/k4mse: 0.0089 0.0093 +0.0235 0.153 0.009 0.0384 0.0279 | tq2: 0.5416 0.1161 -0.0521 1.146 0.501 0.2493 0.2154 | tq2/k2mse: 0.1098 0.1161 +0.3153 0.641 0.194 0.1976 0.1223 |
    9 1.000  2.82 0.278  2.017 | tq4: 0.0507 0.0093 -0.0056 0.326 0.051 0.0683 0.0877 | tq4/k4mse: 0.0089 0.0093 +0.0167 0.139 0.009 0.0408 0.0394 | tq2: 0.5441 0.1156 -0.0580 1.052 0.530 0.2678 0.2752 | tq2/k2mse: 0.1109 0.1156 +0.2282 0.543 0.165 0.2050 0.1682 |
   10 1.000  2.87 0.238  1.763 | tq4: 0.0513 0.0094 -0.0001 0.305 0.051 0.0815 0.0855 | tq4/k4mse: 0.0090 0.0094 +0.0171 0.131 0.010 0.0440 0.0372 | tq2: 0.5531 0.1164 +0.0168 0.999 0.545 0.3548 0.2812 | tq2/k2mse: 0.1127 0.1164 +0.2477 0.496 0.168 0.2460 0.1724 |
   11 1.000  2.88 0.224  1.912 | tq4: 0.0519 0.0093 +0.0007 0.344 0.052 0.0699 0.0809 | tq4/k4mse: 0.0090 0.0093 +0.0218 0.148 0.010 0.0401 0.0347 | tq2: 0.5566 0.1158 -0.0124 1.128 0.555 0.3258 0.2720 | tq2/k2mse: 0.1133 0.1158 +0.3052 0.580 0.187 0.2155 0.1624 |
   12 1.000  2.86 0.208  1.940 | tq4: 0.0513 0.0093 -0.0007 0.328 0.052 0.0842 0.0980 | tq4/k4mse: 0.0090 0.0093 +0.0148 0.139 0.009 0.0505 0.0448 | tq2: 0.5534 0.1158 +0.0034 1.063 0.548 0.3665 0.2944 | tq2/k2mse: 0.1124 0.1158 +0.2168 0.543 0.166 0.2612 0.1766 |
   13 1.000  2.87 0.211  1.671 | tq4: 0.0511 0.0093 -0.0017 0.307 0.049 0.0954 0.0789 | tq4/k4mse: 0.0090 0.0093 +0.0198 0.131 0.009 0.0540 0.0366 | tq2: 0.5503 0.1161 -0.0040 0.999 0.524 0.4187 0.2668 | tq2/k2mse: 0.1125 0.1161 +0.2691 0.501 0.170 0.2776 0.1769 |
   14 1.000  2.87 0.229  1.788 | tq4: 0.0514 0.0093 +0.0014 0.300 0.050 0.0931 0.0813 | tq4/k4mse: 0.0090 0.0093 +0.0191 0.130 0.010 0.0540 0.0366 | tq2: 0.5546 0.1160 -0.0022 0.990 0.547 0.4227 0.2590 | tq2/k2mse: 0.1125 0.1160 +0.2740 0.499 0.181 0.3133 0.1962 |
   15 1.000  2.87 0.238  1.487 | tq4: 0.0516 0.0093 +0.0021 0.287 0.051 0.0737 0.0731 | tq4/k4mse: 0.0091 0.0093 +0.0183 0.122 0.009 0.0406 0.0323 | tq2: 0.5565 0.1164 +0.0183 0.943 0.550 0.2987 0.2328 | tq2/k2mse: 0.1134 0.1164 +0.2498 0.451 0.164 0.2377 0.1588 |
mean tq4        K nmse 0.0505  V nmse 0.0093  e_bias -0.0009  e_sd 0.301  prod*d 0.050  out_rel 0.0675  tv 0.0655
mean tq4/k4mse  K nmse 0.0089  V nmse 0.0093  e_bias +0.0168  e_sd 0.128  prod*d 0.009  out_rel 0.0428  tv 0.0297
mean tq2        K nmse 0.5430  V nmse 0.1152  e_bias -0.0041  e_sd 0.974  prod*d 0.528  out_rel 0.3013  tv 0.2200
mean tq2/k2mse  K nmse 0.1101  V nmse 0.1152  e_bias +0.2324  e_sd 0.512  prod*d 0.178  out_rel 0.2219  tv 0.1328
```
