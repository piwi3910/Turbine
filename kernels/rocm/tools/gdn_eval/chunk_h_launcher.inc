// Turbine gdn_eval: appended to AITER chunk_gated_delta_rule_fwd_h.cu cut before its
// LAUNCH/DISPATCH macros (1,536 instantiations), launching the one instantiation
// the evaluation uses: initial state, final state, v_new saved, dense (non-varlen),
// natural exp, no gk, token-major g, F32 state, BF16 chunk snapshots.
template <int BV_P>
void eval_launch(const hip_bfloat16* k, const hip_bfloat16* w, const hip_bfloat16* u, const float* g,
                 const void* h0, void* h, hip_bfloat16* v_new, void* ht, const int32_t* dummy, int T, int H,
                 int Hg, hipStream_t s)
{
    hipLaunchKernelGGL(
        (chunk_gated_delta_rule_fwd_h_hip_kernel<BV_P, true, true, true, false, false, false, false, false, true>),
        dim3(V_DIM / BV_P, H), dim3(BLOCK_THREADS), 0, s, k, w, u, g, (const float*)nullptr, h0,
        (const int32_t*)nullptr, h, v_new, ht, dummy, dummy, (T + BT - 1) / BT, T, H, Hg, Hg * K_DIM,
        (int64_t)T * H, (int64_t)1, (int64_t)H);
}
}  // namespace

void aiter_eval_chunk_h(int bv, const void* k, const void* w, const void* u, const float* g, const void* h0,
                        void* h, void* v_new, void* ht, const int32_t* dummy, int T, int H, int Hg,
                        hipStream_t s)
{
    auto K = (const hip_bfloat16*)k;
    auto W = (const hip_bfloat16*)w;
    auto U = (const hip_bfloat16*)u;
    auto VN = (hip_bfloat16*)v_new;
    if (bv == 64) eval_launch<64>(K, W, U, g, h0, h, VN, ht, dummy, T, H, Hg, s);
    else if (bv == 32) eval_launch<32>(K, W, U, g, h0, h, VN, ht, dummy, T, H, Hg, s);
    else eval_launch<16>(K, W, U, g, h0, h, VN, ht, dummy, T, H, Hg, s);
}
