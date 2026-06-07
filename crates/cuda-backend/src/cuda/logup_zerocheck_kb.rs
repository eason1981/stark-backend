#![cfg(feature = "koala-bear-poseidon2")]

use super::*;
use crate::{
    cuda::logup_zerocheck::{
        BlockCtx, EvalCoreCtx, LogupCtx, LogupMonomialCommonCtx, LogupMonomialCtx,
        MainMatrixPtrs, MonomialAirCtx, ZerocheckCtx,
    },
    monomial::{InteractionMonomialTerm, LambdaTerm, MonomialHeader, PackedVar},
    poly::SqrtEqLayers,
};









#[link(name = "cuda_kb_all", kind = "static")]
extern "C" {
    // gkr.cu — KB bit-reversal + tree build (uses KB field arithmetic, correct for KB data)
    // Note: renamed from _bit_rev_frac_ext_build_k2 via kb_symbol_prefix.h to avoid
    // conflict with ntt_bitrev_kb.cu's stub version.
    fn _kb_gkr_bit_rev_frac_ext_build_k2(inout: *mut std::ffi::c_void, lg_domain_size: u32, alpha: EF) -> i32;

    // gkr.cu
    fn _kb_frac_build_tree_layer(
        layer: *mut Frac<EF>,
        layer_size: usize,
        revert: bool,
        alpha: EF,
        apply_alpha: bool,
    ) -> i32;

    /// Fused two-layer tree build. Applies layers i and i+1 in one kernel pass,
    /// keeping intermediate right-half nodes for revert operations.
    /// `half_i1` = N >> (i+2), where i is the first of the two layers.
    fn _kb_frac_build_tree_two_layers(layer: *mut Frac<EF>, half_i1: usize) -> i32;

    pub fn _kb_frac_compute_round_temp_buffer_size(stride: u32) -> u32;

    fn _kb_frac_compute_round(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        pq_buffer: *const Frac<EF>,
        num_x: usize,
        eq_low_cap: usize,
        lambda: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
    ) -> i32;

    /// Fused compute round + tree layer revert. Combines frac_build_tree_layer(revert=true)
    /// with compute_round for the first inner round. Modifies layer in-place for revert.
    fn _kb_frac_compute_round_and_revert(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        layer: *mut Frac<EF>,
        num_x: usize,
        eq_low_cap: usize,
        lambda: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
    ) -> i32;

    fn _kb_frac_fold_fpext_columns(
        src: *const Frac<EF>,
        dst: *mut Frac<EF>,
        size: usize,
        r: EF,
    ) -> i32;

    /// Fused compute round + fold (out-of-place). Reads from pre-fold src_pq_buffer (size
    /// src_pq_size), computes sumcheck sums, and writes folded output to dst_pq_buffer (size
    /// src_pq_size/2). IMPORTANT: src_pq_buffer and dst_pq_buffer must NOT alias.
    fn _kb_frac_compute_round_and_fold(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        src_pq_buffer: *const Frac<EF>,
        dst_pq_buffer: *mut Frac<EF>,
        src_pq_size: usize,
        eq_low_cap: usize,
        lambda: EF,
        r_prev: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
    ) -> i32;

    /// Fused compute round + fold (in-place). Reads from pre-fold pq_buffer (size src_pq_size),
    /// computes sumcheck sums, and writes folded output to the same buffer (first src_pq_size/2
    /// elements).
    fn _kb_frac_compute_round_and_fold_inplace(
        eq_xi_low: *const EF,
        eq_xi_high: *const EF,
        pq_buffer: *mut Frac<EF>,
        src_pq_size: usize,
        eq_low_cap: usize,
        lambda: EF,
        r_prev: EF,
        out_device: *mut EF,
        tmp_block_sums: *mut EF,
    ) -> i32;

    fn _kb_frac_precompute_m_build(
        pq: *const Frac<EF>,
        rem_n: usize,
        w: usize,
        lambda: EF,
        r_prev: EF,
        inline_fold: bool,
        eq_tail_low: *const EF,
        eq_tail_high: *const EF,
        eq_tail_low_cap: usize,
        tail_tile: usize,
        partial_out: *mut EF,
        partial_len: usize,
        m_total: *mut EF,
    ) -> i32;

    fn _kb_frac_precompute_m_eval_round(
        m_total: *const EF,
        w: usize,
        t: usize,
        eq_r_prefix: *const EF,
        eq_suffix: *const EF,
        out: *mut EF,
    ) -> i32;

    fn _kb_frac_multifold(
        src: *const Frac<EF>,
        dst: *mut Frac<EF>,
        rem_n: usize,
        w: usize,
        eq_r_window: *const EF,
    ) -> i32;

    fn _kb_frac_add_alpha(data: *mut std::ffi::c_void, len: usize, alpha: EF) -> i32;

    fn _kb_frac_vector_scalar_multiply_ext_fp(
        frac_vec: *mut Frac<EF>,
        scalar: F,
        length: u32,
    ) -> i32;

    // utils.cu
    fn _kb_fold_ple_from_evals(
        input_matrix: *const F,
        output_matrix: *mut EF,
        omega_skip_pows: *const F,
        inv_lagrange_denoms: *const EF,
        height: u32,
        width: u32,
        l_skip: u32,
        new_height: u32,
        rotate: bool,
    ) -> i32;

    fn _kb_interpolate_columns(
        interpolated: *mut EF,
        columns: *const *const EF,
        s_deg: usize,
        num_y: usize,
        num_columns: usize,
    ) -> i32;

    fn _kb_frac_matrix_vertically_repeat(
        out: *mut Frac<EF>,
        input: *const Frac<EF>,
        width: u32,
        lifted_height: u32,
        height: u32,
    ) -> i32;

    fn _kb_frac_matrix_vertically_repeat_ext(
        out_numerators: *mut EF,
        out_denominators: *mut EF,
        in_numerators: *const EF,
        in_denominators: *const EF,
        width: u32,
        lifted_height: u32,
        height: u32,
    ) -> i32;

    // gkr_input.cu
    fn _kb_logup_gkr_input_eval(
        is_global: bool,
        fracs: *mut Frac<EF>,
        preprocessed: *const F,
        partitioned_main: *const u64,
        public_values: *const F,
        challenges: *const EF,
        intermediates: *const EF,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        pair_idxs: *const u32,
        used_nodes_len: usize,
        height: u32,
        num_rows_per_tile: u32,
    ) -> i32;

    // logup_round0.cu
    pub fn _kb_logup_r0_temp_sums_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    pub fn _kb_logup_r0_intermediates_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    fn _kb_logup_bary_eval_interactions_round0(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        selectors_cube: *const F,
        preprocessed: *const F,
        main_parts: *const *const F,
        eq_cube: *const EF,
        public_values: *const F,
        numer_weights: *const EF,
        denom_weights: *const EF,
        denom_sum_init: EF,
        d_rules: *const std::ffi::c_void,
        rules_len: usize,
        buffer_size: u32,
        d_intermediates: *mut F,
        skip_domain: u32,
        num_x: u32,
        height: u32,
        num_cosets: u32,
        g_shift: F,
        max_temp_bytes: usize,
    ) -> i32;

    // zerocheck_round0.cu
    pub fn _kb_zerocheck_r0_temp_sums_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    pub fn _kb_zerocheck_r0_intermediates_buffer_size(
        buffer_size: u32,
        skip_domain: u32,
        num_x: u32,
        num_cosets: u32,
        max_temp_bytes: usize,
    ) -> usize;

    fn _kb_zerocheck_ntt_eval_constraints(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        selectors_cube: *const F,
        preprocessed: *const F,
        main_parts: *const *const F,
        eq_cube: *const EF,
        d_lambda_pows: *const EF,
        public_values: *const F,
        d_rules: *const std::ffi::c_void,
        rules_len: usize,
        d_used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        d_intermediates: *mut F,
        skip_domain: u32,
        num_x: u32,
        height: u32,
        num_cosets: u32,
        g_shift: F,
        max_temp_bytes: usize,
    ) -> i32;

    fn _kb_fold_selectors_round0(
        out: *mut EF,
        input: *const F,
        is_first: EF,
        is_last: EF,
        num_x: u32,
    ) -> i32;

    // mle.cu
    pub fn _kb_zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;

    pub fn _kb_zerocheck_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    fn _kb_zerocheck_eval_mle(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        eq_xi: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        lambda_pows: *const EF,
        public_values: *const F,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        lambda_len: usize,
        buffer_size: u32,
        intermediates: *mut EF,
        num_y: u32,
        num_x: u32,
    ) -> i32;

    pub fn _kb_logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;

    pub fn _kb_logup_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    fn _kb_logup_eval_mle(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        eq_xi: *const EF,
        selectors: *const EF,
        preprocessed: MainMatrixPtrs<EF>,
        main: *const MainMatrixPtrs<EF>,
        challenges: *const EF,
        eq_3bs: *const EF,
        public_values: *const F,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        pair_idxs: *const u32,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: *mut EF,
        num_y: u32,
        num_x: u32,
    ) -> i32;

    // batch_mle.cu (batch kernels always use global intermediates when buffer_size > 0)
    pub fn _kb_zerocheck_batch_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    pub fn _kb_logup_batch_mle_intermediates_buffer_size(
        buffer_size: u32,
        num_x: u32,
        num_y: u32,
    ) -> usize;

    fn _kb_zerocheck_batch_eval_mle(
        tmp_sums_buffer: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        zc_ctxs: *const ZerocheckCtx,
        air_block_offsets: *const u32,
        lambda_pows: *const EF,
        lambda_len: usize,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> i32;

    fn _kb_logup_batch_eval_mle(
        tmp_sums_buffer: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        logup_ctxs: *const LogupCtx,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> i32;

    fn _kb_zerocheck_monomial_batched(
        tmp_sums: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        air_ctxs: *const MonomialAirCtx,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> i32;

    fn _kb_zerocheck_monomial_par_y_batched(
        tmp_sums: *mut EF,
        output: *mut EF,
        block_ctxs: *const BlockCtx,
        air_ctxs: *const MonomialAirCtx,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        chunk_size: u32,
        threads_per_block: u32,
    ) -> i32;

    fn _kb_precompute_lambda_combinations(
        out: *mut EF,
        headers: *const MonomialHeader,
        lambda_terms: *const LambdaTerm<F>,
        lambda_pows: *const EF,
        num_monomials: u32,
    ) -> i32;

    // Logup monomial kernels
    fn _kb_precompute_logup_numer_combinations(
        out: *mut EF,
        headers: *const MonomialHeader,
        terms: *const InteractionMonomialTerm<F>,
        eq_3bs: *const EF,
        num_monomials: u32,
    ) -> i32;

    fn _kb_precompute_logup_denom_combinations(
        out: *mut EF,
        headers: *const MonomialHeader,
        terms: *const InteractionMonomialTerm<F>,
        beta_pows: *const EF,
        eq_3bs: *const EF,
        num_monomials: u32,
    ) -> i32;

    fn _kb_logup_monomial_batched(
        tmp_sums: *mut Frac<EF>,
        output: *mut Frac<EF>,
        block_ctxs: *const BlockCtx,
        common_ctxs: *const LogupMonomialCommonCtx,
        numer_ctxs: *const LogupMonomialCtx,
        denom_ctxs: *const LogupMonomialCtx,
        air_block_offsets: *const u32,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> i32;
}

pub unsafe fn interpolate_columns_gpu_kb(
    interpolated: &DeviceBuffer<EF>,
    columns: &DeviceBuffer<*const EF>,
    s_deg: usize,
    num_y: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_interpolate_columns(
        interpolated.as_mut_ptr(),
        columns.as_ptr(),
        s_deg,
        num_y,
        columns.len(),
    ))
}

/// KB-specific bit-reversal + tree-build kernel.
/// Uses KB field arithmetic (FracExtKb with kb31_4_t), unlike the BB fallback.
pub unsafe fn gkr_bit_rev_frac_ext_build_k2_kb(
    inout: &DeviceBuffer<(EF, EF)>,
    lg_domain_size: u32,
    alpha: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_gkr_bit_rev_frac_ext_build_k2(
        inout.as_mut_raw_ptr(),
        lg_domain_size,
        alpha,
    ))
}

pub unsafe fn frac_build_tree_layer_kb(
    layer: &mut DeviceBuffer<Frac<EF>>,
    layer_size: usize,
    revert: bool,
    alpha: EF,
    apply_alpha: bool,
) -> Result<(), CudaError> {
    debug_assert!(layer.len() >= layer_size);
    CudaError::from_result(_kb_frac_build_tree_layer(
        layer.as_mut_ptr(),
        layer_size,
        revert,
        alpha,
        apply_alpha,
    ))
}

/// Fused two-layer tree build kernel.
/// `half_i1` = N >> (i+2), where i is the first of the two layers being fused.
pub unsafe fn frac_build_tree_two_layers_kb(
    layer: &mut DeviceBuffer<Frac<EF>>,
    half_i1: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_frac_build_tree_two_layers(layer.as_mut_ptr(), half_i1))
}

// `eq_xi` will not store evaluations for the first hypercube coordinate because the prover factors
// out the first eq term.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_kb(
    eq_xi: &SqrtEqLayers,
    pq_buffer: &DeviceBuffer<Frac<EF>>,
    num_x: usize,
    lambda: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    debug_assert!(pq_buffer.len() >= 2 * num_x);
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _kb_frac_compute_round_temp_buffer_size(num_x.try_into().unwrap());
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_kb_frac_compute_round(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        pq_buffer.as_ptr(),
        num_x,
        1 << low_n,
        lambda,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
    ))
}

/// Fused compute round + tree layer revert kernel.
///
/// Combines `frac_build_tree_layer(revert=true)` with `compute_round` for the first inner round.
/// The revert operation modifies `layer` in-place: `layer[i] = layer[i] - layer[i + half]` for `i <
/// half`.
///
/// This eliminates one kernel launch per outer round.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_revert_kb(
    eq_xi: &SqrtEqLayers,
    layer: &mut DeviceBuffer<Frac<EF>>,
    num_x: usize,
    lambda: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    #[cfg(debug_assertions)]
    {
        let len = tmp_block_sums.len();
        let required = _kb_frac_compute_round_temp_buffer_size(num_x.try_into().unwrap());
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
        assert!(layer.len() >= 2 * num_x, "layer too small for pq_size");
    }
    CudaError::from_result(_kb_frac_compute_round_and_revert(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        layer.as_mut_ptr(),
        num_x,
        1 << low_n,
        lambda,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
    ))
}

/// Folds `Frac<EF>` buffer. Pairs (idx, idx+quarter) and (idx+half, idx+3*quarter),
/// writes results to dst[idx] and dst[idx+quarter]. Output size is `size / 2`.
/// Safe for src == dst (in-place) because each thread handles disjoint indices.
pub unsafe fn fold_ef_frac_columns_kb(
    src: &DeviceBuffer<Frac<EF>>,
    dst: &mut DeviceBuffer<Frac<EF>>,
    size: usize,
    r: EF,
) -> Result<(), CudaError> {
    debug_assert!(src.len() >= size);
    debug_assert!(dst.len() >= size / 2);
    CudaError::from_result(_kb_frac_fold_fpext_columns(
        src.as_ptr(),
        dst.as_mut_ptr(),
        size,
        r,
    ))
}

/// Raw-pointer variant of [`fold_ef_frac_columns_kb`] for use when `dst` is a raw device pointer.
pub unsafe fn fold_ef_frac_columns_raw_kb(
    src: &DeviceBuffer<Frac<EF>>,
    dst: *mut Frac<EF>,
    size: usize,
    r: EF,
) -> Result<(), CudaError> {
    debug_assert!(src.len() >= size);
    CudaError::from_result(_kb_frac_fold_fpext_columns(src.as_ptr(), dst, size, r))
}

/// In-place fold. See [`fold_ef_frac_columns_kb`] for details.
pub unsafe fn fold_ef_frac_columns_inplace_kb(
    buffer: &mut DeviceBuffer<Frac<EF>>,
    size: usize,
    r: EF,
) -> Result<(), CudaError> {
    debug_assert!(buffer.len() >= size);
    let ptr = buffer.as_mut_ptr();
    CudaError::from_result(_kb_frac_fold_fpext_columns(ptr, ptr, size, r))
}

/// Fused compute round + fold kernel.
///
/// Reads from pre-fold `src_pq_buffer` (size `src_pq_size`), performs fold-on-the-fly using
/// `r_prev`, computes s' polynomial evaluations (degree 2), and writes folded output to
/// `dst_pq_buffer` (size `src_pq_size/2`).
///
/// This fuses the fold operation into the next round's compute, eliminating one kernel launch per
/// inner round and reducing memory traffic.
///
/// The eq_xi layers should have max_n = log2(src_pq_size / 4) = log2(post-fold num_x).
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold_kb(
    eq_xi: &SqrtEqLayers,
    src_pq_buffer: &DeviceBuffer<Frac<EF>>,
    dst_pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    // Post-fold: num_x = src_pq_size / 4
    let num_x = src_pq_size >> 2;
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    #[cfg(debug_assertions)]
    {
        assert!(src_pq_size > 2, "src_pq_size must be > 2");
        let pq_size = src_pq_size >> 1;
        assert!(num_x > 0, "num_x must be > 0");
        assert!(
            src_pq_buffer.len() >= src_pq_size,
            "src_pq_buffer too small: {} < {}",
            src_pq_buffer.len(),
            src_pq_size
        );
        assert!(
            dst_pq_buffer.len() >= pq_size,
            "dst_pq_buffer too small: {} < {}",
            dst_pq_buffer.len(),
            pq_size
        );
        let len = tmp_block_sums.len();
        let required = _kb_frac_compute_round_temp_buffer_size(num_x as u32);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_kb_frac_compute_round_and_fold(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        src_pq_buffer.as_ptr(),
        dst_pq_buffer.as_mut_ptr(),
        src_pq_size,
        1 << low_n,
        lambda,
        r_prev,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
    ))
}

/// Raw-pointer variant of [`frac_compute_round_and_fold_kb`] — `dst_pq` is a raw device pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold_raw_kb(
    eq_xi_low: *const EF,
    eq_xi_high: *const EF,
    src_pq: *const Frac<EF>,
    dst_pq: *mut Frac<EF>,
    src_pq_size: usize,
    eq_low_cap: usize,
    lambda: EF,
    r_prev: EF,
    out: *mut EF,
    tmp: *mut EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_frac_compute_round_and_fold(
        eq_xi_low,
        eq_xi_high,
        src_pq,
        dst_pq,
        src_pq_size,
        eq_low_cap,
        lambda,
        r_prev,
        out,
        tmp,
    ))
}

/// In-place fused compute round + fold kernel. See [`frac_compute_round_and_fold_kb`] for details.
///
/// Uses a dedicated in-place kernel that doesn't have `__restrict__` on the pq pointer,
/// avoiding undefined behavior from aliased restrict pointers.
///
/// **IN-PLACE SAFETY:** Each thread writes only to indices it reads from in the first half of the
/// buffer, so there are no cross-thread conflicts. The second half is read-only.
#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_compute_round_and_fold_inplace_kb(
    eq_xi: &SqrtEqLayers,
    pq_buffer: &mut DeviceBuffer<Frac<EF>>,
    src_pq_size: usize,
    lambda: EF,
    r_prev: EF,
    out_device: &mut DeviceBuffer<EF>,
    tmp_block_sums: &mut DeviceBuffer<EF>,
) -> Result<(), CudaError> {
    let low_n = eq_xi.low_n();
    let high_n = eq_xi.high_n();
    // Post-fold: num_x = src_pq_size / 4
    let num_x = src_pq_size >> 2;
    debug_assert_eq!(2 << (low_n + high_n), num_x);
    #[cfg(debug_assertions)]
    {
        assert!(src_pq_size > 2, "src_pq_size must be > 2");
        assert!(num_x > 0, "num_x must be > 0");
        assert!(
            pq_buffer.len() >= src_pq_size,
            "pq_buffer too small: {} < {}",
            pq_buffer.len(),
            src_pq_size
        );
        let len = tmp_block_sums.len();
        let required = _kb_frac_compute_round_temp_buffer_size(num_x as u32);
        assert!(
            len >= required as usize,
            "tmp_block_sums len={len} < required={required}"
        );
    }
    CudaError::from_result(_kb_frac_compute_round_and_fold_inplace(
        eq_xi.low.get_ptr(low_n),
        eq_xi.high.get_ptr(high_n),
        pq_buffer.as_mut_ptr(),
        src_pq_size,
        1 << low_n,
        lambda,
        r_prev,
        out_device.as_mut_ptr(),
        tmp_block_sums.as_mut_ptr(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_precompute_m_build_raw_kb(
    pq: *const Frac<EF>,
    rem_n: usize,
    w: usize,
    lambda: EF,
    r_prev: EF,
    inline_fold: bool,
    eq_tail_low: *const EF,
    eq_tail_high: *const EF,
    eq_tail_low_cap: usize,
    tail_tile: usize,
    partial_out: *mut EF,
    partial_len: usize,
    m_total: *mut EF,
) -> Result<(), CudaError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    debug_assert!(eq_tail_low_cap.is_power_of_two());
    debug_assert!(tail_tile > 0);
    CudaError::from_result(_kb_frac_precompute_m_build(
        pq,
        rem_n,
        w,
        lambda,
        r_prev,
        inline_fold,
        eq_tail_low,
        eq_tail_high,
        eq_tail_low_cap,
        tail_tile,
        partial_out,
        partial_len,
        m_total,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn frac_precompute_m_eval_round_raw_kb(
    m_total: *const EF,
    w: usize,
    t: usize,
    eq_r_prefix: *const EF,
    eq_suffix: *const EF,
    out: *mut EF,
) -> Result<(), CudaError> {
    debug_assert!(w > 0);
    debug_assert!(t < w);
    CudaError::from_result(_kb_frac_precompute_m_eval_round(
        m_total,
        w,
        t,
        eq_r_prefix,
        eq_suffix,
        out,
    ))
}

pub unsafe fn frac_multifold_raw_kb(
    src: *const Frac<EF>,
    dst: *mut Frac<EF>,
    rem_n: usize,
    w: usize,
    eq_r_window: *const EF,
) -> Result<(), CudaError> {
    debug_assert!(rem_n > 0);
    debug_assert!(w > 0 && w <= rem_n);
    CudaError::from_result(_kb_frac_multifold(src, dst, rem_n, w, eq_r_window))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn fold_ple_from_evals_kb(
    input_matrix: &DeviceBuffer<F>,
    output_matrix: *mut EF,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<EF>,
    height: u32,
    width: u32,
    l_skip: u32,
    new_height: u32,
    rotate: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_fold_ple_from_evals(
        input_matrix.as_ptr(),
        output_matrix,
        omega_skip_pows.as_ptr(),
        inv_lagrange_denoms.as_ptr(),
        height,
        width,
        l_skip,
        new_height,
        rotate,
    ))
}

/// # Safety
/// - `fracs` must be a pointer to a device buffer with capacity at least `num_interactions *
///   height`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_gkr_input_eval_kb(
    is_global: bool,
    fracs: *mut Frac<EF>,
    preprocessed: &DeviceBuffer<F>,
    partitioned_main: &DeviceBuffer<u64>,
    public_values: &DeviceBuffer<F>,
    challenges: &DeviceBuffer<EF>,
    intermediates: &DeviceBuffer<EF>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    pair_idxs: &DeviceBuffer<u32>,
    height: u32,
    num_rows_per_tile: u32,
) -> Result<(), CudaError> {
    debug_assert_eq!(used_nodes.len(), pair_idxs.len());
    CudaError::from_result(_kb_logup_gkr_input_eval(
        is_global,
        fracs,
        preprocessed.as_ptr(),
        partitioned_main.as_ptr(),
        public_values.as_ptr(),
        challenges.as_ptr(),
        intermediates.as_ptr(),
        rules.as_raw_ptr(),
        used_nodes.as_ptr(),
        pair_idxs.as_ptr(),
        used_nodes.len(),
        height,
        num_rows_per_tile,
    ))
}

pub unsafe fn frac_add_alpha_kb(
    data: &DeviceBuffer<Frac<EF>>,
    alpha: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_frac_add_alpha(data.as_mut_raw_ptr(), data.len(), alpha))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
/// - `eq_cube` must be a pointer to device buffer with at least `num_x` elements representing
///   evaluations on hypercube.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_ntt_eval_constraints_kb(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    selectors_cube: &DeviceBuffer<F>,
    preprocessed: *const F,
    main_ptrs: &DeviceBuffer<*const F>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    public_values: &DeviceBuffer<F>,
    rules: &DeviceBuffer<u128>,
    used_nodes: &DeviceBuffer<usize>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_zerocheck_ntt_eval_constraints(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        selectors_cube.as_ptr(),
        preprocessed,
        main_ptrs.as_ptr(),
        eq_cube,
        lambda_pows.as_ptr(),
        public_values.as_ptr(),
        rules.as_raw_ptr(),
        rules.len(),
        used_nodes.as_ptr(),
        used_nodes.len(),
        lambda_pows.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        skip_domain,
        num_x,
        height,
        num_cosets,
        g_shift,
        max_temp_bytes,
    ))
}

/// # Safety
/// - `buffer_size` does not refer to the capacity of `intermediates`. It refers to "how many DAG
///   nodes per row need to be buffered". The capacity is a multiple of `buffer_size` which is
///   runtime calculated based on `buffer_size`.
/// - `eq_cube` must be a pointer to device buffer with at least `num_x` elements representing
///   evaluations on hypercube.
/// - `output` will not be written to by this function. Only `tmp_sums_buffer` is written.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_bary_eval_interactions_round0_kb(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    selectors_cube: &DeviceBuffer<F>,
    preprocessed: *const F,
    main_ptrs: &DeviceBuffer<*const F>,
    eq_cube: *const EF,
    public_values: &DeviceBuffer<F>,
    numer_weights: &DeviceBuffer<EF>,
    denom_weights: &DeviceBuffer<EF>,
    denom_sum_init: EF,
    rules: &DeviceBuffer<u128>,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<F>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_logup_bary_eval_interactions_round0(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        selectors_cube.as_ptr(),
        preprocessed,
        main_ptrs.as_ptr(),
        eq_cube,
        public_values.as_ptr(),
        numer_weights.as_ptr(),
        denom_weights.as_ptr(),
        denom_sum_init,
        rules.as_raw_ptr(),
        rules.len(),
        buffer_size,
        intermediates.as_mut_ptr(),
        skip_domain,
        num_x,
        height,
        num_cosets,
        g_shift,
        max_temp_bytes,
    ))
}

/// Evaluate zerocheck MLE constraints on GPU with raw device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_eval_mle_kb(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    eq_xi: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: *const MainMatrixPtrs<EF>,
    lambda_pows: *const EF,
    lambda_len: usize,
    public_values: *const F,
    rules: *const std::ffi::c_void,
    rules_len: usize,
    used_nodes: *const usize,
    used_nodes_len: usize,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<EF>,
    num_y: u32,
    num_x: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_zerocheck_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        eq_xi,
        selectors,
        preprocessed,
        main_ptrs,
        lambda_pows,
        public_values,
        rules,
        rules_len,
        used_nodes,
        used_nodes_len,
        lambda_len,
        buffer_size,
        intermediates.as_mut_ptr(),
        num_y,
        num_x,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_batch_eval_mle_kb(
    tmp_sums_buffer: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    zc_ctxs: &DeviceBuffer<ZerocheckCtx>,
    air_block_offsets: &DeviceBuffer<u32>,
    lambda_pows: &DeviceBuffer<EF>,
    lambda_len: usize,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_zerocheck_batch_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        zc_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        lambda_pows.as_ptr(),
        lambda_len,
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
    ))
}

/// Evaluate logup MLE interactions on GPU with raw device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_eval_mle_kb(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    eq_xi: *const EF,
    selectors: *const EF,
    preprocessed: MainMatrixPtrs<EF>,
    main_ptrs: *const MainMatrixPtrs<EF>,
    challenges: *const EF,
    eq_3bs: *const EF,
    public_values: *const F,
    rules: *const std::ffi::c_void,
    used_nodes: *const usize,
    pair_idxs: *const u32,
    used_nodes_len: usize,
    buffer_size: u32,
    intermediates: &mut DeviceBuffer<EF>,
    num_y: u32,
    num_x: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_logup_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        eq_xi,
        selectors,
        preprocessed,
        main_ptrs,
        challenges,
        eq_3bs,
        public_values,
        rules,
        used_nodes,
        pair_idxs,
        used_nodes_len,
        buffer_size,
        intermediates.as_mut_ptr(),
        num_y,
        num_x,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_batch_eval_mle_kb(
    tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    logup_ctxs: &DeviceBuffer<LogupCtx>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_logup_batch_eval_mle(
        tmp_sums_buffer.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        logup_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_monomial_batched_kb(
    tmp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<MonomialAirCtx>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_zerocheck_monomial_batched(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        air_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn zerocheck_monomial_par_y_batched_kb(
    tmp_sums: &mut DeviceBuffer<EF>,
    output: &mut DeviceBuffer<EF>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    air_ctxs: &DeviceBuffer<MonomialAirCtx>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    chunk_size: u32,
    threads_per_block: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_zerocheck_monomial_par_y_batched(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        air_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        chunk_size,
        threads_per_block,
    ))
}

pub unsafe fn precompute_lambda_combinations_kb(
    out: &mut DeviceBuffer<EF>,
    headers: *const MonomialHeader,
    lambda_terms: *const LambdaTerm<F>,
    lambda_pows: &DeviceBuffer<EF>,
    num_monomials: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_precompute_lambda_combinations(
        out.as_mut_ptr(),
        headers,
        lambda_terms,
        lambda_pows.as_ptr(),
        num_monomials,
    ))
}

pub unsafe fn precompute_logup_numer_combinations_kb(
    out: &mut DeviceBuffer<EF>,
    headers: *const MonomialHeader,
    terms: *const InteractionMonomialTerm<F>,
    eq_3bs: &DeviceBuffer<EF>,
    num_monomials: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_precompute_logup_numer_combinations(
        out.as_mut_ptr(),
        headers,
        terms,
        eq_3bs.as_ptr(),
        num_monomials,
    ))
}

pub unsafe fn precompute_logup_denom_combinations_kb(
    out: &mut DeviceBuffer<EF>,
    headers: *const MonomialHeader,
    terms: *const InteractionMonomialTerm<F>,
    beta_pows: &DeviceBuffer<EF>,
    eq_3bs: &DeviceBuffer<EF>,
    num_monomials: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_precompute_logup_denom_combinations(
        out.as_mut_ptr(),
        headers,
        terms,
        beta_pows.as_ptr(),
        eq_3bs.as_ptr(),
        num_monomials,
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn logup_monomial_batched_kb(
    tmp_sums: &mut DeviceBuffer<Frac<EF>>,
    output: &mut DeviceBuffer<Frac<EF>>,
    block_ctxs: &DeviceBuffer<BlockCtx>,
    common_ctxs: &DeviceBuffer<LogupMonomialCommonCtx>,
    numer_ctxs: &DeviceBuffer<LogupMonomialCtx>,
    denom_ctxs: &DeviceBuffer<LogupMonomialCtx>,
    air_block_offsets: &DeviceBuffer<u32>,
    num_blocks: u32,
    num_x: u32,
    num_airs: u32,
    threads_per_block: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_logup_monomial_batched(
        tmp_sums.as_mut_ptr(),
        output.as_mut_ptr(),
        block_ctxs.as_ptr(),
        common_ctxs.as_ptr(),
        numer_ctxs.as_ptr(),
        denom_ctxs.as_ptr(),
        air_block_offsets.as_ptr(),
        num_blocks,
        num_x,
        num_airs,
        threads_per_block,
    ))
}

pub unsafe fn frac_vector_scalar_multiply_ext_fp_kb(
    frac_vec: *mut Frac<EF>,
    scalar: F,
    length: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_frac_vector_scalar_multiply_ext_fp(
        frac_vec, scalar, length,
    ))
}

/// Vertically repeats the rows of `input` and writes them to `out`. Both matrices are column-major.
///
/// # Safety
/// - `out` must be a pointer to `DeviceBuffer<F>` with length at least `lifted_height * width`.
/// - `input` must be a pointer to `DeviceBuffer<F>` with length at least `height * width`.
/// - `out` and `input` must not overlap.
pub unsafe fn frac_matrix_vertically_repeat_kb(
    out: *mut Frac<EF>,
    input: *const Frac<EF>,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(lifted_height > height);
    CudaError::from_result(_kb_frac_matrix_vertically_repeat(
        out,
        input,
        width,
        lifted_height,
        height,
    ))
}

pub unsafe fn frac_matrix_vertically_repeat_ext_kb(
    out_numerators: *mut EF,
    out_denominators: *mut EF,
    in_numerators: *const EF,
    in_denominators: *const EF,
    width: u32,
    lifted_height: u32,
    height: u32,
) -> Result<(), CudaError> {
    debug_assert!(lifted_height > height);
    CudaError::from_result(_kb_frac_matrix_vertically_repeat_ext(
        out_numerators,
        out_denominators,
        in_numerators,
        in_denominators,
        width,
        lifted_height,
        height,
    ))
}

/// Create folded selectors around round 0 from hypercube evaluations and univariate factors.
///
/// Note: `is_transition` is not a product of univariate and hypercube factors.
pub unsafe fn fold_selectors_round0_kb(
    out: *mut EF,
    input: *const F,
    is_first: EF,
    is_last: EF,
    num_x: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_fold_selectors_round0(
        out,
        input,
        is_first,
        is_last,
        num_x as u32,
    ))
}
