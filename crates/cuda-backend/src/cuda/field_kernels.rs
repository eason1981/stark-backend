//! `FieldKernels` — dispatch trait for field-specific GPU proving kernels.
//!
//! Separates "which CUDA library to call" from the proving business logic,
//! enabling the same Rust proving code to run over BabyBear (linking to
//! `cuda-backend`) or KoalaBear (linking to `cuda_kb_all`).
//!
//! # Zero-cost
//! The trait is monomorphised at compile time; no virtual dispatch.

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use openvm_stark_backend::{poly_common::Squarable, prover::fractional_sumcheck_gkr::Frac};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField32, TwoAdicField};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt;

use crate::{
    cuda::logup_zerocheck::MainMatrixPtrs,
    ntt_field::GpuNttField,
    poly::{EqEvalLayers, EqEvalSegments, SqrtEqLayersFor},
    stacked_reduction::UnstackedSlice,
    whir::BatchingTracePacket,
};

/// Dispatch trait for all field-specific GPU proving kernels used by the
/// WHIR and stacked-reduction proving pipelines.
///
/// `Val` is the base field (e.g. BabyBear or KoalaBear) and `ValExt` is the
/// degree-4 extension used throughout the proof system.
pub trait FieldKernels: 'static + Copy + Send + Sync {
    /// If true, use the CPU path for the GKR fractional sumcheck instead of GPU.
    /// Default: false (use GPU). Override to true for fields where the GPU GKR
    /// has correctness issues (e.g. KoalaBear at round 7+).
    fn use_cpu_gkr() -> bool {
        false
    }
    /// If true, compute the GKR logup input layer (interaction frac leaves) on CPU instead
    /// of GPU. Needed for fields where the GPU `logup_gkr_input_eval` kernel is wrong for
    /// multi-partition AIRs (e.g. KoalaBear with ByteChip/ProgramChip cached+common mains).
    /// Independent of `use_cpu_gkr` so the GPU sumcheck can run on correct CPU-computed leaves.
    fn use_cpu_gkr_input_eval() -> bool {
        false
    }
    /// If true, use the CPU fallback for the per-round logup/zerocheck MLE evaluations, round0
    /// batch-MLE, and fold steps (sumcheck_polys_batch_eval, round0 constraints/interactions,
    /// mat_evals fold). Decoupled from `use_cpu_gkr_input_eval` so the GPU input-eval can be
    /// enabled independently of these (separately-suspect) GPU MLE/fold paths.
    /// Default: false (use GPU).
    fn use_cpu_zerocheck_mle() -> bool {
        false
    }

    // ── Monomial-basis batch MLE kernels (default = BabyBear; KoalaBear overrides) ──
    // These were previously called as hardcoded BabyBear free functions in batch_mle_monomial.rs,
    // which produced wrong results for KoalaBear (BB arithmetic on KB data). Dispatch via the
    // trait so KB uses its _kb kernels.
    /// Zerocheck monomial batched MLE (low num_y path).
    #[allow(clippy::too_many_arguments)]
    unsafe fn zerocheck_monomial_batched(
        tmp_sums: &mut DeviceBuffer<EF>,
        output: &mut DeviceBuffer<EF>,
        block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
        air_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::MonomialAirCtx>,
        air_block_offsets: &DeviceBuffer<u32>,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::zerocheck_monomial_batched(
            tmp_sums, output, block_ctxs, air_ctxs, air_block_offsets, num_blocks, num_x, num_airs,
            threads_per_block,
        )
    }
    /// Zerocheck monomial par-Y batched MLE (high num_y path).
    #[allow(clippy::too_many_arguments)]
    unsafe fn zerocheck_monomial_par_y_batched(
        tmp_sums: &mut DeviceBuffer<EF>,
        output: &mut DeviceBuffer<EF>,
        block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
        air_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::MonomialAirCtx>,
        air_block_offsets: &DeviceBuffer<u32>,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        chunk_size: u32,
        threads_per_block: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::zerocheck_monomial_par_y_batched(
            tmp_sums, output, block_ctxs, air_ctxs, air_block_offsets, num_blocks, num_x, num_airs,
            chunk_size, threads_per_block,
        )
    }
    /// Logup monomial batched MLE.
    #[allow(clippy::too_many_arguments)]
    unsafe fn logup_monomial_batched(
        tmp_sums: &mut DeviceBuffer<Frac<EF>>,
        output: &mut DeviceBuffer<Frac<EF>>,
        block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
        common_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupMonomialCommonCtx>,
        numer_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupMonomialCtx>,
        denom_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupMonomialCtx>,
        air_block_offsets: &DeviceBuffer<u32>,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::logup_monomial_batched(
            tmp_sums, output, block_ctxs, common_ctxs, numer_ctxs, denom_ctxs, air_block_offsets,
            num_blocks, num_x, num_airs, threads_per_block,
        )
    }
    /// Zerocheck batched DAG MLE (high monomial-to-rules ratio path).
    #[allow(clippy::too_many_arguments)]
    unsafe fn zerocheck_batch_eval_mle(
        tmp_sums_buffer: &mut DeviceBuffer<EF>,
        output: &mut DeviceBuffer<EF>,
        block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
        zc_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::ZerocheckCtx>,
        air_block_offsets: &DeviceBuffer<u32>,
        lambda_pows: &DeviceBuffer<EF>,
        lambda_len: usize,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::zerocheck_batch_eval_mle(
            tmp_sums_buffer, output, block_ctxs, zc_ctxs, air_block_offsets, lambda_pows,
            lambda_len, num_blocks, num_x, num_airs, threads_per_block,
        )
    }
    /// Logup batched DAG MLE.
    #[allow(clippy::too_many_arguments)]
    unsafe fn logup_batch_eval_mle(
        tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>,
        output: &mut DeviceBuffer<Frac<EF>>,
        block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
        logup_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupCtx>,
        air_block_offsets: &DeviceBuffer<u32>,
        num_blocks: u32,
        num_x: u32,
        num_airs: u32,
        threads_per_block: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::logup_batch_eval_mle(
            tmp_sums_buffer, output, block_ctxs, logup_ctxs, air_block_offsets, num_blocks, num_x,
            num_airs, threads_per_block,
        )
    }

    type Val: GpuNttField
        + PrimeField32
        + TwoAdicField
        + Field
        + Copy
        + Clone
        + Send
        + Sync
        + 'static;
    type ValExt: Field
        + TwoAdicField
        + ExtensionField<Self::Val>
        + BasedVectorSpace<Self::Val>
        + Squarable
        + Copy
        + Clone
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + fmt::Debug
        + fmt::Display
        + 'static;

    // ─── stacked-reduction kernels ────────────────────────────────────────

    unsafe fn stacked_r0_temp_buf_size(trace_height: u32, trace_width: u32, l_skip: u32) -> u32;

    #[allow(clippy::too_many_arguments)]
    unsafe fn stacked_sumcheck_round0(
        eq_r_ns: &EqEvalSegments<Self::ValExt>,
        trace_ptr: *const Self::Val,
        lambda_pows: *const Self::ValExt,
        block_sums: &mut DeviceBuffer<Self::ValExt>,
        output: &mut DeviceBuffer<Self::ValExt>,
        height: usize,
        width: usize,
        l_skip: usize,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn stacked_fold_ple(
        src: *const Self::Val,
        dst: *mut Self::ValExt,
        omega_skip_pows: &DeviceBuffer<Self::Val>,
        inv_lagrange_denoms: &DeviceBuffer<Self::ValExt>,
        trace_height: usize,
        trace_width: usize,
        l_skip: usize,
    ) -> Result<(), CudaError>;

    unsafe fn init_k_rot_from_eq_segments(
        eq_r_ns: &EqEvalSegments<Self::ValExt>,
        k_rot_ns: &mut DeviceBuffer<Self::ValExt>,
        k_rot_uni_0: Self::ValExt,
        k_rot_uni_1: Self::ValExt,
        max_n: u32,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn stacked_sumcheck_mle_round(
        q_evals: &DeviceBuffer<*const Self::ValExt>,
        eq_r_ns: &EqEvalSegments<Self::ValExt>,
        k_rot_ns: &EqEvalSegments<Self::ValExt>,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const Self::ValExt,
        output: &mut DeviceBuffer<u64>,
        q_height: usize,
        window_len: usize,
        num_y: usize,
        sm_count: u32,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn stacked_sumcheck_mle_round_degenerate(
        q_evals: &DeviceBuffer<*const Self::ValExt>,
        eq_ub_ptr: &DeviceBuffer<Self::ValExt>,
        eq_r: Self::ValExt,
        k_rot_r: Self::ValExt,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const Self::ValExt,
        output: &mut DeviceBuffer<u64>,
        q_height: usize,
        window_len: usize,
        l_skip: usize,
        round: usize,
    ) -> Result<(), CudaError>;

    // ─── shared poly/sumcheck kernels ─────────────────────────────────────

    fn vector_scalar_multiply_ext(
        vec: &mut DeviceBuffer<Self::ValExt>,
        scalar: Self::ValExt,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn fold_mle(
        input_matrices: &DeviceBuffer<*const Self::ValExt>,
        output_matrices: &DeviceBuffer<*mut Self::ValExt>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u16,
        output_height: u32,
        max_output_cells: u32,
        r_val: Self::ValExt,
    ) -> Result<(), CudaError>;

    // ─── logup/zerocheck kernels ──────────────────────────────────────────

    /// Fold base-field selector columns (is_transition/is_first/is_last) after the
    /// univariate skip round, producing extension-field outputs.
    unsafe fn fold_selectors_round0(
        out: *mut Self::ValExt,
        input: *const Self::Val,
        is_first: Self::ValExt,
        is_last: Self::ValExt,
        num_x: usize,
    ) -> Result<(), CudaError>;

    /// Interpolate columns of a matrix to produce `s_deg` extra rows used in
    /// subsequent sumcheck rounds.
    unsafe fn interpolate_columns_gpu(
        interpolated: &DeviceBuffer<Self::ValExt>,
        columns: &DeviceBuffer<*const Self::ValExt>,
        s_deg: usize,
        num_y: usize,
    ) -> Result<(), CudaError>;

    /// Fold multiple MLE matrices simultaneously, each potentially at a different
    /// height (supplied via `log_output_heights`).
    #[allow(clippy::too_many_arguments)]
    unsafe fn batch_fold_mle(
        input_matrices: &DeviceBuffer<*const Self::ValExt>,
        output_matrices: &DeviceBuffer<*mut Self::ValExt>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u16,
        log_output_heights: &DeviceBuffer<u8>,
        max_output_cells: u32,
        r_val: Self::ValExt,
    ) -> Result<(), CudaError>;

    unsafe fn triangular_fold_mle(
        output: &mut EqEvalSegments<Self::ValExt>,
        input: &EqEvalSegments<Self::ValExt>,
        r: Self::ValExt,
        output_max_n: usize,
    ) -> Result<(), CudaError>;

    // ─── WHIR kernels ─────────────────────────────────────────────────────

    unsafe fn whir_algebraic_batch_traces(
        output: &mut DeviceBuffer<Self::Val>,
        packets: &DeviceBuffer<BatchingTracePacket<Self::Val>>,
        mu_powers: &DeviceBuffer<Self::ValExt>,
        skip_domain: u32,
    ) -> Result<(), CudaError>;

    fn whir_sumcheck_coeff_moments_required_temp_buffer_size(height: u32) -> u32;

    unsafe fn whir_sumcheck_coeff_moments_round(
        f_coeffs: &DeviceBuffer<Self::ValExt>,
        w_moments: &DeviceBuffer<Self::ValExt>,
        output: &mut DeviceBuffer<Self::ValExt>,
        tmp_block_sums: &mut DeviceBuffer<Self::ValExt>,
        height: u32,
    ) -> Result<(), CudaError>;

    unsafe fn whir_fold_coeffs_and_moments(
        f_coeffs: &DeviceBuffer<Self::ValExt>,
        w_moments: &DeviceBuffer<Self::ValExt>,
        f_folded: &mut DeviceBuffer<Self::ValExt>,
        w_folded: &mut DeviceBuffer<Self::ValExt>,
        alpha: Self::ValExt,
        height: u32,
    ) -> Result<(), CudaError>;

    unsafe fn w_moments_accumulate(
        w_moments: &mut DeviceBuffer<Self::ValExt>,
        z0_pows2: &DeviceBuffer<Self::ValExt>,
        z_pows2: &DeviceBuffer<Self::Val>,
        gamma: Self::ValExt,
        num_queries: u32,
        log_height: u32,
    ) -> Result<(), CudaError>;

    // ─── matrix/mle/poly kernels used by WHIR ─────────────────────────────

    unsafe fn batch_expand_pad(
        output: *mut Self::Val,
        input: *const Self::Val,
        poly_count: u32,
        out_size: u32,
        in_size: u32,
    ) -> Result<(), CudaError>;

    unsafe fn split_ext_to_base_col_major_matrix(
        d_matrix: &mut DeviceBuffer<Self::Val>,
        d_poly: &DeviceBuffer<Self::ValExt>,
        poly_len: u64,
        matrix_height: u32,
    ) -> Result<(), CudaError>;

    unsafe fn mle_interpolate_stage_ext(
        buffer: &mut DeviceBuffer<Self::ValExt>,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> Result<(), CudaError>;

    unsafe fn eval_poly_ext_at_point_from_base(
        base_coeffs: &DeviceBuffer<Self::Val>,
        coeff_len: usize,
        x: Self::ValExt,
    ) -> Result<Self::ValExt, crate::KernelError>;

    unsafe fn transpose_fp_to_fpext_vec(
        output: &mut DeviceBuffer<Self::ValExt>,
        input: &DeviceBuffer<Self::Val>,
    ) -> Result<(), CudaError>;

    // ─── poly stage (used by WHIR evals_eq_hypercube) ─────────────────────

    unsafe fn eq_hypercube_stage_ext(
        out: *mut Self::ValExt,
        x_i: Self::ValExt,
        step: u32,
    ) -> Result<(), CudaError>;

    unsafe fn eq_hypercube_nonoverlapping_stage_ext(
        out: *mut Self::ValExt,
        input: *const Self::ValExt,
        x_i: Self::ValExt,
        step: u32,
    ) -> Result<(), CudaError>;

    unsafe fn eq_hypercube_interleaved_stage_ext(
        out: *mut Self::ValExt,
        input: *const Self::ValExt,
        x_i: Self::ValExt,
        step: u32,
    ) -> Result<(), CudaError>;

    // ─── logup / GKR fractional sumcheck kernels ──────────────────────────────

    unsafe fn frac_compute_round_temp_buffer_size(stride: u32) -> u32;

    unsafe fn frac_build_tree_layer(
        layer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        layer_size: usize,
        revert: bool,
        alpha: Self::ValExt,
        apply_alpha: bool,
    ) -> Result<(), CudaError>;

    unsafe fn frac_build_tree_two_layers(
        layer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        half_i1: usize,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_compute_round(
        eq_xi: &SqrtEqLayersFor<Self::ValExt>,
        pq_buffer: &DeviceBuffer<Frac<Self::ValExt>>,
        num_x: usize,
        lambda: Self::ValExt,
        out_device: &mut DeviceBuffer<Self::ValExt>,
        tmp_block_sums: &mut DeviceBuffer<Self::ValExt>,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_compute_round_and_revert(
        eq_xi: &mut SqrtEqLayersFor<Self::ValExt>,
        layer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        num_x: usize,
        lambda: Self::ValExt,
        out_device: &mut DeviceBuffer<Self::ValExt>,
        tmp_block_sums: &mut DeviceBuffer<Self::ValExt>,
    ) -> Result<(), CudaError>;

    unsafe fn fold_ef_frac_columns(
        src: &DeviceBuffer<Frac<Self::ValExt>>,
        dst: *mut Frac<Self::ValExt>,
        size: usize,
        r: Self::ValExt,
    ) -> Result<(), CudaError>;

    unsafe fn fold_ef_frac_columns_inplace(
        pq_buffer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        size: usize,
        r: Self::ValExt,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_compute_round_and_fold(
        eq_xi: &mut SqrtEqLayersFor<Self::ValExt>,
        src_pq_buffer: &DeviceBuffer<Frac<Self::ValExt>>,
        dst_pq_buffer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        src_pq_size: usize,
        lambda: Self::ValExt,
        r_prev: Self::ValExt,
        out_device: &mut DeviceBuffer<Self::ValExt>,
        tmp_block_sums: &mut DeviceBuffer<Self::ValExt>,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_compute_round_and_fold_inplace(
        eq_xi: &mut SqrtEqLayersFor<Self::ValExt>,
        pq_buffer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        src_pq_size: usize,
        lambda: Self::ValExt,
        r_prev: Self::ValExt,
        out_device: &mut DeviceBuffer<Self::ValExt>,
        tmp_block_sums: &mut DeviceBuffer<Self::ValExt>,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_precompute_m_build_raw(
        pq: *const Frac<Self::ValExt>,
        rem_n: usize,
        w: usize,
        lambda: Self::ValExt,
        r_prev: Self::ValExt,
        inline_fold: bool,
        eq_tail_low: *const Self::ValExt,
        eq_tail_high: *const Self::ValExt,
        eq_tail_low_cap: usize,
        tail_tile: usize,
        partial_out: *mut Self::ValExt,
        partial_len: usize,
        m_total: *mut Self::ValExt,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_precompute_m_eval_round_raw(
        m_total: *const Self::ValExt,
        w: usize,
        t: usize,
        eq_r_prefix: *const Self::ValExt,
        eq_suffix: *const Self::ValExt,
        out: *mut Self::ValExt,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_multifold_raw(
        src: *const Frac<Self::ValExt>,
        dst: *mut Frac<Self::ValExt>,
        rem_n: usize,
        w: usize,
        eq_r_window: *const Self::ValExt,
    ) -> Result<(), CudaError>;

    unsafe fn frac_add_alpha(
        data: &DeviceBuffer<Frac<Self::ValExt>>,
        alpha: Self::ValExt,
    ) -> Result<(), CudaError>;

    unsafe fn frac_vector_scalar_multiply_ext_fp(
        frac_vec: &mut DeviceBuffer<Frac<Self::ValExt>>,
        scalar: Self::Val,
    ) -> Result<(), CudaError>;

    unsafe fn frac_matrix_vertically_repeat(
        out: *mut Frac<Self::ValExt>,
        input: &DeviceBuffer<Frac<Self::ValExt>>,
        width: u32,
        lifted_height: u32,
        height: u32,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn frac_matrix_vertically_repeat_ext(
        out_n: *mut Self::ValExt,
        out_d: *mut Self::ValExt,
        in_n: *const Self::ValExt,
        in_d: *const Self::ValExt,
        width: u32,
        lifted_height: u32,
        height: u32,
    ) -> Result<(), CudaError>;

    unsafe fn bit_rev_frac_ext(
        d_out: &DeviceBuffer<(Self::ValExt, Self::ValExt)>,
        d_inp: &DeviceBuffer<(Self::ValExt, Self::ValExt)>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> Result<(), CudaError>;

    unsafe fn bit_rev_frac_ext_build_k2(
        inout: &DeviceBuffer<(Self::ValExt, Self::ValExt)>,
        lg_domain_size: u32,
        alpha: Self::ValExt,
    ) -> Result<(), CudaError>;

    // ─── logup / zerocheck eval kernels ───────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    unsafe fn fold_ple_from_evals(
        input: &DeviceBuffer<Self::Val>,
        output: *mut Self::ValExt,
        omega_skip_pows: &DeviceBuffer<Self::Val>,
        inv_lagrange_denoms: &DeviceBuffer<Self::ValExt>,
        height: u32,
        width: u32,
        l_skip: u32,
        new_height: u32,
        rotate: bool,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn logup_gkr_input_eval(
        is_global: bool,
        fracs: *mut Frac<Self::ValExt>,
        preprocessed: &DeviceBuffer<Self::Val>,
        partitioned_main: &DeviceBuffer<u64>,
        public_values: &DeviceBuffer<Self::Val>,
        challenges: &DeviceBuffer<Self::ValExt>,
        intermediates: &DeviceBuffer<Self::ValExt>,
        rules: &DeviceBuffer<u128>,
        used_nodes: &DeviceBuffer<usize>,
        pair_idxs: &DeviceBuffer<u32>,
        height: u32,
        num_rows_per_tile: u32,
    ) -> Result<(), CudaError>;

    fn logup_r0_temp_sums_buffer_size(
        buffer_size: u32, skip_domain: u32, num_x: u32,
        num_cosets: u32, max_temp_bytes: usize,
    ) -> usize;

    fn logup_r0_intermediates_buffer_size(
        buffer_size: u32, skip_domain: u32, num_x: u32,
        num_cosets: u32, max_temp_bytes: usize,
    ) -> usize;

    fn zerocheck_r0_temp_sums_buffer_size(
        buffer_size: u32, skip_domain: u32, num_x: u32,
        num_cosets: u32, max_temp_bytes: usize,
    ) -> usize;

    fn zerocheck_r0_intermediates_buffer_size(
        buffer_size: u32, skip_domain: u32, num_x: u32,
        num_cosets: u32, max_temp_bytes: usize,
    ) -> usize;

    fn zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;
    fn zerocheck_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize;
    fn logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize;
    fn logup_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize;
    fn zerocheck_batch_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize;
    fn logup_batch_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize;

    #[allow(clippy::too_many_arguments)]
    unsafe fn zerocheck_ntt_eval_constraints(
        tmp_sums_buffer: &mut DeviceBuffer<Self::ValExt>,
        output: &mut DeviceBuffer<Self::ValExt>,
        selectors_cube: &DeviceBuffer<Self::Val>,
        preprocessed: *const Self::Val,
        main_parts: &DeviceBuffer<*const Self::Val>,
        eq_cube: *const Self::ValExt,
        lambda_pows: &DeviceBuffer<Self::ValExt>,
        public_values: &DeviceBuffer<Self::Val>,
        rules: &DeviceBuffer<u128>,
        used_nodes: &DeviceBuffer<usize>,
        buffer_size: u32,
        intermediates: &mut DeviceBuffer<Self::Val>,
        skip_domain: u32, num_x: u32, height: u32, num_cosets: u32,
        g_shift: Self::Val, max_temp_bytes: usize,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn logup_bary_eval_interactions_round0(
        tmp_sums_buffer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        output: &mut DeviceBuffer<Frac<Self::ValExt>>,
        selectors_cube: &DeviceBuffer<Self::Val>,
        preprocessed: *const Self::Val,
        main_ptrs: &DeviceBuffer<*const Self::Val>,
        eq_cube: *const Self::ValExt,
        public_values: &DeviceBuffer<Self::Val>,
        numer_weights: &DeviceBuffer<Self::ValExt>,
        denom_weights: &DeviceBuffer<Self::ValExt>,
        denom_sum_init: Self::ValExt,
        rules: &DeviceBuffer<u128>,
        buffer_size: u32,
        intermediates: &mut DeviceBuffer<Self::Val>,
        skip_domain: u32, num_x: u32, height: u32, num_cosets: u32,
        g_shift: Self::Val, max_temp_bytes: usize,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn zerocheck_eval_mle(
        tmp_sums_buffer: &mut DeviceBuffer<Self::ValExt>,
        output: &mut DeviceBuffer<Self::ValExt>,
        eq_xi: *const Self::ValExt,
        selectors: *const Self::ValExt,
        preprocessed: MainMatrixPtrs<Self::ValExt>,
        main: *const MainMatrixPtrs<Self::ValExt>,
        lambda_pows: *const Self::ValExt,
        lambda_len: usize,
        public_values: *const Self::Val,
        rules: *const std::ffi::c_void,
        rules_len: usize,
        used_nodes: *const usize,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: &mut DeviceBuffer<Self::ValExt>,
        num_y: u32, num_x: u32,
    ) -> Result<(), CudaError>;

    #[allow(clippy::too_many_arguments)]
    unsafe fn logup_eval_mle(
        tmp_sums_buffer: &mut DeviceBuffer<Frac<Self::ValExt>>,
        output: &mut DeviceBuffer<Frac<Self::ValExt>>,
        eq_xi: *const Self::ValExt,
        selectors: *const Self::ValExt,
        preprocessed: MainMatrixPtrs<Self::ValExt>,
        main: *const MainMatrixPtrs<Self::ValExt>,
        challenges: *const Self::ValExt,
        eq_3bs: *const Self::ValExt,
        public_values: *const Self::Val,
        rules: *const std::ffi::c_void,
        used_nodes: *const usize,
        pair_idxs: *const u32,
        used_nodes_len: usize,
        buffer_size: u32,
        intermediates: &mut DeviceBuffer<Self::ValExt>,
        num_y: u32, num_x: u32,
    ) -> Result<(), CudaError>;

    // ─── monomial precompute kernels ─────────────────────────────────────────

    unsafe fn precompute_lambda_combinations(
        out: &mut DeviceBuffer<Self::ValExt>,
        headers: *const crate::monomial::MonomialHeader,
        lambda_terms: *const crate::monomial::LambdaTerm<Self::Val>,
        lambda_pows: &DeviceBuffer<Self::ValExt>,
        num_monomials: u32,
    ) -> Result<(), CudaError>;

    unsafe fn precompute_logup_numer_combinations(
        out: &mut DeviceBuffer<Self::ValExt>,
        headers: *const crate::monomial::MonomialHeader,
        terms: *const crate::monomial::InteractionMonomialTerm<Self::Val>,
        eq_3bs: &DeviceBuffer<Self::ValExt>,
        num_monomials: u32,
    ) -> Result<(), CudaError>;

    unsafe fn precompute_logup_denom_combinations(
        out: &mut DeviceBuffer<Self::ValExt>,
        headers: *const crate::monomial::MonomialHeader,
        terms: *const crate::monomial::InteractionMonomialTerm<Self::Val>,
        beta_pows: &DeviceBuffer<Self::ValExt>,
        eq_3bs: &DeviceBuffer<Self::ValExt>,
        num_monomials: u32,
    ) -> Result<(), CudaError>;

    /// Populates `out` with `eq(u, y)` for `y` in the boolean hypercube `{0,1}^n`,
    /// where `xs = u`.
    ///
    /// Default implementation uses `eq_hypercube_stage_ext`.
    unsafe fn evals_eq_hypercube(
        out: &mut DeviceBuffer<Self::ValExt>,
        xs: &[Self::ValExt],
    ) -> Result<(), crate::KernelError> {
        use openvm_cuda_common::copy::MemCopyH2D;
        use p3_field::PrimeCharacteristicRing;
        let n = xs.len();
        assert!(out.len() >= 1 << n);
        [Self::ValExt::ONE]
            .copy_to(out)
            .map_err(crate::KernelError::MemCopy)?;
        for (i, &x_i) in xs.iter().enumerate() {
            let step = 1u32 << i;
            Self::eq_hypercube_stage_ext(out.as_mut_ptr(), x_i, step)
                .map_err(crate::KernelError::Kernel)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BabyBear implementation
// ─────────────────────────────────────────────────────────────────────────────

use crate::prelude::{EF, F};

/// BabyBear kernel implementation — delegates to `cuda-backend` static library.
#[derive(Copy, Clone, Debug, Default)]
pub struct BabyBearKernels;

impl FieldKernels for BabyBearKernels {
    type Val = F;
    type ValExt = EF;

    unsafe fn stacked_r0_temp_buf_size(height: u32, width: u32, l_skip: u32) -> u32 {
        crate::cuda::stacked_reduction::_stacked_reduction_r0_required_temp_buffer_size(
            height, width, l_skip,
        )
    }

    unsafe fn stacked_sumcheck_round0(
        eq_r_ns: &EqEvalSegments<EF>,
        trace_ptr: *const F,
        lambda_pows: *const EF,
        block_sums: &mut DeviceBuffer<EF>,
        output: &mut DeviceBuffer<EF>,
        height: usize,
        width: usize,
        l_skip: usize,
    ) -> Result<(), CudaError> {
        crate::cuda::stacked_reduction::stacked_reduction_sumcheck_round0(
            eq_r_ns, trace_ptr, lambda_pows, block_sums, output, height, width, l_skip,
        )
    }

    unsafe fn stacked_fold_ple(
        src: *const F,
        dst: *mut EF,
        omega_skip_pows: &DeviceBuffer<F>,
        inv_lagrange_denoms: &DeviceBuffer<EF>,
        trace_height: usize,
        trace_width: usize,
        l_skip: usize,
    ) -> Result<(), CudaError> {
        crate::cuda::stacked_reduction::stacked_reduction_fold_ple(
            src,
            dst,
            omega_skip_pows,
            inv_lagrange_denoms,
            trace_height,
            trace_width,
            l_skip,
        )
    }

    unsafe fn init_k_rot_from_eq_segments(
        eq_r_ns: &EqEvalSegments<EF>,
        k_rot_ns: &mut DeviceBuffer<EF>,
        k_rot_uni_0: EF,
        k_rot_uni_1: EF,
        max_n: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::stacked_reduction::initialize_k_rot_from_eq_segments(
            eq_r_ns, k_rot_ns, k_rot_uni_0, k_rot_uni_1, max_n,
        )
    }

    unsafe fn stacked_sumcheck_mle_round(
        q_evals: &DeviceBuffer<*const EF>,
        eq_r_ns: &EqEvalSegments<EF>,
        k_rot_ns: &EqEvalSegments<EF>,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        output: &mut DeviceBuffer<u64>,
        q_height: usize,
        window_len: usize,
        num_y: usize,
        sm_count: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::stacked_reduction::stacked_reduction_sumcheck_mle_round(
            q_evals,
            eq_r_ns,
            k_rot_ns,
            unstacked_cols,
            lambda_pows,
            output,
            q_height,
            window_len,
            num_y,
            sm_count,
        )
    }

    unsafe fn stacked_sumcheck_mle_round_degenerate(
        q_evals: &DeviceBuffer<*const EF>,
        eq_ub_ptr: &DeviceBuffer<EF>,
        eq_r: EF,
        k_rot_r: EF,
        unstacked_cols: *const UnstackedSlice,
        lambda_pows: *const EF,
        output: &mut DeviceBuffer<u64>,
        q_height: usize,
        window_len: usize,
        l_skip: usize,
        round: usize,
    ) -> Result<(), CudaError> {
        crate::cuda::stacked_reduction::stacked_reduction_sumcheck_mle_round_degenerate(
            q_evals,
            eq_ub_ptr,
            eq_r,
            k_rot_r,
            unstacked_cols,
            lambda_pows,
            output,
            q_height,
            window_len,
            l_skip,
            round,
        )
    }

    fn vector_scalar_multiply_ext(
        vec: &mut DeviceBuffer<EF>,
        scalar: EF,
    ) -> Result<(), CudaError> {
        crate::cuda::poly::vector_scalar_multiply_ext(vec, scalar)
    }

    unsafe fn fold_mle(
        input_matrices: &DeviceBuffer<*const EF>,
        output_matrices: &DeviceBuffer<*mut EF>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u16,
        output_height: u32,
        max_output_cells: u32,
        r_val: EF,
    ) -> Result<(), CudaError> {
        crate::cuda::sumcheck::fold_mle(
            input_matrices,
            output_matrices,
            widths,
            num_matrices,
            output_height,
            max_output_cells,
            r_val,
        )
    }

    unsafe fn triangular_fold_mle(
        output: &mut EqEvalSegments<EF>,
        input: &EqEvalSegments<EF>,
        r: EF,
        output_max_n: usize,
    ) -> Result<(), CudaError> {
        crate::cuda::sumcheck::triangular_fold_mle(output, input, r, output_max_n)
    }

    unsafe fn fold_selectors_round0(
        out: *mut EF,
        input: *const F,
        is_first: EF,
        is_last: EF,
        num_x: usize,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::fold_selectors_round0(out, input, is_first, is_last, num_x)
    }

    unsafe fn interpolate_columns_gpu(
        interpolated: &DeviceBuffer<EF>,
        columns: &DeviceBuffer<*const EF>,
        s_deg: usize,
        num_y: usize,
    ) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::interpolate_columns_gpu(interpolated, columns, s_deg, num_y)
    }

    unsafe fn batch_fold_mle(
        input_matrices: &DeviceBuffer<*const EF>,
        output_matrices: &DeviceBuffer<*mut EF>,
        widths: &DeviceBuffer<u32>,
        num_matrices: u16,
        log_output_heights: &DeviceBuffer<u8>,
        max_output_cells: u32,
        r_val: EF,
    ) -> Result<(), CudaError> {
        crate::cuda::sumcheck::batch_fold_mle(
            input_matrices,
            output_matrices,
            widths,
            num_matrices,
            log_output_heights,
            max_output_cells,
            r_val,
        )
    }

    unsafe fn whir_algebraic_batch_traces(
        output: &mut DeviceBuffer<F>,
        packets: &DeviceBuffer<BatchingTracePacket<F>>,
        mu_powers: &DeviceBuffer<EF>,
        skip_domain: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::whir::whir_algebraic_batch_traces(output, packets, mu_powers, skip_domain)
    }

    fn whir_sumcheck_coeff_moments_required_temp_buffer_size(height: u32) -> u32 {
        unsafe {
            crate::cuda::whir::_whir_sumcheck_coeff_moments_required_temp_buffer_size(height)
        }
    }

    unsafe fn whir_sumcheck_coeff_moments_round(
        f_coeffs: &DeviceBuffer<EF>,
        w_moments: &DeviceBuffer<EF>,
        output: &mut DeviceBuffer<EF>,
        tmp_block_sums: &mut DeviceBuffer<EF>,
        height: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::whir::whir_sumcheck_coeff_moments_round(
            f_coeffs,
            w_moments,
            output,
            tmp_block_sums,
            height,
        )
    }

    unsafe fn whir_fold_coeffs_and_moments(
        f_coeffs: &DeviceBuffer<EF>,
        w_moments: &DeviceBuffer<EF>,
        f_folded: &mut DeviceBuffer<EF>,
        w_folded: &mut DeviceBuffer<EF>,
        alpha: EF,
        height: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::whir::whir_fold_coeffs_and_moments(
            f_coeffs, w_moments, f_folded, w_folded, alpha, height,
        )
    }

    unsafe fn w_moments_accumulate(
        w_moments: &mut DeviceBuffer<EF>,
        z0_pows2: &DeviceBuffer<EF>,
        z_pows2: &DeviceBuffer<F>,
        gamma: EF,
        num_queries: u32,
        log_height: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::whir::w_moments_accumulate(
            w_moments, z0_pows2, z_pows2, gamma, num_queries, log_height,
        )
    }

    unsafe fn batch_expand_pad(
        output: *mut F,
        input: *const F,
        poly_count: u32,
        out_size: u32,
        in_size: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::matrix::batch_expand_pad(output, input, poly_count, out_size, in_size)
    }

    unsafe fn split_ext_to_base_col_major_matrix(
        d_matrix: &mut DeviceBuffer<F>,
        d_poly: &DeviceBuffer<EF>,
        poly_len: u64,
        matrix_height: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::matrix::split_ext_to_base_col_major_matrix(d_matrix, d_poly, poly_len, matrix_height)
    }

    unsafe fn mle_interpolate_stage_ext(
        buffer: &mut DeviceBuffer<EF>,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> Result<(), CudaError> {
        crate::cuda::mle_interpolate::mle_interpolate_stage_ext(buffer, step, is_eval_to_coeff)
    }

    unsafe fn eval_poly_ext_at_point_from_base(
        base_coeffs: &DeviceBuffer<F>,
        coeff_len: usize,
        x: EF,
    ) -> Result<EF, crate::KernelError> {
        crate::cuda::poly::eval_poly_ext_at_point_from_base(base_coeffs, coeff_len, x)
    }

    unsafe fn transpose_fp_to_fpext_vec(
        output: &mut DeviceBuffer<EF>,
        input: &DeviceBuffer<F>,
    ) -> Result<(), CudaError> {
        crate::cuda::poly::transpose_fp_to_fpext_vec(output, input)
    }

    unsafe fn eq_hypercube_stage_ext(
        out: *mut EF,
        x_i: EF,
        step: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::poly::eq_hypercube_stage_ext(out, x_i, step)
    }

    unsafe fn eq_hypercube_nonoverlapping_stage_ext(
        out: *mut EF,
        input: *const EF,
        x_i: EF,
        step: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::poly::eq_hypercube_nonoverlapping_stage_ext(out, input, x_i, step)
    }

    unsafe fn eq_hypercube_interleaved_stage_ext(
        out: *mut EF,
        input: *const EF,
        x_i: EF,
        step: u32,
    ) -> Result<(), CudaError> {
        crate::cuda::poly::eq_hypercube_interleaved_stage_ext(out, input, x_i, step)
    }

    // ── logup / GKR ── (delegate to cuda::logup_zerocheck::* directly)
    unsafe fn frac_compute_round_temp_buffer_size(stride: u32) -> u32 {
        crate::cuda::logup_zerocheck::_frac_compute_round_temp_buffer_size(stride)
    }
    unsafe fn frac_build_tree_layer(layer: &mut DeviceBuffer<Frac<EF>>, layer_size: usize, revert: bool, alpha: EF, apply_alpha: bool) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_build_tree_layer(layer, layer_size, revert, alpha, apply_alpha)
    }
    unsafe fn frac_build_tree_two_layers(layer: &mut DeviceBuffer<Frac<EF>>, half_i1: usize) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_build_tree_two_layers(layer, half_i1)
    }
    unsafe fn frac_compute_round(eq_xi: &SqrtEqLayersFor<EF>, pq_buffer: &DeviceBuffer<Frac<EF>>, num_x: usize, lambda: EF, out_device: &mut DeviceBuffer<EF>, tmp_block_sums: &mut DeviceBuffer<EF>) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_compute_round(eq_xi, pq_buffer, num_x, lambda, out_device, tmp_block_sums)
    }
    unsafe fn frac_compute_round_and_revert(eq_xi: &mut SqrtEqLayersFor<EF>, layer: &mut DeviceBuffer<Frac<EF>>, num_x: usize, lambda: EF, out_device: &mut DeviceBuffer<EF>, tmp_block_sums: &mut DeviceBuffer<EF>) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_compute_round_and_revert(eq_xi, layer, num_x, lambda, out_device, tmp_block_sums)
    }
    unsafe fn fold_ef_frac_columns(src: &DeviceBuffer<Frac<EF>>, dst: *mut Frac<EF>, size: usize, r: EF) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::fold_ef_frac_columns_raw(src, dst, size, r)
    }
    unsafe fn fold_ef_frac_columns_inplace(pq_buffer: &mut DeviceBuffer<Frac<EF>>, size: usize, r: EF) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::fold_ef_frac_columns_inplace(pq_buffer, size, r)
    }
    unsafe fn frac_compute_round_and_fold(eq_xi: &mut SqrtEqLayersFor<EF>, src_pq_buffer: &DeviceBuffer<Frac<EF>>, dst_pq_buffer: &mut DeviceBuffer<Frac<EF>>, src_pq_size: usize, lambda: EF, r_prev: EF, out_device: &mut DeviceBuffer<EF>, tmp_block_sums: &mut DeviceBuffer<EF>) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_compute_round_and_fold(eq_xi, src_pq_buffer, dst_pq_buffer, src_pq_size, lambda, r_prev, out_device, tmp_block_sums)
    }
    unsafe fn frac_compute_round_and_fold_inplace(eq_xi: &mut SqrtEqLayersFor<EF>, pq_buffer: &mut DeviceBuffer<Frac<EF>>, src_pq_size: usize, lambda: EF, r_prev: EF, out_device: &mut DeviceBuffer<EF>, tmp_block_sums: &mut DeviceBuffer<EF>) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_compute_round_and_fold_inplace(eq_xi, pq_buffer, src_pq_size, lambda, r_prev, out_device, tmp_block_sums)
    }
    unsafe fn frac_precompute_m_build_raw(pq: *const Frac<EF>, rem_n: usize, w: usize, lambda: EF, r_prev: EF, inline_fold: bool, eq_tail_low: *const EF, eq_tail_high: *const EF, eq_tail_low_cap: usize, tail_tile: usize, partial_out: *mut EF, partial_len: usize, m_total: *mut EF) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_precompute_m_build_raw(pq, rem_n, w, lambda, r_prev, inline_fold, eq_tail_low, eq_tail_high, eq_tail_low_cap, tail_tile, partial_out, partial_len, m_total)
    }
    unsafe fn frac_precompute_m_eval_round_raw(m_total: *const EF, w: usize, t: usize, eq_r_prefix: *const EF, eq_suffix: *const EF, out: *mut EF) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_precompute_m_eval_round_raw(m_total, w, t, eq_r_prefix, eq_suffix, out)
    }
    unsafe fn frac_multifold_raw(src: *const Frac<EF>, dst: *mut Frac<EF>, rem_n: usize, w: usize, eq_r_window: *const EF) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_multifold_raw(src, dst, rem_n, w, eq_r_window)
    }
    unsafe fn frac_add_alpha(data: &DeviceBuffer<Frac<EF>>, alpha: EF) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_add_alpha(data, alpha)
    }
    unsafe fn frac_vector_scalar_multiply_ext_fp(frac_vec: &mut DeviceBuffer<Frac<EF>>, scalar: F) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_vector_scalar_multiply_ext_fp(frac_vec.as_mut_ptr(), scalar, frac_vec.len() as u32)
    }
    unsafe fn frac_matrix_vertically_repeat(out: *mut Frac<EF>, input: &DeviceBuffer<Frac<EF>>, width: u32, lifted_height: u32, height: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_matrix_vertically_repeat(out, input.as_ptr(), width, lifted_height, height)
    }
    unsafe fn frac_matrix_vertically_repeat_ext(out_n: *mut EF, out_d: *mut EF, in_n: *const EF, in_d: *const EF, width: u32, lifted_height: u32, height: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::frac_matrix_vertically_repeat_ext(out_n, out_d, in_n, in_d, width, lifted_height, height)
    }
    unsafe fn bit_rev_frac_ext(d_out: &DeviceBuffer<(EF, EF)>, d_inp: &DeviceBuffer<(EF, EF)>, lg_domain_size: u32, padded_poly_size: u32, poly_count: u32) -> Result<(), CudaError> {
        crate::cuda::ntt::bit_rev_frac_ext(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count)
    }
    unsafe fn bit_rev_frac_ext_build_k2(inout: &DeviceBuffer<(EF, EF)>, lg_domain_size: u32, alpha: EF) -> Result<(), CudaError> {
        crate::cuda::ntt::bit_rev_frac_ext_build_k2(inout, lg_domain_size, alpha)
    }
    unsafe fn fold_ple_from_evals(input: &DeviceBuffer<F>, output: *mut EF, omega_skip_pows: &DeviceBuffer<F>, inv_lagrange_denoms: &DeviceBuffer<EF>, height: u32, width: u32, l_skip: u32, new_height: u32, rotate: bool) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::fold_ple_from_evals(input, output, omega_skip_pows, inv_lagrange_denoms, height, width, l_skip, new_height, rotate)
    }
    unsafe fn logup_gkr_input_eval(is_global: bool, fracs: *mut Frac<EF>, preprocessed: &DeviceBuffer<F>, partitioned_main: &DeviceBuffer<u64>, public_values: &DeviceBuffer<F>, challenges: &DeviceBuffer<EF>, intermediates: &DeviceBuffer<EF>, rules: &DeviceBuffer<u128>, used_nodes: &DeviceBuffer<usize>, pair_idxs: &DeviceBuffer<u32>, height: u32, num_rows_per_tile: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::logup_gkr_input_eval(is_global, fracs, preprocessed, partitioned_main, public_values, challenges, intermediates, rules, used_nodes, pair_idxs, height, num_rows_per_tile)
    }
    fn logup_r0_temp_sums_buffer_size(buffer_size: u32, skip_domain: u32, num_x: u32, num_cosets: u32, max_temp_bytes: usize) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes) }
    }
    fn logup_r0_intermediates_buffer_size(buffer_size: u32, skip_domain: u32, num_x: u32, num_cosets: u32, max_temp_bytes: usize) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_logup_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes) }
    }
    fn zerocheck_r0_temp_sums_buffer_size(buffer_size: u32, skip_domain: u32, num_x: u32, num_cosets: u32, max_temp_bytes: usize) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_zerocheck_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes) }
    }
    fn zerocheck_r0_intermediates_buffer_size(buffer_size: u32, skip_domain: u32, num_x: u32, num_cosets: u32, max_temp_bytes: usize) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_zerocheck_r0_intermediates_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes) }
    }
    fn zerocheck_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_zerocheck_mle_temp_sums_buffer_size(num_x, num_y) }
    }
    fn zerocheck_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_zerocheck_mle_intermediates_buffer_size(buffer_size, num_x, num_y) }
    }
    fn logup_mle_temp_sums_buffer_size(num_x: u32, num_y: u32) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_logup_mle_temp_sums_buffer_size(num_x, num_y) }
    }
    fn logup_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_logup_mle_intermediates_buffer_size(buffer_size, num_x, num_y) }
    }
    fn zerocheck_batch_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_zerocheck_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y) }
    }
    fn logup_batch_mle_intermediates_buffer_size(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
        unsafe { crate::cuda::logup_zerocheck::_logup_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y) }
    }
    unsafe fn zerocheck_ntt_eval_constraints(tmp_sums_buffer: &mut DeviceBuffer<EF>, output: &mut DeviceBuffer<EF>, selectors_cube: &DeviceBuffer<F>, preprocessed: *const F, main_parts: &DeviceBuffer<*const F>, eq_cube: *const EF, lambda_pows: &DeviceBuffer<EF>, public_values: &DeviceBuffer<F>, rules: &DeviceBuffer<u128>, used_nodes: &DeviceBuffer<usize>, buffer_size: u32, intermediates: &mut DeviceBuffer<F>, skip_domain: u32, num_x: u32, height: u32, num_cosets: u32, g_shift: F, max_temp_bytes: usize) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::zerocheck_ntt_eval_constraints(tmp_sums_buffer, output, selectors_cube, preprocessed, main_parts, eq_cube, lambda_pows, public_values, rules, used_nodes, buffer_size, intermediates, skip_domain, num_x, height, num_cosets, g_shift, max_temp_bytes)
    }
    unsafe fn logup_bary_eval_interactions_round0(tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>, output: &mut DeviceBuffer<Frac<EF>>, selectors_cube: &DeviceBuffer<F>, preprocessed: *const F, main_ptrs: &DeviceBuffer<*const F>, eq_cube: *const EF, public_values: &DeviceBuffer<F>, numer_weights: &DeviceBuffer<EF>, denom_weights: &DeviceBuffer<EF>, denom_sum_init: EF, rules: &DeviceBuffer<u128>, buffer_size: u32, intermediates: &mut DeviceBuffer<F>, skip_domain: u32, num_x: u32, height: u32, num_cosets: u32, g_shift: F, max_temp_bytes: usize) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::logup_bary_eval_interactions_round0(tmp_sums_buffer, output, selectors_cube, preprocessed, main_ptrs, eq_cube, public_values, numer_weights, denom_weights, denom_sum_init, rules, buffer_size, intermediates, skip_domain, num_x, height, num_cosets, g_shift, max_temp_bytes)
    }
    unsafe fn zerocheck_eval_mle(tmp_sums_buffer: &mut DeviceBuffer<EF>, output: &mut DeviceBuffer<EF>, eq_xi: *const EF, selectors: *const EF, preprocessed: MainMatrixPtrs<EF>, main: *const MainMatrixPtrs<EF>, lambda_pows: *const EF, lambda_len: usize, public_values: *const F, rules: *const std::ffi::c_void, rules_len: usize, used_nodes: *const usize, used_nodes_len: usize, buffer_size: u32, intermediates: &mut DeviceBuffer<EF>, num_y: u32, num_x: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::zerocheck_eval_mle(tmp_sums_buffer, output, eq_xi, selectors, preprocessed, main, lambda_pows, lambda_len, public_values, rules, rules_len, used_nodes, used_nodes_len, buffer_size, intermediates, num_y, num_x)
    }
    unsafe fn logup_eval_mle(tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>, output: &mut DeviceBuffer<Frac<EF>>, eq_xi: *const EF, selectors: *const EF, preprocessed: MainMatrixPtrs<EF>, main: *const MainMatrixPtrs<EF>, challenges: *const EF, eq_3bs: *const EF, public_values: *const F, rules: *const std::ffi::c_void, used_nodes: *const usize, pair_idxs: *const u32, used_nodes_len: usize, buffer_size: u32, intermediates: &mut DeviceBuffer<EF>, num_y: u32, num_x: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::logup_eval_mle(tmp_sums_buffer, output, eq_xi, selectors, preprocessed, main, challenges, eq_3bs, public_values, rules, used_nodes, pair_idxs, used_nodes_len, buffer_size, intermediates, num_y, num_x)
    }
    unsafe fn precompute_lambda_combinations(out: &mut DeviceBuffer<EF>, headers: *const crate::monomial::MonomialHeader, lambda_terms: *const crate::monomial::LambdaTerm<F>, lambda_pows: &DeviceBuffer<EF>, num_monomials: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::precompute_lambda_combinations(out, headers, lambda_terms, lambda_pows, num_monomials)
    }
    unsafe fn precompute_logup_numer_combinations(out: &mut DeviceBuffer<EF>, headers: *const crate::monomial::MonomialHeader, terms: *const crate::monomial::InteractionMonomialTerm<F>, eq_3bs: &DeviceBuffer<EF>, num_monomials: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::precompute_logup_numer_combinations(out, headers, terms, eq_3bs, num_monomials)
    }
    unsafe fn precompute_logup_denom_combinations(out: &mut DeviceBuffer<EF>, headers: *const crate::monomial::MonomialHeader, terms: *const crate::monomial::InteractionMonomialTerm<F>, beta_pows: &DeviceBuffer<EF>, eq_3bs: &DeviceBuffer<EF>, num_monomials: u32) -> Result<(), CudaError> {
        crate::cuda::logup_zerocheck::precompute_logup_denom_combinations(out, headers, terms, beta_pows, eq_3bs, num_monomials)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KoalaBear implementation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "koala-bear-poseidon2")]
pub use kb_kernels::KoalaBearKernels;

#[cfg(feature = "koala-bear-poseidon2")]
mod kb_kernels {
    use super::*;
    use crate::cuda::{
        logup_zerocheck_kb, mle_interpolate_kb, matrix_kb, poly_kb, stacked_reduction_kb,
        sumcheck_kb, whir_kb,
    };
    use crate::prelude::{EF, F};
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;

    type KbF = KoalaBear;
    type KbEF = BinomialExtensionField<KoalaBear, 4>;

    // Both KoalaBear and BabyBear are #[repr(transparent)] over u32; their extension fields
    // BinomialExtensionField<KB, 4> and BinomialExtensionField<BB, 4> are both [u32; 4].
    // These casts reinterpret KB types as BB types for the KB FFI functions (which still
    // declare BB-typed parameters but link to cuda_kb_all kernels that operate on KB elements).

    #[inline(always)]
    unsafe fn kbef_as_ef_ref(r: &EqEvalSegments<KbEF>) -> &EqEvalSegments<EF> {
        &*(r as *const EqEvalSegments<KbEF> as *const EqEvalSegments<EF>)
    }

    #[inline(always)]
    unsafe fn kbef_buf_as_ef_mut(b: &mut DeviceBuffer<KbEF>) -> &mut DeviceBuffer<EF> {
        &mut *(b as *mut DeviceBuffer<KbEF> as *mut DeviceBuffer<EF>)
    }

    #[inline(always)]
    unsafe fn kbef_ptr_buf_as_ef(b: &DeviceBuffer<*const KbEF>) -> &DeviceBuffer<*const EF> {
        &*(b as *const DeviceBuffer<*const KbEF> as *const DeviceBuffer<*const EF>)
    }

    #[inline(always)]
    unsafe fn kbef_mut_ptr_buf_as_ef(b: &DeviceBuffer<*mut KbEF>) -> &DeviceBuffer<*mut EF> {
        &*(b as *const DeviceBuffer<*mut KbEF> as *const DeviceBuffer<*mut EF>)
    }

    #[inline(always)]
    fn kbef_scalar_as_ef(v: KbEF) -> EF {
        unsafe { std::mem::transmute(v) }
    }

    #[inline(always)]
    fn kbef_buf_deref_as_ef(b: &DeviceBuffer<KbEF>) -> &DeviceBuffer<EF> {
        unsafe { &*(b as *const DeviceBuffer<KbEF> as *const DeviceBuffer<EF>) }
    }

    /// KoalaBear kernel implementation — delegates to `cuda_kb_all` static library.
    #[derive(Copy, Clone, Debug, Default)]
    pub struct KoalaBearKernels;

    impl FieldKernels for KoalaBearKernels {
        type Val = KbF;
        type ValExt = KbEF;

        fn use_cpu_gkr() -> bool {
            // GPU KB GKR sumcheck now works: the earlier failures were caused by a uint64
            // overflow in recip_b0() (KB extension-field inverse), fixed in
            // cuda-common/include/ff/koala_bear_ext.hpp. Verified by test_kb_gkr_gpu_vs_cpu_*.
            false
        }

        fn use_cpu_gkr_input_eval() -> bool {
            // GPU KB logup_gkr_input_eval is correct: verified byte-identical to CPU for all
            // 15 reth188 AIRs (incl. multi-partition, num_int up to 35, 146M fracs) via
            // SWIRL_VERIFY_INPUT_EVAL. The earlier "multi-partition bug" was a misdiagnosis;
            // the real root cause was the recip_b0 overflow (now fixed). GPU path removes the
            // ~17s/chunk CPU input-eval bottleneck.
            false
        }

        fn use_cpu_zerocheck_mle() -> bool {
            // GPU per-round logup/zerocheck MLE + round0 + fold now correct after fixing:
            // (1) recip_b0 uint64 overflow, (2) uninitialized KB DEVICE_NTT_TWIDDLES (round0
            // coset NTT), (3) monomial batch MLE kernels dispatching to BB instead of _kb.
            // Verified: test_kb_fib/interactions_roundtrip + full KB suite pass with this false.
            false
        }

        unsafe fn stacked_r0_temp_buf_size(height: u32, width: u32, l_skip: u32) -> u32 {
            stacked_reduction_kb::_kb_stacked_reduction_r0_required_temp_buffer_size(
                height, width, l_skip,
            )
        }

        unsafe fn stacked_sumcheck_round0(
            eq_r_ns: &EqEvalSegments<KbEF>,
            trace_ptr: *const KbF,
            lambda_pows: *const KbEF,
            block_sums: &mut DeviceBuffer<KbEF>,
            output: &mut DeviceBuffer<KbEF>,
            height: usize,
            width: usize,
            l_skip: usize,
        ) -> Result<(), CudaError> {
            stacked_reduction_kb::stacked_reduction_sumcheck_round0(
                kbef_as_ef_ref(eq_r_ns),
                trace_ptr as *const F,
                lambda_pows as *const EF,
                kbef_buf_as_ef_mut(block_sums),
                kbef_buf_as_ef_mut(output),
                height, width, l_skip,
            )
        }

        unsafe fn stacked_fold_ple(
            src: *const KbF,
            dst: *mut KbEF,
            omega_skip_pows: &DeviceBuffer<KbF>,
            inv_lagrange_denoms: &DeviceBuffer<KbEF>,
            trace_height: usize,
            trace_width: usize,
            l_skip: usize,
        ) -> Result<(), CudaError> {
            stacked_reduction_kb::stacked_reduction_fold_ple(
                src as *const F,
                dst as *mut EF,
                &*(omega_skip_pows as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>),
                kbef_buf_deref_as_ef(inv_lagrange_denoms),
                trace_height, trace_width, l_skip,
            )
        }

        unsafe fn init_k_rot_from_eq_segments(
            eq_r_ns: &EqEvalSegments<KbEF>,
            k_rot_ns: &mut DeviceBuffer<KbEF>,
            k_rot_uni_0: KbEF,
            k_rot_uni_1: KbEF,
            max_n: u32,
        ) -> Result<(), CudaError> {
            stacked_reduction_kb::initialize_k_rot_from_eq_segments(
                kbef_as_ef_ref(eq_r_ns),
                kbef_buf_as_ef_mut(k_rot_ns),
                kbef_scalar_as_ef(k_rot_uni_0),
                kbef_scalar_as_ef(k_rot_uni_1),
                max_n,
            )
        }

        unsafe fn stacked_sumcheck_mle_round(
            q_evals: &DeviceBuffer<*const KbEF>,
            eq_r_ns: &EqEvalSegments<KbEF>,
            k_rot_ns: &EqEvalSegments<KbEF>,
            unstacked_cols: *const UnstackedSlice,
            lambda_pows: *const KbEF,
            output: &mut DeviceBuffer<u64>,
            q_height: usize,
            window_len: usize,
            num_y: usize,
            sm_count: u32,
        ) -> Result<(), CudaError> {
            stacked_reduction_kb::stacked_reduction_sumcheck_mle_round(
                kbef_ptr_buf_as_ef(q_evals),
                kbef_as_ef_ref(eq_r_ns),
                kbef_as_ef_ref(k_rot_ns),
                unstacked_cols,
                lambda_pows as *const EF,
                output,
                q_height, window_len, num_y, sm_count,
            )
        }

        unsafe fn stacked_sumcheck_mle_round_degenerate(
            q_evals: &DeviceBuffer<*const KbEF>,
            eq_ub_ptr: &DeviceBuffer<KbEF>,
            eq_r: KbEF,
            k_rot_r: KbEF,
            unstacked_cols: *const UnstackedSlice,
            lambda_pows: *const KbEF,
            output: &mut DeviceBuffer<u64>,
            q_height: usize,
            window_len: usize,
            l_skip: usize,
            round: usize,
        ) -> Result<(), CudaError> {
            stacked_reduction_kb::stacked_reduction_sumcheck_mle_round_degenerate(
                kbef_ptr_buf_as_ef(q_evals),
                kbef_buf_deref_as_ef(eq_ub_ptr),
                kbef_scalar_as_ef(eq_r),
                kbef_scalar_as_ef(k_rot_r),
                unstacked_cols,
                lambda_pows as *const EF,
                output,
                q_height, window_len, l_skip, round,
            )
        }

        fn vector_scalar_multiply_ext(
            vec: &mut DeviceBuffer<KbEF>,
            scalar: KbEF,
        ) -> Result<(), CudaError> {
            poly_kb::vector_scalar_multiply_ext(
                unsafe { kbef_buf_as_ef_mut(vec) },
                kbef_scalar_as_ef(scalar),
            )
        }

        unsafe fn fold_mle(
            input_matrices: &DeviceBuffer<*const KbEF>,
            output_matrices: &DeviceBuffer<*mut KbEF>,
            widths: &DeviceBuffer<u32>,
            num_matrices: u16,
            output_height: u32,
            max_output_cells: u32,
            r_val: KbEF,
        ) -> Result<(), CudaError> {
            sumcheck_kb::fold_mle(
                kbef_ptr_buf_as_ef(input_matrices),
                kbef_mut_ptr_buf_as_ef(output_matrices),
                widths,
                num_matrices,
                output_height,
                max_output_cells,
                kbef_scalar_as_ef(r_val),
            )
        }

        unsafe fn triangular_fold_mle(
            output: &mut EqEvalSegments<KbEF>,
            input: &EqEvalSegments<KbEF>,
            r: KbEF,
            output_max_n: usize,
        ) -> Result<(), CudaError> {
            let out_bb = &mut *(output as *mut EqEvalSegments<KbEF> as *mut EqEvalSegments<EF>);
            sumcheck_kb::triangular_fold_mle(out_bb, kbef_as_ef_ref(input), kbef_scalar_as_ef(r), output_max_n)
        }

        unsafe fn fold_selectors_round0(
            out: *mut KbEF,
            input: *const KbF,
            is_first: KbEF,
            is_last: KbEF,
            num_x: usize,
        ) -> Result<(), CudaError> {
            logup_zerocheck_kb::fold_selectors_round0_kb(
                out as *mut EF,
                input as *const F,
                kbef_scalar_as_ef(is_first),
                kbef_scalar_as_ef(is_last),
                num_x,
            )
        }

        unsafe fn interpolate_columns_gpu(
            interpolated: &DeviceBuffer<KbEF>,
            columns: &DeviceBuffer<*const KbEF>,
            s_deg: usize,
            num_y: usize,
        ) -> Result<(), CudaError> {
            logup_zerocheck_kb::interpolate_columns_gpu_kb(
                &*(interpolated as *const DeviceBuffer<KbEF> as *const DeviceBuffer<EF>),
                kbef_ptr_buf_as_ef(columns),
                s_deg,
                num_y,
            )
        }

        unsafe fn batch_fold_mle(
            input_matrices: &DeviceBuffer<*const KbEF>,
            output_matrices: &DeviceBuffer<*mut KbEF>,
            widths: &DeviceBuffer<u32>,
            num_matrices: u16,
            log_output_heights: &DeviceBuffer<u8>,
            max_output_cells: u32,
            r_val: KbEF,
        ) -> Result<(), CudaError> {
            sumcheck_kb::batch_fold_mle(
                kbef_ptr_buf_as_ef(input_matrices),
                kbef_mut_ptr_buf_as_ef(output_matrices),
                widths,
                num_matrices,
                log_output_heights,
                max_output_cells,
                kbef_scalar_as_ef(r_val),
            )
        }

        unsafe fn whir_algebraic_batch_traces(
            output: &mut DeviceBuffer<KbF>,
            packets: &DeviceBuffer<BatchingTracePacket<KbF>>,
            mu_powers: &DeviceBuffer<KbEF>,
            skip_domain: u32,
        ) -> Result<(), CudaError> {
            // BatchingTracePacket<KbF> and BatchingTracePacket<F> have same layout
            let output_f = &mut *(output as *mut DeviceBuffer<KbF> as *mut DeviceBuffer<F>);
            let packets_f = &*(packets as *const DeviceBuffer<BatchingTracePacket<KbF>>
                as *const DeviceBuffer<BatchingTracePacket<F>>);
            whir_kb::whir_algebraic_batch_traces(
                output_f,
                packets_f,
                kbef_buf_deref_as_ef(mu_powers),
                skip_domain,
            )
        }

        fn whir_sumcheck_coeff_moments_required_temp_buffer_size(height: u32) -> u32 {
            unsafe { whir_kb::_kb_whir_sumcheck_coeff_moments_required_temp_buffer_size(height) }
        }

        unsafe fn whir_sumcheck_coeff_moments_round(
            f_coeffs: &DeviceBuffer<KbEF>,
            w_moments: &DeviceBuffer<KbEF>,
            output: &mut DeviceBuffer<KbEF>,
            tmp_block_sums: &mut DeviceBuffer<KbEF>,
            height: u32,
        ) -> Result<(), CudaError> {
            whir_kb::whir_sumcheck_coeff_moments_round(
                kbef_buf_deref_as_ef(f_coeffs),
                kbef_buf_deref_as_ef(w_moments),
                kbef_buf_as_ef_mut(output),
                kbef_buf_as_ef_mut(tmp_block_sums),
                height,
            )
        }

        unsafe fn whir_fold_coeffs_and_moments(
            f_coeffs: &DeviceBuffer<KbEF>,
            w_moments: &DeviceBuffer<KbEF>,
            f_folded: &mut DeviceBuffer<KbEF>,
            w_folded: &mut DeviceBuffer<KbEF>,
            alpha: KbEF,
            height: u32,
        ) -> Result<(), CudaError> {
            whir_kb::whir_fold_coeffs_and_moments(
                kbef_buf_deref_as_ef(f_coeffs),
                kbef_buf_deref_as_ef(w_moments),
                kbef_buf_as_ef_mut(f_folded),
                kbef_buf_as_ef_mut(w_folded),
                kbef_scalar_as_ef(alpha),
                height,
            )
        }

        unsafe fn w_moments_accumulate(
            w_moments: &mut DeviceBuffer<KbEF>,
            z0_pows2: &DeviceBuffer<KbEF>,
            z_pows2: &DeviceBuffer<KbF>,
            gamma: KbEF,
            num_queries: u32,
            log_height: u32,
        ) -> Result<(), CudaError> {
            whir_kb::w_moments_accumulate(
                kbef_buf_as_ef_mut(w_moments),
                kbef_buf_deref_as_ef(z0_pows2),
                &*(z_pows2 as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>),
                kbef_scalar_as_ef(gamma),
                num_queries,
                log_height,
            )
        }

        unsafe fn batch_expand_pad(
            output: *mut KbF,
            input: *const KbF,
            poly_count: u32,
            out_size: u32,
            in_size: u32,
        ) -> Result<(), CudaError> {
            matrix_kb::batch_expand_pad(output as *mut F, input as *const F, poly_count, out_size, in_size)
        }

        unsafe fn split_ext_to_base_col_major_matrix(
            d_matrix: &mut DeviceBuffer<KbF>,
            d_poly: &DeviceBuffer<KbEF>,
            poly_len: u64,
            matrix_height: u32,
        ) -> Result<(), CudaError> {
            matrix_kb::split_ext_to_base_col_major_matrix(
                &mut *(d_matrix as *mut DeviceBuffer<KbF> as *mut DeviceBuffer<F>),
                kbef_buf_deref_as_ef(d_poly),
                poly_len, matrix_height,
            )
        }

        unsafe fn mle_interpolate_stage_ext(
            buffer: &mut DeviceBuffer<KbEF>,
            step: u32,
            is_eval_to_coeff: bool,
        ) -> Result<(), CudaError> {
            mle_interpolate_kb::mle_interpolate_stage_ext(kbef_buf_as_ef_mut(buffer), step, is_eval_to_coeff)
        }

        unsafe fn eval_poly_ext_at_point_from_base(
            base_coeffs: &DeviceBuffer<KbF>,
            coeff_len: usize,
            x: KbEF,
        ) -> Result<KbEF, crate::KernelError> {
            let base_bb = &*(base_coeffs as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let result = poly_kb::eval_poly_ext_at_point_from_base(base_bb, coeff_len, kbef_scalar_as_ef(x))?;
            Ok(std::mem::transmute(result))
        }

        unsafe fn transpose_fp_to_fpext_vec(
            output: &mut DeviceBuffer<KbEF>,
            input: &DeviceBuffer<KbF>,
        ) -> Result<(), CudaError> {
            poly_kb::transpose_fp_to_fpext_vec(
                kbef_buf_as_ef_mut(output),
                &*(input as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>),
            )
        }

        unsafe fn eq_hypercube_stage_ext(
            out: *mut KbEF,
            x_i: KbEF,
            step: u32,
        ) -> Result<(), CudaError> {
            poly_kb::eq_hypercube_stage_ext(out as *mut EF, kbef_scalar_as_ef(x_i), step)
        }

        unsafe fn eq_hypercube_nonoverlapping_stage_ext(
            out: *mut KbEF,
            input: *const KbEF,
            x_i: KbEF,
            step: u32,
        ) -> Result<(), CudaError> {
            poly_kb::eq_hypercube_nonoverlapping_stage_ext(
                out as *mut EF,
                input as *const EF,
                kbef_scalar_as_ef(x_i),
                step,
            )
        }

        unsafe fn eq_hypercube_interleaved_stage_ext(
            out: *mut KbEF,
            input: *const KbEF,
            x_i: KbEF,
            step: u32,
        ) -> Result<(), CudaError> {
            poly_kb::eq_hypercube_interleaved_stage_ext(
                out as *mut EF,
                input as *const EF,
                kbef_scalar_as_ef(x_i),
                step,
            )
        }

        // ─── GKR fractional sumcheck kernels (KoalaBear) ─────────────────────────

        unsafe fn frac_compute_round_temp_buffer_size(stride: u32) -> u32 {
            logup_zerocheck_kb::_kb_frac_compute_round_temp_buffer_size(stride)
        }

        unsafe fn frac_build_tree_layer(
            layer: &mut DeviceBuffer<Frac<KbEF>>,
            layer_size: usize,
            revert: bool,
            alpha: KbEF,
            apply_alpha: bool,
        ) -> Result<(), CudaError> {
            // SAFETY: Frac<KbEF> and Frac<EF> have identical memory layout ([u32;8]).
            let layer_bb = &mut *(layer as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_build_tree_layer_kb(layer_bb, layer_size, revert, kbef_scalar_as_ef(alpha), apply_alpha)
        }

        unsafe fn frac_build_tree_two_layers(
            layer: &mut DeviceBuffer<Frac<KbEF>>,
            half_i1: usize,
        ) -> Result<(), CudaError> {
            let layer_bb = &mut *(layer as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_build_tree_two_layers_kb(layer_bb, half_i1)
        }

        unsafe fn frac_compute_round(
            eq_xi: &SqrtEqLayersFor<KbEF>,
            pq_buffer: &DeviceBuffer<Frac<KbEF>>,
            num_x: usize,
            lambda: KbEF,
            out_device: &mut DeviceBuffer<KbEF>,
            tmp_block_sums: &mut DeviceBuffer<KbEF>,
        ) -> Result<(), CudaError> {
            // SAFETY: SqrtEqLayersFor<KbEF> and SqrtEqLayers (= SqrtEqLayersFor<EF>)
            // have identical layouts.
            let sq = &*(eq_xi as *const SqrtEqLayersFor<KbEF> as *const crate::poly::SqrtEqLayers);
            let pq_bb = &*(pq_buffer as *const DeviceBuffer<Frac<KbEF>> as *const DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_compute_round_kb(sq, pq_bb, num_x, kbef_scalar_as_ef(lambda), kbef_buf_as_ef_mut(out_device), kbef_buf_as_ef_mut(tmp_block_sums))
        }

        unsafe fn frac_compute_round_and_revert(
            eq_xi: &mut SqrtEqLayersFor<KbEF>,
            layer: &mut DeviceBuffer<Frac<KbEF>>,
            num_x: usize,
            lambda: KbEF,
            out_device: &mut DeviceBuffer<KbEF>,
            tmp_block_sums: &mut DeviceBuffer<KbEF>,
        ) -> Result<(), CudaError> {
            let sq = &*(eq_xi as *const SqrtEqLayersFor<KbEF> as *const crate::poly::SqrtEqLayers);
            let layer_bb = &mut *(layer as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_compute_round_and_revert_kb(sq, layer_bb, num_x, kbef_scalar_as_ef(lambda), kbef_buf_as_ef_mut(out_device), kbef_buf_as_ef_mut(tmp_block_sums))
        }

        unsafe fn fold_ef_frac_columns(
            src: &DeviceBuffer<Frac<KbEF>>,
            dst: *mut Frac<KbEF>,
            size: usize,
            r: KbEF,
        ) -> Result<(), CudaError> {
            let src_bb = &*(src as *const DeviceBuffer<Frac<KbEF>> as *const DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::fold_ef_frac_columns_raw_kb(src_bb, dst as *mut Frac<EF>, size, kbef_scalar_as_ef(r))
        }

        unsafe fn fold_ef_frac_columns_inplace(
            pq_buffer: &mut DeviceBuffer<Frac<KbEF>>,
            size: usize,
            r: KbEF,
        ) -> Result<(), CudaError> {
            let buf_bb = &mut *(pq_buffer as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::fold_ef_frac_columns_inplace_kb(buf_bb, size, kbef_scalar_as_ef(r))
        }

        unsafe fn frac_compute_round_and_fold(
            eq_xi: &mut SqrtEqLayersFor<KbEF>,
            src_pq_buffer: &DeviceBuffer<Frac<KbEF>>,
            dst_pq_buffer: &mut DeviceBuffer<Frac<KbEF>>,
            src_pq_size: usize,
            lambda: KbEF,
            r_prev: KbEF,
            out_device: &mut DeviceBuffer<KbEF>,
            tmp_block_sums: &mut DeviceBuffer<KbEF>,
        ) -> Result<(), CudaError> {
            let sq = &*(eq_xi as *const SqrtEqLayersFor<KbEF> as *const crate::poly::SqrtEqLayers);
            let src_bb = &*(src_pq_buffer as *const DeviceBuffer<Frac<KbEF>> as *const DeviceBuffer<Frac<EF>>);
            let dst_bb = &mut *(dst_pq_buffer as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_compute_round_and_fold_kb(sq, src_bb, dst_bb, src_pq_size, kbef_scalar_as_ef(lambda), kbef_scalar_as_ef(r_prev), kbef_buf_as_ef_mut(out_device), kbef_buf_as_ef_mut(tmp_block_sums))
        }

        unsafe fn frac_compute_round_and_fold_inplace(
            eq_xi: &mut SqrtEqLayersFor<KbEF>,
            pq_buffer: &mut DeviceBuffer<Frac<KbEF>>,
            src_pq_size: usize,
            lambda: KbEF,
            r_prev: KbEF,
            out_device: &mut DeviceBuffer<KbEF>,
            tmp_block_sums: &mut DeviceBuffer<KbEF>,
        ) -> Result<(), CudaError> {
            let sq = &*(eq_xi as *const SqrtEqLayersFor<KbEF> as *const crate::poly::SqrtEqLayers);
            let buf_bb = &mut *(pq_buffer as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_compute_round_and_fold_inplace_kb(sq, buf_bb, src_pq_size, kbef_scalar_as_ef(lambda), kbef_scalar_as_ef(r_prev), kbef_buf_as_ef_mut(out_device), kbef_buf_as_ef_mut(tmp_block_sums))
        }

        unsafe fn frac_precompute_m_build_raw(
            pq: *const Frac<KbEF>,
            rem_n: usize,
            w: usize,
            lambda: KbEF,
            r_prev: KbEF,
            inline_fold: bool,
            eq_tail_low: *const KbEF,
            eq_tail_high: *const KbEF,
            eq_tail_low_cap: usize,
            tail_tile: usize,
            partial_out: *mut KbEF,
            partial_len: usize,
            m_total: *mut KbEF,
        ) -> Result<(), CudaError> {
            logup_zerocheck_kb::frac_precompute_m_build_raw_kb(
                pq as *const Frac<EF>,
                rem_n, w,
                kbef_scalar_as_ef(lambda),
                kbef_scalar_as_ef(r_prev),
                inline_fold,
                eq_tail_low as *const EF,
                eq_tail_high as *const EF,
                eq_tail_low_cap,
                tail_tile,
                partial_out as *mut EF,
                partial_len,
                m_total as *mut EF,
            )
        }

        unsafe fn frac_precompute_m_eval_round_raw(
            m_total: *const KbEF,
            w: usize,
            t: usize,
            eq_r_prefix: *const KbEF,
            eq_suffix: *const KbEF,
            out: *mut KbEF,
        ) -> Result<(), CudaError> {
            logup_zerocheck_kb::frac_precompute_m_eval_round_raw_kb(
                m_total as *const EF,
                w, t,
                eq_r_prefix as *const EF,
                eq_suffix as *const EF,
                out as *mut EF,
            )
        }

        unsafe fn frac_multifold_raw(
            src: *const Frac<KbEF>,
            dst: *mut Frac<KbEF>,
            rem_n: usize,
            w: usize,
            eq_r_window: *const KbEF,
        ) -> Result<(), CudaError> {
            logup_zerocheck_kb::frac_multifold_raw_kb(
                src as *const Frac<EF>,
                dst as *mut Frac<EF>,
                rem_n, w,
                eq_r_window as *const EF,
            )
        }

        unsafe fn frac_add_alpha(
            data: &DeviceBuffer<Frac<KbEF>>,
            alpha: KbEF,
        ) -> Result<(), CudaError> {
            let data_bb = &*(data as *const DeviceBuffer<Frac<KbEF>> as *const DeviceBuffer<Frac<EF>>);
            logup_zerocheck_kb::frac_add_alpha_kb(data_bb, kbef_scalar_as_ef(alpha))
        }

        unsafe fn bit_rev_frac_ext(
            d_out: &DeviceBuffer<(KbEF, KbEF)>,
            d_inp: &DeviceBuffer<(KbEF, KbEF)>,
            lg_domain_size: u32,
            padded_poly_size: u32,
            poly_count: u32,
        ) -> Result<(), CudaError> {
            // TODO: Use KB-specific bit_rev_frac_ext when available.
            // For now, reinterpret as BB types (same u32 layout) and call BB NTT kernel.
            let out_bb = &*(d_out as *const DeviceBuffer<(KbEF, KbEF)> as *const DeviceBuffer<(EF, EF)>);
            let inp_bb = &*(d_inp as *const DeviceBuffer<(KbEF, KbEF)> as *const DeviceBuffer<(EF, EF)>);
            crate::cuda::ntt::bit_rev_frac_ext(out_bb, inp_bb, lg_domain_size, padded_poly_size, poly_count)
        }

        unsafe fn bit_rev_frac_ext_build_k2(
            inout: &DeviceBuffer<(KbEF, KbEF)>,
            lg_domain_size: u32,
            alpha: KbEF,
        ) -> Result<(), CudaError> {
            // Use KB-specific gkr bit_rev kernel (uses KB field arithmetic, not BB fallback).
            // The BB fallback applies BB field operations to KB data, producing wrong results
            // for total_leaves = 2^(l_skip + n_logup) ≥ 2^(l_skip + 22) due to field mismatch.
            let buf = &*(inout as *const DeviceBuffer<(KbEF, KbEF)> as *const DeviceBuffer<(EF, EF)>);
            crate::cuda::logup_zerocheck_kb::gkr_bit_rev_frac_ext_build_k2_kb(
                buf, lg_domain_size, kbef_scalar_as_ef(alpha)
            )
        }
        // ── logup: missing methods ──────────────────────────────────────────────
        unsafe fn frac_vector_scalar_multiply_ext_fp(frac_vec: &mut DeviceBuffer<Frac<KbEF>>, scalar: KbF) -> Result<(), CudaError> {
            let fv_bb = &mut *(frac_vec as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            crate::cuda::logup_zerocheck_kb::frac_vector_scalar_multiply_ext_fp_kb(fv_bb.as_mut_ptr(), std::mem::transmute::<KbF, F>(scalar), fv_bb.len() as u32)
        }
        unsafe fn frac_matrix_vertically_repeat(out: *mut Frac<KbEF>, input: &DeviceBuffer<Frac<KbEF>>, w: u32, lh: u32, h: u32) -> Result<(), CudaError> {
            let i_bb = &*(input as *const DeviceBuffer<Frac<KbEF>> as *const DeviceBuffer<Frac<EF>>);
            crate::cuda::logup_zerocheck_kb::frac_matrix_vertically_repeat_kb(out as *mut Frac<EF>, i_bb.as_ptr(), w, lh, h)
        }
        unsafe fn frac_matrix_vertically_repeat_ext(on: *mut KbEF, od: *mut KbEF, in_n: *const KbEF, in_d: *const KbEF, w: u32, lh: u32, h: u32) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::frac_matrix_vertically_repeat_ext_kb(on as *mut EF, od as *mut EF, in_n as *const EF, in_d as *const EF, w, lh, h)
        }
        // ── Monomial-basis batch MLE: dispatch to KB kernels (args already EF-typed) ──
        unsafe fn zerocheck_monomial_batched(
            tmp_sums: &mut DeviceBuffer<EF>, output: &mut DeviceBuffer<EF>,
            block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
            air_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::MonomialAirCtx>,
            air_block_offsets: &DeviceBuffer<u32>, num_blocks: u32, num_x: u32, num_airs: u32,
            threads_per_block: u32,
        ) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::zerocheck_monomial_batched_kb(
                tmp_sums, output, block_ctxs, air_ctxs, air_block_offsets, num_blocks, num_x,
                num_airs, threads_per_block,
            )
        }
        unsafe fn zerocheck_monomial_par_y_batched(
            tmp_sums: &mut DeviceBuffer<EF>, output: &mut DeviceBuffer<EF>,
            block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
            air_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::MonomialAirCtx>,
            air_block_offsets: &DeviceBuffer<u32>, num_blocks: u32, num_x: u32, num_airs: u32,
            chunk_size: u32, threads_per_block: u32,
        ) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::zerocheck_monomial_par_y_batched_kb(
                tmp_sums, output, block_ctxs, air_ctxs, air_block_offsets, num_blocks, num_x,
                num_airs, chunk_size, threads_per_block,
            )
        }
        unsafe fn logup_monomial_batched(
            tmp_sums: &mut DeviceBuffer<Frac<EF>>, output: &mut DeviceBuffer<Frac<EF>>,
            block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
            common_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupMonomialCommonCtx>,
            numer_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupMonomialCtx>,
            denom_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupMonomialCtx>,
            air_block_offsets: &DeviceBuffer<u32>, num_blocks: u32, num_x: u32, num_airs: u32,
            threads_per_block: u32,
        ) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::logup_monomial_batched_kb(
                tmp_sums, output, block_ctxs, common_ctxs, numer_ctxs, denom_ctxs,
                air_block_offsets, num_blocks, num_x, num_airs, threads_per_block,
            )
        }
        unsafe fn zerocheck_batch_eval_mle(
            tmp_sums_buffer: &mut DeviceBuffer<EF>, output: &mut DeviceBuffer<EF>,
            block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
            zc_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::ZerocheckCtx>,
            air_block_offsets: &DeviceBuffer<u32>, lambda_pows: &DeviceBuffer<EF>,
            lambda_len: usize, num_blocks: u32, num_x: u32, num_airs: u32, threads_per_block: u32,
        ) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::zerocheck_batch_eval_mle_kb(
                tmp_sums_buffer, output, block_ctxs, zc_ctxs, air_block_offsets, lambda_pows,
                lambda_len, num_blocks, num_x, num_airs, threads_per_block,
            )
        }
        unsafe fn logup_batch_eval_mle(
            tmp_sums_buffer: &mut DeviceBuffer<Frac<EF>>, output: &mut DeviceBuffer<Frac<EF>>,
            block_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::BlockCtx>,
            logup_ctxs: &DeviceBuffer<crate::cuda::logup_zerocheck::LogupCtx>,
            air_block_offsets: &DeviceBuffer<u32>, num_blocks: u32, num_x: u32, num_airs: u32,
            threads_per_block: u32,
        ) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::logup_batch_eval_mle_kb(
                tmp_sums_buffer, output, block_ctxs, logup_ctxs, air_block_offsets, num_blocks,
                num_x, num_airs, threads_per_block,
            )
        }
        unsafe fn fold_ple_from_evals(input: &DeviceBuffer<KbF>, output: *mut KbEF, omega: &DeviceBuffer<KbF>, ild: &DeviceBuffer<KbEF>, h: u32, w: u32, ls: u32, nh: u32, rot: bool) -> Result<(), CudaError> {
            let i_bb = &*(input as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let o_bb = &*(omega as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            crate::cuda::logup_zerocheck_kb::fold_ple_from_evals_kb(i_bb, output as *mut EF, o_bb, kbef_buf_deref_as_ef(ild), h, w, ls, nh, rot)
        }
        unsafe fn logup_gkr_input_eval(is_g: bool, fracs: *mut Frac<KbEF>, prep: &DeviceBuffer<KbF>, pm: &DeviceBuffer<u64>, pv: &DeviceBuffer<KbF>, ch: &DeviceBuffer<KbEF>, inter: &DeviceBuffer<KbEF>, rules: &DeviceBuffer<u128>, un: &DeviceBuffer<usize>, pi: &DeviceBuffer<u32>, h: u32, nrpt: u32) -> Result<(), CudaError> {
            let p_bb = &*(prep as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let pv_bb = &*(pv as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            crate::cuda::logup_zerocheck_kb::logup_gkr_input_eval_kb(is_g, fracs as *mut Frac<EF>, p_bb, pm, pv_bb, kbef_buf_deref_as_ef(ch), kbef_buf_deref_as_ef(inter), rules, un, pi, h, nrpt)
        }
        fn logup_r0_temp_sums_buffer_size(b: u32, s: u32, x: u32, c: u32, m: usize) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_logup_r0_temp_sums_buffer_size(b, s, x, c, m) } }
        fn logup_r0_intermediates_buffer_size(b: u32, s: u32, x: u32, c: u32, m: usize) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_logup_r0_intermediates_buffer_size(b, s, x, c, m) } }
        fn zerocheck_r0_temp_sums_buffer_size(b: u32, s: u32, x: u32, c: u32, m: usize) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_zerocheck_r0_temp_sums_buffer_size(b, s, x, c, m) } }
        fn zerocheck_r0_intermediates_buffer_size(b: u32, s: u32, x: u32, c: u32, m: usize) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_zerocheck_r0_intermediates_buffer_size(b, s, x, c, m) } }
        fn zerocheck_mle_temp_sums_buffer_size(x: u32, y: u32) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_zerocheck_mle_temp_sums_buffer_size(x, y) } }
        fn zerocheck_mle_intermediates_buffer_size(b: u32, x: u32, y: u32) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_zerocheck_mle_intermediates_buffer_size(b, x, y) } }
        fn logup_mle_temp_sums_buffer_size(x: u32, y: u32) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_logup_mle_temp_sums_buffer_size(x, y) } }
        fn logup_mle_intermediates_buffer_size(b: u32, x: u32, y: u32) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_logup_mle_intermediates_buffer_size(b, x, y) } }
        fn zerocheck_batch_mle_intermediates_buffer_size(b: u32, x: u32, y: u32) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_zerocheck_batch_mle_intermediates_buffer_size(b, x, y) } }
        fn logup_batch_mle_intermediates_buffer_size(b: u32, x: u32, y: u32) -> usize { unsafe { crate::cuda::logup_zerocheck_kb::_kb_logup_batch_mle_intermediates_buffer_size(b, x, y) } }
        unsafe fn zerocheck_ntt_eval_constraints(ts: &mut DeviceBuffer<KbEF>, out: &mut DeviceBuffer<KbEF>, sel: &DeviceBuffer<KbF>, prep: *const KbF, main: &DeviceBuffer<*const KbF>, eq: *const KbEF, lp: &DeviceBuffer<KbEF>, pv: &DeviceBuffer<KbF>, rules: &DeviceBuffer<u128>, un: &DeviceBuffer<usize>, bs: u32, inter: &mut DeviceBuffer<KbF>, sd: u32, nx: u32, h: u32, nc: u32, gs: KbF, mt: usize) -> Result<(), CudaError> {
            let sel_bb = &*(sel as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let main_bb = &*(main as *const DeviceBuffer<*const KbF> as *const DeviceBuffer<*const F>);
            let pv_bb = &*(pv as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let inter_bb = &mut *(inter as *mut DeviceBuffer<KbF> as *mut DeviceBuffer<F>);
            crate::cuda::logup_zerocheck_kb::zerocheck_ntt_eval_constraints_kb(kbef_buf_as_ef_mut(ts), kbef_buf_as_ef_mut(out), sel_bb, prep as *const F, main_bb, eq as *const EF, kbef_buf_deref_as_ef(lp), pv_bb, rules, un, bs, inter_bb, sd, nx, h, nc, std::mem::transmute::<KbF, F>(gs), mt)
        }
        unsafe fn logup_bary_eval_interactions_round0(ts: &mut DeviceBuffer<Frac<KbEF>>, out: &mut DeviceBuffer<Frac<KbEF>>, sel: &DeviceBuffer<KbF>, prep: *const KbF, main: &DeviceBuffer<*const KbF>, eq: *const KbEF, pv: &DeviceBuffer<KbF>, nw: &DeviceBuffer<KbEF>, dw: &DeviceBuffer<KbEF>, dsi: KbEF, rules: &DeviceBuffer<u128>, bs: u32, inter: &mut DeviceBuffer<KbF>, sd: u32, nx: u32, h: u32, nc: u32, gs: KbF, mt: usize) -> Result<(), CudaError> {
            let ts_bb = &mut *(ts as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            let out_bb = &mut *(out as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            let sel_bb = &*(sel as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let main_bb = &*(main as *const DeviceBuffer<*const KbF> as *const DeviceBuffer<*const F>);
            let pv_bb = &*(pv as *const DeviceBuffer<KbF> as *const DeviceBuffer<F>);
            let inter_bb = &mut *(inter as *mut DeviceBuffer<KbF> as *mut DeviceBuffer<F>);
            crate::cuda::logup_zerocheck_kb::logup_bary_eval_interactions_round0_kb(ts_bb, out_bb, sel_bb, prep as *const F, main_bb, eq as *const EF, pv_bb, kbef_buf_deref_as_ef(nw), kbef_buf_deref_as_ef(dw), kbef_scalar_as_ef(dsi), rules, bs, inter_bb, sd, nx, h, nc, std::mem::transmute::<KbF, F>(gs), mt)
        }
        unsafe fn zerocheck_eval_mle(ts: &mut DeviceBuffer<KbEF>, out: &mut DeviceBuffer<KbEF>, eq: *const KbEF, sel: *const KbEF, prep: MainMatrixPtrs<KbEF>, main: *const MainMatrixPtrs<KbEF>, lp: *const KbEF, ll: usize, pv: *const KbF, rules: *const std::ffi::c_void, rl: usize, un: *const usize, unl: usize, bs: u32, inter: &mut DeviceBuffer<KbEF>, ny: u32, nx: u32) -> Result<(), CudaError> {
            let prep_bb: MainMatrixPtrs<EF> = std::mem::transmute(prep);
            let main_bb = main as *const MainMatrixPtrs<EF>;
            crate::cuda::logup_zerocheck_kb::zerocheck_eval_mle_kb(kbef_buf_as_ef_mut(ts), kbef_buf_as_ef_mut(out), eq as *const EF, sel as *const EF, prep_bb, main_bb, lp as *const EF, ll, pv as *const F, rules, rl, un, unl, bs, kbef_buf_as_ef_mut(inter), ny, nx)
        }
        unsafe fn logup_eval_mle(ts: &mut DeviceBuffer<Frac<KbEF>>, out: &mut DeviceBuffer<Frac<KbEF>>, eq: *const KbEF, sel: *const KbEF, prep: MainMatrixPtrs<KbEF>, main: *const MainMatrixPtrs<KbEF>, ch: *const KbEF, e3b: *const KbEF, pv: *const KbF, rules: *const std::ffi::c_void, un: *const usize, pi: *const u32, unl: usize, bs: u32, inter: &mut DeviceBuffer<KbEF>, ny: u32, nx: u32) -> Result<(), CudaError> {
            let ts_bb = &mut *(ts as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            let out_bb = &mut *(out as *mut DeviceBuffer<Frac<KbEF>> as *mut DeviceBuffer<Frac<EF>>);
            let prep_bb: MainMatrixPtrs<EF> = std::mem::transmute(prep);
            let main_bb = main as *const MainMatrixPtrs<EF>;
            let inter_bb = kbef_buf_as_ef_mut(inter);
            crate::cuda::logup_zerocheck_kb::logup_eval_mle_kb(ts_bb, out_bb, eq as *const EF, sel as *const EF, prep_bb, main_bb, ch as *const EF, e3b as *const EF, pv as *const F, rules, un, pi, unl, bs, inter_bb, ny, nx)
        }

        unsafe fn precompute_lambda_combinations(out: &mut DeviceBuffer<KbEF>, headers: *const crate::monomial::MonomialHeader, lambda_terms: *const crate::monomial::LambdaTerm<KbF>, lambda_pows: &DeviceBuffer<KbEF>, num_monomials: u32) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::precompute_lambda_combinations_kb(kbef_buf_as_ef_mut(out), headers, lambda_terms as *const crate::monomial::LambdaTerm<F>, kbef_buf_deref_as_ef(lambda_pows), num_monomials)
        }
        unsafe fn precompute_logup_numer_combinations(out: &mut DeviceBuffer<KbEF>, headers: *const crate::monomial::MonomialHeader, terms: *const crate::monomial::InteractionMonomialTerm<KbF>, eq_3bs: &DeviceBuffer<KbEF>, num_monomials: u32) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::precompute_logup_numer_combinations_kb(kbef_buf_as_ef_mut(out), headers, terms as *const crate::monomial::InteractionMonomialTerm<F>, kbef_buf_deref_as_ef(eq_3bs), num_monomials)
        }
        unsafe fn precompute_logup_denom_combinations(out: &mut DeviceBuffer<KbEF>, headers: *const crate::monomial::MonomialHeader, terms: *const crate::monomial::InteractionMonomialTerm<KbF>, beta_pows: &DeviceBuffer<KbEF>, eq_3bs: &DeviceBuffer<KbEF>, num_monomials: u32) -> Result<(), CudaError> {
            crate::cuda::logup_zerocheck_kb::precompute_logup_denom_combinations_kb(kbef_buf_as_ef_mut(out), headers, terms as *const crate::monomial::InteractionMonomialTerm<F>, kbef_buf_deref_as_ef(beta_pows), kbef_buf_deref_as_ef(eq_3bs), num_monomials)
        }

    }
}