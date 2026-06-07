/*
 * KoalaBear proving kernel symbol prefix definitions.
 *
 * This file renames all extern "C" launcher functions to their _kb_ prefixed
 * equivalents so that the KoalaBear proving library (cuda_proving_kb) can coexist
 * with the BabyBear proving library (cuda-backend) in the same linked binary.
 *
 * Pattern: each _funcname → _kb_funcname
 *
 * Functions are gathered from:
 *   cuda/src/logup_zerocheck/logup_round0.cu
 *   cuda/src/logup_zerocheck/zerocheck_round0.cu
 *   cuda/src/logup_zerocheck/batch_mle.cu
 *   cuda/src/logup_zerocheck/mle.cu
 *   cuda/src/logup_zerocheck/gkr.cu
 *   cuda/src/logup_zerocheck/gkr_input.cu
 *   cuda/src/logup_zerocheck/utils.cu
 *   cuda/src/logup_zerocheck/batch_mle_monomial.cu
 *   cuda/src/whir.cu
 *   cuda/src/sumcheck.cu
 *   cuda/src/poly.cu
 *   cuda/src/matrix.cu
 *   cuda/src/mle_interpolate.cu
 *   cuda/src/stacked_reduction.cu
 */

#pragma once

// logup_zerocheck/logup_round0.cu
#define _logup_r0_temp_sums_buffer_size         _kb_logup_r0_temp_sums_buffer_size
#define _logup_r0_intermediates_buffer_size      _kb_logup_r0_intermediates_buffer_size
#define _logup_bary_eval_interactions_round0     _kb_logup_bary_eval_interactions_round0

// logup_zerocheck/zerocheck_round0.cu
#define _zerocheck_r0_temp_sums_buffer_size      _kb_zerocheck_r0_temp_sums_buffer_size
#define _zerocheck_r0_intermediates_buffer_size  _kb_zerocheck_r0_intermediates_buffer_size
#define _zerocheck_ntt_eval_constraints          _kb_zerocheck_ntt_eval_constraints
#define _fold_selectors_round0                   _kb_fold_selectors_round0

// logup_zerocheck/batch_mle.cu
#define _zerocheck_batch_mle_intermediates_buffer_size  _kb_zerocheck_batch_mle_intermediates_buffer_size
#define _logup_batch_mle_intermediates_buffer_size      _kb_logup_batch_mle_intermediates_buffer_size
#define _zerocheck_batch_eval_mle                _kb_zerocheck_batch_eval_mle
#define _logup_batch_eval_mle                    _kb_logup_batch_eval_mle

// logup_zerocheck/mle.cu
#define _zerocheck_mle_temp_sums_buffer_size     _kb_zerocheck_mle_temp_sums_buffer_size
#define _zerocheck_mle_intermediates_buffer_size _kb_zerocheck_mle_intermediates_buffer_size
#define _zerocheck_eval_mle                      _kb_zerocheck_eval_mle
#define _logup_mle_temp_sums_buffer_size         _kb_logup_mle_temp_sums_buffer_size
#define _logup_mle_intermediates_buffer_size     _kb_logup_mle_intermediates_buffer_size
#define _logup_eval_mle                          _kb_logup_eval_mle

// logup_zerocheck/gkr.cu
#define _frac_build_tree_layer                   _kb_frac_build_tree_layer
#define _frac_build_tree_two_layers              _kb_frac_build_tree_two_layers
#define _frac_compute_round_temp_buffer_size     _kb_frac_compute_round_temp_buffer_size
#define _frac_compute_round                      _kb_frac_compute_round
#define _frac_compute_round_and_revert           _kb_frac_compute_round_and_revert
#define _frac_compute_round_and_fold             _kb_frac_compute_round_and_fold
#define _frac_compute_round_and_fold_inplace     _kb_frac_compute_round_and_fold_inplace
#define _frac_precompute_m_build                 _kb_frac_precompute_m_build
#define _frac_precompute_m_eval_round            _kb_frac_precompute_m_eval_round
#define _frac_multifold                          _kb_frac_multifold
#define _frac_fold_fpext_columns                 _kb_frac_fold_fpext_columns
#define _frac_add_alpha                          _kb_frac_add_alpha
#define _frac_vector_scalar_multiply_ext_fp      _kb_frac_vector_scalar_multiply_ext_fp
// Note: ntt_bitrev_kb.cu also defines _kb_bit_rev_frac_ext_build_k2 directly.
// Rename gkr.cu's version to a private name to avoid duplicate-symbol link errors.
// The KB Rust FieldKernels impl currently calls the BB fallback, not either of these.
#define _bit_rev_frac_ext_build_k2               _kb_gkr_bit_rev_frac_ext_build_k2

// logup_zerocheck/gkr_input.cu
#define _logup_gkr_input_eval                    _kb_logup_gkr_input_eval

// logup_zerocheck/utils.cu
#define _fold_ple_from_evals                     _kb_fold_ple_from_evals
#define _interpolate_columns                     _kb_interpolate_columns
#define _frac_matrix_vertically_repeat           _kb_frac_matrix_vertically_repeat
#define _frac_matrix_vertically_repeat_ext       _kb_frac_matrix_vertically_repeat_ext

// logup_zerocheck/batch_mle_monomial.cu
#define _precompute_lambda_combinations          _kb_precompute_lambda_combinations
#define _zerocheck_monomial_batched              _kb_zerocheck_monomial_batched
#define _zerocheck_monomial_par_y_batched        _kb_zerocheck_monomial_par_y_batched
#define _precompute_logup_numer_combinations     _kb_precompute_logup_numer_combinations
#define _precompute_logup_denom_combinations     _kb_precompute_logup_denom_combinations
#define _logup_monomial_batched                  _kb_logup_monomial_batched

// whir.cu
#define _whir_algebraic_batch_traces             _kb_whir_algebraic_batch_traces
#define _whir_sumcheck_coeff_moments_required_temp_buffer_size  _kb_whir_sumcheck_coeff_moments_required_temp_buffer_size
#define _whir_sumcheck_coeff_moments_round       _kb_whir_sumcheck_coeff_moments_round
#define _whir_fold_coeffs_and_moments            _kb_whir_fold_coeffs_and_moments
#define _w_moments_accumulate                    _kb_w_moments_accumulate

// sumcheck.cu
#define _fold_mle                                _kb_fold_mle
#define _fold_mle_column                         _kb_fold_mle_column
#define _batch_fold_mle                          _kb_batch_fold_mle
#define _fold_ple_from_coeffs                    _kb_fold_ple_from_coeffs
#define _reduce_over_x_and_cols                  _kb_reduce_over_x_and_cols
#define _sumcheck_mle_round                      _kb_sumcheck_mle_round
#define _triangular_fold_mle                     _kb_triangular_fold_mle

// poly.cu
#define _algebraic_batch_matrices                _kb_algebraic_batch_matrices
#define _eq_hypercube_stage_ext                  _kb_eq_hypercube_stage_ext
#define _mobius_eq_hypercube_stage_ext           _kb_mobius_eq_hypercube_stage_ext
#define _eq_hypercube_nonoverlapping_stage_ext   _kb_eq_hypercube_nonoverlapping_stage_ext
#define _eq_hypercube_interleaved_stage_ext      _kb_eq_hypercube_interleaved_stage_ext
#define _batch_eq_hypercube_stage                _kb_batch_eq_hypercube_stage
#define _eval_poly_ext_at_point                  _kb_eval_poly_ext_at_point
#define _vector_scalar_multiply_ext              _kb_vector_scalar_multiply_ext
#define _transpose_fp_to_fpext_vec               _kb_transpose_fp_to_fpext_vec

// matrix.cu
#define _matrix_transpose_fp                     _kb_matrix_transpose_fp
#define _matrix_transpose_fpext                  _kb_matrix_transpose_fpext
#define _matrix_get_rows_fp                      _kb_matrix_get_rows_fp
#define _split_ext_to_base_col_major_matrix      _kb_split_ext_to_base_col_major_matrix
#define _batch_rotate_pad                        _kb_batch_rotate_pad
#define _lift_padded_matrix_evals                _kb_lift_padded_matrix_evals
#define _collapse_strided_matrix                 _kb_collapse_strided_matrix
#define _batch_expand_pad                        _kb_batch_expand_pad
#define _batch_expand_pad_wide                   _kb_batch_expand_pad_wide

// mle_interpolate.cu
#define _mle_interpolate_stage                   _kb_mle_interpolate_stage
#define _mle_interpolate_stage_ext               _kb_mle_interpolate_stage_ext
#define _mle_interpolate_stage_2d                _kb_mle_interpolate_stage_2d
#define _mle_interpolate_fused_2d                _kb_mle_interpolate_fused_2d
#define _mle_interpolate_shared_2d               _kb_mle_interpolate_shared_2d

// stacked_reduction.cu
#define _stacked_reduction_r0_required_temp_buffer_size  _kb_stacked_reduction_r0_required_temp_buffer_size
#define _stacked_reduction_sumcheck_round0       _kb_stacked_reduction_sumcheck_round0
#define _stacked_reduction_fold_ple              _kb_stacked_reduction_fold_ple
#define _initialize_k_rot_from_eq_segments       _kb_initialize_k_rot_from_eq_segments
#define _stacked_reduction_sumcheck_mle_round    _kb_stacked_reduction_sumcheck_mle_round
#define _stacked_reduction_sumcheck_mle_round_degenerate  _kb_stacked_reduction_sumcheck_mle_round_degenerate

// batch_ntt_small.cu — needed for DEVICE_NTT_TWIDDLES definition
#define _batch_ntt_small                         _kb_batch_ntt_small
#define _generate_device_ntt_twiddles            _kb_generate_device_ntt_twiddles

// supra/ntt.cu — NTT launcher extern "C"
#define _ct_mixed_radix_narrow                   _kb_ct_mixed_radix_narrow

// supra/ntt_params.cu — extern "C" launchers
#define _generate_all_twiddles                   _kb_generate_all_twiddles
#define _generate_partial_twiddles               _kb_generate_partial_twiddles

// supra/ntt_params.cu — __constant__ twiddle arrays (C++ globals, must rename to avoid
// collision with BabyBear supra_ntt library when both are linked into the same binary).
#define FORWARD_TWIDDLES                         FORWARD_TWIDDLES_KB
#define INVERSE_TWIDDLES                         INVERSE_TWIDDLES_KB
#define FORWARD_PARTIAL_TWIDDLES                 FORWARD_PARTIAL_TWIDDLES_KB
#define INVERSE_PARTIAL_TWIDDLES                 INVERSE_PARTIAL_TWIDDLES_KB
// supra/ntt_params.cu — __global__ kernel (C++ mangled, but rename to be safe)
#define generate_all_twiddles                    generate_all_twiddles_kb

// ── Type-name renames ──────────────────────────────────────────────────────────────────────
// CUDA mangles __global__ kernel signatures using struct/class names directly.
// BB and KB both define `struct Fp` and `struct FpExt`; without renaming, every kernel
// that takes Fp/FpExt parameters produces IDENTICAL mangled __device_stub__ symbols in
// both BB (cuda-backend) and KB (cuda_kb_all), causing link-time duplicate-symbol errors.
//
// Renaming Fp → FpKb and FpExt → FpExtKb ensures all KB device stubs have distinct names.
// This also renames the struct definitions themselves (in kb_include/fp.h and fpext.h)
// so the types are consistently called FpKb/FpExtKb throughout the KB compilation unit.
#define Fp                                       FpKb
#define FpExt                                    FpExtKb
// BatchingTracePacket appears in whir.cu kernel signatures — rename for consistency
#define BatchingTracePacket                      BatchingTracePacketKb
// FracExt appears in gkr.cu / gkr_input.cu kernel signatures
#define FracExt                                  FracExtKb

// C++ namespaces used in CUDA kernel files — rename to avoid C++ ODR / duplicate-symbol
// issues from C++ templates and inline device functions sharing the same namespace name.
#define device_ntt                               device_ntt_kb
#define fractional_sumcheck_gkr                  fractional_sumcheck_gkr_kb
#define logup_gkr_input_evaluation               logup_gkr_input_evaluation_kb
#define logup_round0                             logup_round0_kb
#define logup_zerocheck_mle                      logup_zerocheck_mle_kb
#define plain_sumcheck                           plain_sumcheck_kb
#define zerocheck_round0                         zerocheck_round0_kb
