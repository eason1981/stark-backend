use itertools::Itertools;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, SymbolicConstraints, SymbolicDagBuilder,
        SymbolicExpressionDag,
    },
    poly_common::{evals_eq_hypercube_serial, UnivariatePoly},
    prover::{
        fractional_sumcheck_gkr::Frac,
        sumcheck::{sumcheck_round0_deg, sumcheck_uni_round0_poly},
        ColMajorMatrix, DeviceStarkProvingKey, EvalHelper, StridedColMajorMatrixView,
    },
};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use tracing::{debug, warn};

use super::errors::Round0EvalError;
use crate::{
    cuda::field_kernels::FieldKernels,
    data_transporter::transport_matrix_d2h_col_major,
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
    base::DeviceMatrix,
};

const ROUND0_COSET_PARALLEL_THRESHOLD: u32 = 32768;
const MAX_LOCKSTEP_NUM_COSETS: u32 = 4;

fn uses_round0_coset_parallel(num_x: u32, skip_domain: u32) -> bool {
    num_x.saturating_mul(skip_domain) < ROUND0_COSET_PARALLEL_THRESHOLD
}

fn validate_round0_num_cosets(
    num_x: u32,
    skip_domain: u32,
    num_cosets: u32,
) -> Result<(), Round0EvalError> {
    if num_cosets > MAX_LOCKSTEP_NUM_COSETS && !uses_round0_coset_parallel(num_x, skip_domain) {
        return Err(CudaError::new(1).into());
    }
    Ok(())
}

/// Evaluate plain AIR constraints (not interactions) for a single AIR, given prepared trace input.
///
/// `num_cosets` should equal `constraint_degree - 1` because we evaluate the quotient polynomial.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_gpu<
    FK: FieldKernels,
    HS: GpuHashScheme<BaseField = FK::Val, ExtField = FK::ValExt>,
>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    selectors_cube: &DeviceBuffer<FK::Val>,
    main_parts: &DeviceBuffer<*const FK::Val>,
    public_values: &DeviceBuffer<FK::Val>,
    eq_cube: *const FK::ValExt,
    lambda_pows: &DeviceBuffer<FK::ValExt>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: FK::Val,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<FK::ValExt>, Round0EvalError> {
    let constraints_dag = &pk.vk.symbolic_constraints;
    if constraints_dag.constraints.constraint_idx.is_empty() || num_cosets == 0 {
        // No plain AIR constraints, return empty buffer
        return Ok(DeviceBuffer::new());
    }
    validate_round0_num_cosets(num_x, skip_domain, num_cosets)?;

    let rules = &pk.other_data.zerocheck_round0;

    let buffer_size: u32 = rules.inner.buffer_size;
    // TODO: FK::zerocheck_r0_intermediates_buffer_size — add this method to FieldKernels trait
    let intermed_capacity = unsafe {
        FK::zerocheck_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<FK::Val>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<FK::Val>::new()
    };

    // TODO: FK::zerocheck_r0_temp_sums_buffer_size — add this method to FieldKernels trait
    let temp_sums_buffer_capacity = unsafe {
        FK::zerocheck_r0_temp_sums_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<FK::ValExt>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes = intermed_capacity * size_of::<FK::Val>()
        + temp_sums_buffer_capacity * size_of::<FK::ValExt>();
    if used_temp_bytes > max_temp_bytes {
        // We do not error if the required bytes is greater than the requested max, but this may
        // lead to unexpected peak memory usage.
        warn!("zerocheck used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})");
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut sp_evals =
        DeviceBuffer::<FK::ValExt>::with_capacity(num_cosets as usize * skip_domain as usize);
    // SAFETY:
    // - No bounds checks are done in this kernel. It fully assumes that the Rules are trusted and
    //   all nodes are valid.
    // TODO: FK::zerocheck_ntt_eval_constraints — add this method to FieldKernels trait
    unsafe {
        FK::zerocheck_ntt_eval_constraints(
            &mut temp_sums_buffer,
            &mut sp_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            lambda_pows,
            public_values,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            buffer_size,
            &mut intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
        )?;
    }

    Ok(sp_evals)
}

/// Evaluate interaction constraints (excluding plain AIR constraints) for a single AIR, given
/// prepared trace input.
///
/// `constraints` includes interaction expressions for the AIR.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_interactions_gpu<
    FK: FieldKernels,
    HS: GpuHashScheme<BaseField = FK::Val, ExtField = FK::ValExt>,
>(
    pk: &DeviceStarkProvingKey<GenericGpuBackend<HS>>,
    symbolic: &SymbolicConstraints<FK::Val>,
    selectors_cube: &DeviceBuffer<FK::Val>,
    main_parts: &DeviceBuffer<*const FK::Val>,
    public_values: &DeviceBuffer<FK::Val>,
    eq_cube: *const FK::ValExt,
    beta_pows: &[FK::ValExt],
    eq_3bs: &[FK::ValExt],
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: FK::Val,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<Frac<FK::ValExt>>, Round0EvalError> {
    // Check if this trace has interactions
    if eq_3bs.is_empty() {
        return Ok(DeviceBuffer::new());
    }
    validate_round0_num_cosets(num_x, skip_domain, num_cosets)?;
    let large_domain = num_cosets * skip_domain;

    // We create a new "interactions DAG" where the new .constraints are the interaction [count,
    // message_0, message_1, ..] expressions themselves, while the .interactions are empty
    // We track the indices with InteractionNode

    // Copied from build_symbolic_constraints_dag to handle sorting of constraints
    // NOTE: For logup round0, the kernel uses weights indexed by rule_idx, not constraint_idx.
    // So we deduplicate constraint_idx and use dag_idx_to_rule_idx for weight mapping.
    let (rules, d_numer_weights, d_denom_weights, denom_sum_init) = {
        let mut dag_builder = SymbolicDagBuilder::new();
        let mut sorted_used_dag_idxs = Vec::new();
        for interaction in &symbolic.interactions {
            let count = dag_builder.add_expr(&interaction.count);
            sorted_used_dag_idxs.push(count);
            sorted_used_dag_idxs.extend(
                interaction
                    .message
                    .iter()
                    .map(|field_expr| dag_builder.add_expr(field_expr)),
            );
        }
        sorted_used_dag_idxs.sort();
        // Deduplicate for the dag since logup round0 kernel doesn't use used_nodes
        sorted_used_dag_idxs.dedup();
        let dag = SymbolicExpressionDag {
            nodes: dag_builder.nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let rules = SymbolicRulesGpu::new(&dag, true);
        let mut numer_weights = vec![FK::ValExt::ZERO; rules.rules.len()];
        let mut denom_weights = vec![FK::ValExt::ZERO; rules.rules.len()];
        let mut denom_sum_init = FK::ValExt::ZERO;
        for (interaction_idx, interaction) in symbolic.interactions.iter().enumerate() {
            // CAUTION: an expression node could be used in multiple interactions, and might even be
            // used as `count` in one, but message field in another. We only care about their
            // weighted sum with eq_3b, so we compute the weights ahead of time.
            let count_dag_idx =
                dag_builder.expr_to_idx[&(&interaction.count as *const SymbolicExpression<_>)];
            let count_rule_idx = rules.dag_idx_to_rule_idx[&count_dag_idx];
            numer_weights[count_rule_idx] += eq_3bs[interaction_idx];
            denom_sum_init += eq_3bs[interaction_idx]
                * beta_pows[interaction.message.len()]
                * FK::Val::from_u32(interaction.bus_index as u32 + 1);

            for (message_idx, message) in interaction.message.iter().enumerate() {
                let message_dag_idx =
                    dag_builder.expr_to_idx[&(message as *const SymbolicExpression<_>)];
                let message_rule_idx = rules.dag_idx_to_rule_idx[&message_dag_idx];
                denom_weights[message_rule_idx] +=
                    eq_3bs[interaction_idx] * beta_pows[message_idx];
            }
        }
        let d_numer_weights = numer_weights.to_device()?;
        let d_denom_weights = denom_weights.to_device()?;
        (rules, d_numer_weights, d_denom_weights, denom_sum_init)
    };

    let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    // TODO: FK::logup_r0_intermediates_buffer_size — add this method to FieldKernels trait
    let intermed_capacity = unsafe {
        FK::logup_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup_r0:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<FK::Val>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<FK::Val>::new()
    };

    // TODO: FK::logup_r0_temp_sums_buffer_size — add this method to FieldKernels trait
    let temp_sums_buffer_capacity = unsafe {
        FK::logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
    };
    debug!("logup_r0:tmp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer =
        DeviceBuffer::<Frac<FK::ValExt>>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes = intermed_capacity * size_of::<FK::Val>()
        + temp_sums_buffer_capacity * size_of::<Frac<FK::ValExt>>();
    if used_temp_bytes > max_temp_bytes {
        warn!(
            "logup_round0 used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})"
        );
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut s_evals = DeviceBuffer::<Frac<FK::ValExt>>::with_capacity(large_domain as usize);

    // TODO: FK::logup_bary_eval_interactions_round0 — add this method to FieldKernels trait
    unsafe {
        FK::logup_bary_eval_interactions_round0(
            &mut temp_sums_buffer,
            &mut s_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            public_values,
            &d_numer_weights,
            &d_denom_weights,
            denom_sum_init,
            &d_rules,
            buffer_size,
            &mut intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
        )?;
    }

    // ── Debug: print first few s_evals entries to compare KB GPU vs expected ──
    #[cfg(debug_assertions)]
    if std::env::var("SWIRL_DEBUG_ROUND0").is_ok() {
        use openvm_cuda_common::copy::MemCopyD2H;
        let host = s_evals.to_host().unwrap_or_default();
        tracing::warn!(
            "round0 debug: denom_sum_init={:?} height={} num_cosets={} skip_domain={}",
            denom_sum_init, height, num_cosets, skip_domain,
        );
        let nw = d_numer_weights.to_host().unwrap_or_default();
        let dw = d_denom_weights.to_host().unwrap_or_default();
        tracing::warn!("  numer_weights[..3]={:?}", &nw[..nw.len().min(3)]);
        tracing::warn!("  denom_weights[..3]={:?}", &dw[..dw.len().min(3)]);
        for (i, f) in host.iter().take(4).enumerate() {
            tracing::warn!("  s_evals[{i}] = p={:?} q={:?}", f.p, f.q);
        }
    }

    Ok(s_evals)
}

/// CPU fallback for `evaluate_round0_interactions_gpu`.
///
/// Downloads the GPU trace to CPU and uses `sumcheck_uni_round0_poly` to compute the
/// correct batch MLE of interaction polynomials. Used for KB where the GPU kernel
/// `_kb_logup_bary_eval_interactions_round0` produces wrong results.
///
/// Returns `None` if there are no interactions (`eq_3bs` is empty).
pub fn evaluate_round0_interactions_cpu_fallback<FK: FieldKernels>(
    constraints_dag: &SymbolicExpressionDag<FK::Val>,
    symbolic: &SymbolicConstraints<FK::Val>,
    cached_mains: &[&DeviceMatrix<FK::Val>],
    common_main: &DeviceMatrix<FK::Val>,
    xi_chunk: &[FK::ValExt],
    needs_next: bool,
    beta_pows: &[FK::ValExt],
    eq_3bs: &[FK::ValExt],
    l_skip: usize,
    n: isize,
    height: usize,
    num_cosets: usize,
    omega_root: FK::Val,
    public_values: &[FK::Val],
) -> Result<Option<[UnivariatePoly<FK::ValExt>; 2]>, Round0EvalError> {
    let _ = omega_root; // used structurally but not needed for the CPU sumcheck path
    if eq_3bs.is_empty() || num_cosets == 0 {
        return Ok(None);
    }

    // Download ALL GPU matrices to CPU: cached_mains then common_main (matches EvalHelper partition order)
    let mut cpu_mats = Vec::with_capacity(cached_mains.len() + 1);
    for cm in cached_mains {
        cpu_mats.push(transport_matrix_d2h_col_major(*cm)?);
    }
    cpu_mats.push(transport_matrix_d2h_col_major(common_main)?);

    // Compute eq(xi, x) evaluations for the 2^n_lift hypercube in natural order.
    // NOTE: eq_xis.layers[n_lift] from new_rev_with_kernels stores eq(xi, bit_rev(x)) which
    // is the GPU's bit-reversed convention. Since the CPU GKR runs on natural-order leaves,
    // we must use natural-order eq here so the batch MLE matches the GKR claim.
    let n_lift = xi_chunk.len(); // = n.max(0)
    let eq_cube: Vec<FK::ValExt> = evals_eq_hypercube_serial(xi_chunk);

    // Build selector matrix: col0=is_first, col1=is_transition, col2=is_last
    let mut sel_vals = FK::Val::zero_vec(3 * height);
    if height > 0 {
        sel_vals[0] = FK::Val::ONE;
        for i in 0..height.saturating_sub(1) {
            sel_vals[height + i] = FK::Val::ONE;
        }
        sel_vals[3 * height - 1] = FK::Val::ONE;
    }
    let sel_mat = ColMajorMatrix::new(sel_vals, 3);

    let eval_helper = EvalHelper {
        constraints_dag,
        interactions: symbolic.interactions.clone(),
        public_values: public_values.to_vec(),
        preprocessed_trace: None,
        needs_next,
        constraint_degree: 0,
    };

    // Build mats list: selector first, then all cached_mains + common_main (with rotation if needed)
    let sel_view: StridedColMajorMatrixView<FK::Val> = sel_mat.as_view().into();
    let mut mats: Vec<(StridedColMajorMatrixView<FK::Val>, bool)> = vec![(sel_view, false)];
    for cpu_mat in &cpu_mats {
        let mat_view: StridedColMajorMatrixView<FK::Val> = cpu_mat.as_view().into();
        mats.push((mat_view, false));
        if needs_next {
            mats.push((mat_view, true));
        }
    }

    let [mut numer_poly, denom_poly] = sumcheck_uni_round0_poly::<FK::Val, FK::ValExt, _, 2>(
        l_skip,
        n_lift,
        num_cosets,
        &mats,
        |_z, x, row_parts| {
            let eq = eq_cube[x];
            let [numer, denom] = eval_helper.acc_interactions(row_parts, beta_pows, eq_3bs);
            [eq * numer, eq * denom]
        },
    );

    // When n < 0 the trace is shorter than the lifted domain (cyclic repeat); apply norm factor
    if n.is_negative() {
        let norm_factor = FK::Val::from_u32(1 << n.unsigned_abs()).inverse();
        for c in numer_poly.coeffs_mut() {
            *c *= norm_factor;
        }
    }

    Ok(Some([numer_poly, denom_poly]))
}

/// CPU fallback for the round-0 zerocheck constraint polynomial (KB GPU kernel is buggy
/// for l_skip>0 coset handling). Mirrors the CPU reference `sp_0_zerochecks`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_cpu_fallback<FK: FieldKernels>(
    constraints_dag: &SymbolicExpressionDag<FK::Val>,
    symbolic: &SymbolicConstraints<FK::Val>,
    cached_mains: &[&DeviceMatrix<FK::Val>],
    common_main: &DeviceMatrix<FK::Val>,
    xi_chunk: &[FK::ValExt],
    needs_next: bool,
    lambda_pows: &[FK::ValExt],
    l_skip: usize,
    height: usize,
    constraint_deg: usize,
    public_values: &[FK::Val],
) -> Result<Option<UnivariatePoly<FK::ValExt>>, Round0EvalError> {
    // deg 0/1: zerocheck sp_0 must be identically zero (no constraint poly).
    if constraint_deg <= 1 {
        return Ok(None);
    }
    let num_cosets = constraint_deg - 1;

    let mut cpu_mats = Vec::with_capacity(cached_mains.len() + 1);
    for cm in cached_mains {
        cpu_mats.push(transport_matrix_d2h_col_major(*cm)?);
    }
    cpu_mats.push(transport_matrix_d2h_col_major(common_main)?);

    let n_lift = xi_chunk.len();
    let eq_cube: Vec<FK::ValExt> = evals_eq_hypercube_serial(xi_chunk);

    let mut sel_vals = FK::Val::zero_vec(3 * height);
    if height > 0 {
        sel_vals[0] = FK::Val::ONE;
        for i in 0..height.saturating_sub(1) {
            sel_vals[height + i] = FK::Val::ONE;
        }
        sel_vals[3 * height - 1] = FK::Val::ONE;
    }
    let sel_mat = ColMajorMatrix::new(sel_vals, 3);

    let eval_helper = EvalHelper {
        constraints_dag,
        interactions: symbolic.interactions.clone(),
        public_values: public_values.to_vec(),
        preprocessed_trace: None,
        needs_next,
        constraint_degree: constraint_deg as u8,
    };

    let sel_view: StridedColMajorMatrixView<FK::Val> = sel_mat.as_view().into();
    let mut mats: Vec<(StridedColMajorMatrixView<FK::Val>, bool)> = vec![(sel_view, false)];
    for cpu_mat in &cpu_mats {
        let mat_view: StridedColMajorMatrixView<FK::Val> = cpu_mat.as_view().into();
        mats.push((mat_view, false));
        if needs_next {
            mats.push((mat_view, true));
        }
    }

    // q(Z) = sum_x eq(xi,x) * C(Z,x) / (Z^{2^l_skip} - 1) on (deg-1) cosets.
    let [q] = sumcheck_uni_round0_poly::<FK::Val, FK::ValExt, _, 1>(
        l_skip,
        n_lift,
        num_cosets,
        &mats,
        |z, x, row_parts| {
            let eq = eq_cube[x];
            let c = eval_helper.acc_constraints(row_parts, lambda_pows);
            let zerofier = z.exp_power_of_2(l_skip) - FK::Val::ONE;
            [eq * c * zerofier.inverse()]
        },
    );

    // sp_0 = (Z^{2^l_skip} - 1) * q
    let sp_0_deg = sumcheck_round0_deg(l_skip, constraint_deg);
    let skip = 1usize << l_skip;
    let coeffs: Vec<FK::ValExt> = (0..=sp_0_deg)
        .map(|i| {
            let mut c = -*q.coeffs().get(i).unwrap_or(&FK::ValExt::ZERO);
            if i >= skip {
                c = c + *q.coeffs().get(i - skip).unwrap_or(&FK::ValExt::ZERO);
            }
            c
        })
        .collect();

    Ok(Some(UnivariatePoly::new(coeffs)))
}
