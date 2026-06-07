use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_common::{copy::{MemCopyD2H, MemCopyH2D}, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicConstraints,
    prover::{
        fractional_sumcheck_gkr::Frac,
        stacked_pcs::{StackedLayout, StackedSlice},
        ColMajorMatrix, DeviceMultiStarkProvingKey, EvalHelper, MatrixDimensions, ProvingContext,
    },
};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use tracing::instrument;

use super::errors::InteractionGpuError;
use crate::{
    cuda::field_kernels::FieldKernels,
    data_transporter::transport_matrix_d2h_col_major,
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
};

const TASK_SIZE: u32 = 65536;

#[allow(dead_code)]
#[derive(Clone)]
pub struct TraceInteractionMeta {
    pub trace_idx: usize,
    pub air_idx: usize,
    pub layout_slices: Vec<StackedSlice>,
}

// TODO[jpw]: revisit if this function is needed
pub fn collect_trace_interactions<
    FK: FieldKernels,
    HS: GpuHashScheme<BaseField = FK::Val, ExtField = FK::ValExt>,
>(
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    layout: &StackedLayout,
) -> Vec<Option<TraceInteractionMeta>> {
    // Pre-group layout slices by trace to avoid repeated scans later.
    let mut slices_by_trace: Vec<Vec<(usize, StackedSlice)>> =
        vec![Vec::new(); ctx.per_trace.len()];
    for &(trace_idx, interaction_idx, ref slice) in &layout.sorted_cols {
        if let Some(entries) = slices_by_trace.get_mut(trace_idx) {
            entries.push((interaction_idx, *slice));
        }
    }

    ctx.per_trace
        .iter()
        .enumerate()
        .map(|(trace_idx, (air_idx, _))| {
            let vk = &pk.per_air[*air_idx].vk;
            if !vk.has_interaction() {
                return None;
            }

            let mut layout_entries = vec![None; vk.num_interactions()];
            for (interaction_idx, slice) in &slices_by_trace[trace_idx] {
                if let Some(slot) = layout_entries.get_mut(*interaction_idx) {
                    *slot = Some(*slice);
                }
            }

            let layout_slices = layout_entries
                .into_iter()
                .enumerate()
                .map(|(idx, maybe_slice)| {
                    maybe_slice.unwrap_or_else(|| {
                        panic!(
                            "missing stacked slice for interaction {} of trace {}",
                            idx, trace_idx
                        )
                    })
                })
                .collect_vec();

            Some(TraceInteractionMeta {
                trace_idx,
                air_idx: *air_idx,
                layout_slices,
            })
        })
        .collect()
}

/// Evaluate interactions from trace evaluation matrices to get (p, q) fractional sumcheck input.
/// Returns leaves buffer (WITHOUT alpha applied) and alpha value to be applied in first tree layer.
#[instrument(name = "prover.rap_constraints.logup_gkr.input_evals", skip_all)]
pub fn log_gkr_input_evals<
    FK: FieldKernels,
    HS: GpuHashScheme<BaseField = FK::Val, ExtField = FK::ValExt>,
>(
    trace_interactions: &[Option<TraceInteractionMeta>],
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    l_skip: usize,
    alpha_logup: FK::ValExt,
    d_challenges: &DeviceBuffer<FK::ValExt>,
    total_leaves: usize,
) -> Result<(DeviceBuffer<Frac<FK::ValExt>>, FK::ValExt), InteractionGpuError> {
    if trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok((DeviceBuffer::new(), alpha_logup));
    }

    let leaves = DeviceBuffer::<Frac<FK::ValExt>>::with_capacity(total_leaves);
    leaves.fill_zero()?;
    let null_preprocessed = DeviceBuffer::<FK::Val>::new();

    let mut d_partition_ptrs = DeviceBuffer::<u64>::new();
    let mut tmp = DeviceBuffer::<Frac<FK::ValExt>>::new();
    for meta in trace_interactions.iter().flatten() {
        let air_ctx = &ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];

        let preprocessed_matrix = pk_air
            .preprocessed_data
            .as_ref()
            .map(|committed| &committed.trace);

        let mut partitioned_main = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
        for committed in &air_ctx.cached_mains {
            partitioned_main.push(&committed.trace);
        }
        partitioned_main.push(&air_ctx.common_main);

        let rules = &pk_air.other_data.interaction_rules;
        let num_interactions = pk_air.vk.symbolic_constraints.interactions.len();

        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.buffer())
            .unwrap_or(&null_preprocessed);
        let d_public_values = if air_ctx.public_values.is_empty() {
            DeviceBuffer::<FK::Val>::new()
        } else {
            air_ctx.public_values.to_device()?
        };

        let height = air_ctx.height();
        debug_assert_eq!(height, partitioned_main[0].height());
        let partition_ptrs = partitioned_main
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect_vec();
        if partition_ptrs.len() > d_partition_ptrs.len() {
            d_partition_ptrs = DeviceBuffer::with_capacity(partition_ptrs.len());
        }
        partition_ptrs.copy_to(&mut d_partition_ptrs)?;

        let buffer_size = rules.inner.buffer_size;
        // TODO[jpw]: remove magic 10
        let is_global = buffer_size > 10;
        let intermediates = if is_global {
            DeviceBuffer::<FK::ValExt>::with_capacity((TASK_SIZE as usize) * buffer_size as usize)
        } else {
            DeviceBuffer::<FK::ValExt>::with_capacity(1)
        };

        let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1);

        let slice = meta.layout_slices.first().unwrap();
        if slice.col_idx != 0 {
            return Err(InteractionGpuError::Layout);
        }
        let dst_offset = slice.row_idx;
        let lifted_height = max(height, 1 << l_skip);
        debug_assert_eq!(slice.len(l_skip), lifted_height);
        // SAFETY: by definition of interactions stacked layout, `leaves` has enough capacity
        let leaves_ptr = unsafe { leaves.as_mut_ptr().add(dst_offset) };

        let trace_output = if height != lifted_height {
            let required = height * num_interactions;
            if required > tmp.len() {
                tmp = DeviceBuffer::with_capacity(required);
            }
            tmp.as_mut_ptr()
        } else {
            leaves_ptr
        };
        // Closure that computes the interaction fracs on CPU (interaction-major layout),
        // returning the Vec. Used both for the CPU fallback path and for the optional
        // GPU-vs-CPU verification (SWIRL_VERIFY_INPUT_EVAL).
        let compute_cpu_fracs = || -> Result<Vec<Frac<FK::ValExt>>, InteractionGpuError> {
            let symbolic = SymbolicConstraints::from(&pk_air.vk.symbolic_constraints);
            let eval_helper = EvalHelper {
                constraints_dag: &pk_air.vk.symbolic_constraints.constraints,
                interactions: symbolic.interactions.clone(),
                public_values: air_ctx.public_values.clone(),
                preprocessed_trace: None, // preprocessed handled via row_parts ordering
                needs_next: pk_air.vk.params.need_rot,
                constraint_degree: 0,
            };

            // Download challenges: d_challenges = [alpha_logup, beta^0, beta^1, ...].
            // eval_interactions expects beta_pows = [beta^0, beta^1, ...], so skip alpha at index 0.
            let challenges_host = d_challenges.to_host()
                .map_err(InteractionGpuError::Copy)?;
            let beta_pows_host = &challenges_host[1..];

            // Download all trace matrices to CPU (in EvalHelper.view_mats order).
            // cpu_mats: (ColMajorMatrix<FK::Val>, is_rotated) - same order as EvalHelper.view_mats
            let mut cpu_mats: Vec<(ColMajorMatrix<FK::Val>, bool)> = Vec::new();
            if let Some(prep) = preprocessed_matrix {
                let m = transport_matrix_d2h_col_major(prep).map_err(InteractionGpuError::Copy)?;
                let needs_rot = pk_air.vk.params.need_rot;
                cpu_mats.push((m.clone(), false));
                if needs_rot { cpu_mats.push((m, true)); }
            }
            let needs_rot = pk_air.vk.params.need_rot;
            for committed in &air_ctx.cached_mains {
                let m = transport_matrix_d2h_col_major(&committed.trace).map_err(InteractionGpuError::Copy)?;
                cpu_mats.push((m.clone(), false));
                if needs_rot { cpu_mats.push((m, true)); }
            }
            {
                let m = transport_matrix_d2h_col_major(&air_ctx.common_main).map_err(InteractionGpuError::Copy)?;
                cpu_mats.push((m.clone(), false));
                if needs_rot { cpu_mats.push((m, true)); }
            }

            // Evaluate interactions for each row on CPU.
            // row_parts[0] = selectors; row_parts[1..] = matrix columns (all as FK::Val).
            // Output MUST match the GPU kernel layout (interaction-major):
            //   cpu_fracs[interaction_idx * height + row]   (see gkr_input.cu out_idx).
            let mut cpu_fracs: Vec<Frac<FK::ValExt>> =
                vec![Frac { p: FK::ValExt::ZERO, q: FK::ValExt::ZERO }; height * num_interactions];
            for row in 0..height {
                let is_first = FK::Val::from_bool(row == 0);
                let is_transition = FK::Val::from_bool(row != height - 1);
                let is_last = FK::Val::from_bool(row == height - 1);
                let mut row_parts: Vec<Vec<FK::Val>> = vec![vec![is_first, is_transition, is_last]];
                for (mat, is_rot) in &cpu_mats {
                    let offset = if *is_rot { 1 } else { 0 };
                    let row_idx = (row + offset) % mat.height();
                    row_parts.push((0..mat.width()).map(|j| mat.column(j)[row_idx]).collect());
                }
                let interaction_evals =
                    eval_helper.eval_interactions::<FK::Val, FK::ValExt>(&row_parts, beta_pows_host);
                for (interaction_idx, (numer_base, denom)) in interaction_evals.into_iter().enumerate() {
                    cpu_fracs[interaction_idx * height + row] =
                        Frac { p: FK::ValExt::from(numer_base), q: denom };
                }
            }
            Ok(cpu_fracs)
        };

        // Run the GPU input-eval kernel into `trace_output`.
        let run_gpu = |trace_output: *mut Frac<FK::ValExt>| -> Result<(), InteractionGpuError> {
            unsafe {
                FK::logup_gkr_input_eval(
                    is_global,
                    trace_output,
                    d_preprocessed,
                    &d_partition_ptrs,
                    &d_public_values,
                    d_challenges,
                    &intermediates,
                    &rules.inner.d_rules,
                    &rules.inner.d_used_nodes,
                    &rules.d_pair_idxs,
                    height as u32,
                    num_rows_per_tile as u32,
                )?;
            }
            Ok(())
        };

        use openvm_cuda_common::d_buffer::DeviceBuffer as DevBuf;
        let verify_input_eval = std::env::var("SWIRL_VERIFY_INPUT_EVAL").is_ok();

        if FK::use_cpu_gkr_input_eval() {
            // CPU fallback path: compute on CPU and upload to trace_output.
            let cpu_fracs = compute_cpu_fracs()?;
            let mut trace_output_buf = unsafe {
                DevBuf::<Frac<FK::ValExt>>::from_raw_parts(
                    trace_output as *mut Frac<FK::ValExt>,
                    height * num_interactions,
                )
            };
            cpu_fracs.as_slice().copy_to(&mut trace_output_buf).map_err(InteractionGpuError::Copy)?;
            std::mem::forget(trace_output_buf);

            // Optional: run GPU into a scratch buffer and compare against CPU (proof unaffected).
            if verify_input_eval {
                use openvm_cuda_common::copy::MemCopyD2H;
                let scratch = DevBuf::<Frac<FK::ValExt>>::with_capacity(height * num_interactions);
                run_gpu(scratch.as_mut_ptr())?;
                let gpu_fracs: Vec<Frac<FK::ValExt>> =
                    scratch.to_host().map_err(InteractionGpuError::Copy)?;
                let mut mism = 0usize;
                let mut first = None;
                for i in 0..(height * num_interactions) {
                    if gpu_fracs[i].p != cpu_fracs[i].p || gpu_fracs[i].q != cpu_fracs[i].q {
                        mism += 1;
                        if first.is_none() {
                            first = Some((i, i / height, i % height));
                        }
                    }
                }
                if mism > 0 {
                    let (i, int_idx, row) = first.unwrap();
                    tracing::warn!(
                        "INPUT_EVAL_VERIFY air_idx={} height={height} num_int={num_interactions}: \
                         {mism}/{} mismatch; first at idx={i} (interaction={int_idx}, row={row}) \
                         CPU=({:?},{:?}) GPU=({:?},{:?})",
                        meta.air_idx, height * num_interactions,
                        cpu_fracs[i].p, cpu_fracs[i].q, gpu_fracs[i].p, gpu_fracs[i].q
                    );
                } else {
                    tracing::warn!(
                        "INPUT_EVAL_VERIFY air_idx={} height={height} num_int={num_interactions}: OK ({} fracs match)",
                        meta.air_idx, height * num_interactions
                    );
                }
            }
        } else {
            // GPU path.
            run_gpu(trace_output)?;
        }
        if height != lifted_height {
            debug_assert_eq!(lifted_height % height, 0);
            debug_assert!(!tmp.is_empty());
            let norm_factor_denom = lifted_height / height;
            let norm_factor = FK::Val::from_usize(norm_factor_denom).inverse();
            unsafe {
                // SAFETY: scaling within buffer length
                // TODO: FK::frac_vector_scalar_multiply_ext_fp — add this method to FieldKernels trait
                // Scale tmp in-place (frac_vector_scalar_multiply uses DeviceBuffer ref)
                {
                    let tmp_slice_len = tmp.len();
                    let _ = tmp_slice_len; // avoid unused var
                    FK::frac_vector_scalar_multiply_ext_fp(&mut tmp, norm_factor)?;
                }
                // SAFETY: stacked interaction layout is defined with respect to lifted height, and
                // the vertical-repeat kernel guards rounded-up tail threads beyond lifted_height.
                // TODO: FK::frac_matrix_vertically_repeat — add this method to FieldKernels trait
                FK::frac_matrix_vertically_repeat(
                    leaves_ptr,
                    &tmp,
                    num_interactions as u32,
                    lifted_height as u32,
                    height as u32,
                )?;
            }
        }
    }

    // NOTE: alpha is NO LONGER applied here - it will be fused into the first tree layer
    // in fractional_sumcheck_gpu for better performance (eliminates one memory pass)
    Ok((leaves, alpha_logup))
}
