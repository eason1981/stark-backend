use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use tracing::debug;

use crate::{
    cuda::{field_kernels::FieldKernels, logup_zerocheck::MainMatrixPtrs},
    error::KernelError,
    ConstraintOnlyRules, InteractionEvalRules,
};

// We interpolate first, so access to vars is free and doesn't need to be buffered
const ZEROCHECK_BUFFER_VARS: bool = false;
const CUDA_GRID_Y_DIM_MAX: u32 = 65535;

fn validate_mle_num_x(num_x: u32) -> Result<(), KernelError> {
    if num_x == 0 || num_x > CUDA_GRID_Y_DIM_MAX {
        return Err(CudaError::new(1).into());
    }
    Ok(())
}

/// Evaluate MLE constraints on GPU.
///
/// Takes device pointers directly, avoiding H2D copies when data is already on device.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_constraints_gpu<FK: FieldKernels>(
    eq_xi_ptr: *const FK::ValExt,
    sels_ptr: *const FK::ValExt,
    prep_ptr: MainMatrixPtrs<FK::ValExt>,
    d_main_ptrs: &DeviceBuffer<MainMatrixPtrs<FK::ValExt>>,
    public_ptr: *const FK::Val,
    lambda_pows: &DeviceBuffer<FK::ValExt>,
    rules: &ConstraintOnlyRules<ZEROCHECK_BUFFER_VARS>,
    num_y: u32,
    num_x: u32,
) -> Result<DeviceBuffer<FK::ValExt>, KernelError> {
    validate_mle_num_x(num_x)?;
    let buffer_size = rules.inner.buffer_size;
    // TODO: FK::zerocheck_mle_intermediates_buffer_size — add this method to FieldKernels trait
    let intermed_capacity =
        unsafe { FK::zerocheck_mle_intermediates_buffer_size(buffer_size, num_x, num_y) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<FK::ValExt>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<FK::ValExt>::new()
    };
    // TODO: FK::zerocheck_mle_temp_sums_buffer_size — add this method to FieldKernels trait
    let temp_sums_buffer_capacity =
        unsafe { FK::zerocheck_mle_temp_sums_buffer_size(num_x, num_y) };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<FK::ValExt>::with_capacity(temp_sums_buffer_capacity);
    let mut output = DeviceBuffer::<FK::ValExt>::with_capacity(num_x as usize);

    // TODO: FK::zerocheck_eval_mle — add this method to FieldKernels trait
    unsafe {
        FK::zerocheck_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            d_main_ptrs.as_ptr(),
            lambda_pows.as_ptr(),
            lambda_pows.len(),
            public_ptr,
            rules.inner.d_rules.as_raw_ptr(),
            rules.inner.d_rules.len(),
            rules.inner.d_used_nodes.as_ptr(),
            rules.inner.d_used_nodes.len(),
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
        )?;
    }
    Ok(output)
}

/// Evaluate MLE interactions on GPU.
///
/// Takes device pointers directly, avoiding H2D copies when data is already on device.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_interactions_gpu<FK: FieldKernels>(
    eq_xi_ptr: *const FK::ValExt,
    sels_ptr: *const FK::ValExt,
    prep_ptr: MainMatrixPtrs<FK::ValExt>,
    d_main_ptrs: &DeviceBuffer<MainMatrixPtrs<FK::ValExt>>,
    public_ptr: *const FK::Val,
    challenges_ptr: *const FK::ValExt,
    eq_3bs_ptr: *const FK::ValExt,
    rules: &InteractionEvalRules,
    num_y: u32,
    num_x: u32,
) -> Result<DeviceBuffer<Frac<FK::ValExt>>, KernelError> {
    validate_mle_num_x(num_x)?;
    let buffer_size = rules.inner.buffer_size;
    // TODO: FK::logup_mle_intermediates_buffer_size — add this method to FieldKernels trait
    let intermed_capacity =
        unsafe { FK::logup_mle_intermediates_buffer_size(buffer_size, num_x, num_y) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<FK::ValExt>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<FK::ValExt>::new()
    };
    // TODO: FK::logup_mle_temp_sums_buffer_size — add this method to FieldKernels trait
    let temp_sums_buffer_capacity =
        unsafe { FK::logup_mle_temp_sums_buffer_size(num_x, num_y) };
    debug!("logup:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer =
        DeviceBuffer::<Frac<FK::ValExt>>::with_capacity(temp_sums_buffer_capacity);
    let mut output = DeviceBuffer::<Frac<FK::ValExt>>::with_capacity(num_x as usize);

    // TODO: FK::logup_eval_mle — add this method to FieldKernels trait
    unsafe {
        FK::logup_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            d_main_ptrs.as_ptr(),
            challenges_ptr,
            eq_3bs_ptr,
            public_ptr,
            rules.inner.d_rules.as_raw_ptr(),
            rules.inner.d_used_nodes.as_ptr(),
            rules.d_pair_idxs.as_ptr(),
            rules.inner.d_used_nodes.len(),
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
        )?;
    }
    Ok(output)
}
