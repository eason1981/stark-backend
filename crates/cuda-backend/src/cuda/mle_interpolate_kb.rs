#![cfg(feature = "koala-bear-poseidon2")]

use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{check, CudaError},
};

use crate::prelude::EF;

#[link(name = "cuda_kb_all", kind = "static")]
extern "C" {
    fn _kb_mle_interpolate_stage_ext(
        buffer: *mut EF,
        total_len: usize,
        step: u32,
        is_eval_to_coeff: bool,
    ) -> i32;
}

/// Same as [mle_interpolate_stage] for extension field `EF`.
///
/// # Safety
/// - The `buffer` must be allocated and initialized on device in the default stream.
/// - `step` must be a power of two and less than or equal to half the length of the buffer.
pub unsafe fn mle_interpolate_stage_ext(
    buffer: &mut DeviceBuffer<EF>,
    step: u32,
    is_eval_to_coeff: bool,
) -> Result<(), CudaError> {
    check(_kb_mle_interpolate_stage_ext(
        buffer.as_mut_ptr(),
        buffer.len(),
        step,
        is_eval_to_coeff,
    ))
}
