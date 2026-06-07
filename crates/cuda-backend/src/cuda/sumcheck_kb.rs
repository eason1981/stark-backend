#![cfg(feature = "koala-bear-poseidon2")]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

use crate::{poly::EqEvalSegments, prelude::EF};

#[link(name = "cuda_kb_all", kind = "static")]
extern "C" {
    fn _kb_fold_mle(
        input_matrices: *const *const EF,
        output_matrices: *const *mut EF,
        widths: *const u32,
        num_matrices: u16,
        output_height: u32,
        max_output_cells: u32,
        r_val: EF,
    ) -> i32;

    fn _kb_triangular_fold_mle(
        output: *mut EF,
        input: *const EF,
        r: EF,
        output_max_n: u32,
    ) -> i32;

    fn _kb_batch_fold_mle(
        input_matrices: *const *const EF,
        output_matrices: *const *mut EF,
        widths: *const u32,
        num_matrices: u16,
        log_output_heights: *const u8,
        max_output_cells: u32,
        r_val: EF,
    ) -> i32;
}

/// # Safety
/// - `input_matrices` must consist of pointers to device memory locations.
/// - `output_matrices` must consist of pointers to device memory locations.
pub unsafe fn fold_mle(
    input_matrices: &DeviceBuffer<*const EF>,
    output_matrices: &DeviceBuffer<*mut EF>,
    widths: &DeviceBuffer<u32>,
    num_matrices: u16,
    output_height: u32,
    max_output_cells: u32,
    r_val: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_fold_mle(
        input_matrices.as_ptr(),
        output_matrices.as_ptr(),
        widths.as_ptr(),
        num_matrices,
        output_height,
        max_output_cells,
        r_val,
    ))
}

/// Folds the segments of `input` onto `output` using random element `r`.
///
/// # Safety
/// - `output` must have max `n` equal to `output_max_n`, for total length `2 * 2^output_max_n`.
/// - `input` must have length `2 * 2^{output_max_n + 1}`.
pub unsafe fn triangular_fold_mle(
    output: &mut EqEvalSegments<EF>,
    input: &EqEvalSegments<EF>,
    r: EF,
    output_max_n: usize,
) -> Result<(), CudaError> {
    debug_assert_eq!(output.buffer.len(), 2 << output_max_n);
    debug_assert_eq!(input.buffer.len(), 4 << output_max_n);
    CudaError::from_result(_kb_triangular_fold_mle(
        output.buffer.as_mut_ptr(),
        input.buffer.as_ptr(),
        r,
        output_max_n as u32,
    ))
}

/// # Safety
/// - `input_matrices` must consist of pointers to device memory locations.
/// - `output_matrices` must consist of pointers to device memory locations.
pub unsafe fn batch_fold_mle(
    input_matrices: &DeviceBuffer<*const EF>,
    output_matrices: &DeviceBuffer<*mut EF>,
    widths: &DeviceBuffer<u32>,
    num_matrices: u16,
    log_output_heights: &DeviceBuffer<u8>,
    max_output_cells: u32,
    r_val: EF,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_batch_fold_mle(
        input_matrices.as_ptr(),
        output_matrices.as_ptr(),
        widths.as_ptr(),
        num_matrices,
        log_output_heights.as_ptr(),
        max_output_cells,
        r_val,
    ))
}
