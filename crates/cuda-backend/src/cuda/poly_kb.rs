#![cfg(feature = "koala-bear-poseidon2")]

use openvm_cuda_common::{
    copy::MemCopyD2H,
    d_buffer::DeviceBuffer,
    error::{check, CudaError},
};

use crate::{
    prelude::{D_EF, EF, F},
    KernelError,
};

#[link(name = "cuda_kb_all", kind = "static")]
extern "C" {
    fn _kb_vector_scalar_multiply_ext(vec: *mut EF, scalar: EF, length: u32) -> i32;

    fn _kb_eq_hypercube_stage_ext(out: *mut EF, x_i: EF, step: u32) -> i32;

    fn _kb_eq_hypercube_nonoverlapping_stage_ext(
        out: *mut EF,
        input: *const EF,
        x_i: EF,
        step: u32,
    ) -> i32;

    fn _kb_eq_hypercube_interleaved_stage_ext(
        out: *mut EF,
        input: *const EF,
        x_i: EF,
        step: u32,
    ) -> i32;

    // `out` must be device ptr
    fn _kb_eval_poly_ext_at_point(
        base_coeffs: *const F,
        coeff_len: usize,
        x: EF,
        out: *mut EF,
    ) -> i32;

    fn _kb_transpose_fp_to_fpext_vec(output: *mut EF, input: *const F, height: u32) -> i32;
}

pub unsafe fn eq_hypercube_stage_ext(out: *mut EF, x_i: EF, step: u32) -> Result<(), CudaError> {
    check(_kb_eq_hypercube_stage_ext(out, x_i, step))
}

pub unsafe fn eq_hypercube_nonoverlapping_stage_ext(
    out: *mut EF,
    input: *const EF,
    x_i: EF,
    step: u32,
) -> Result<(), CudaError> {
    check(_kb_eq_hypercube_nonoverlapping_stage_ext(out, input, x_i, step))
}

pub unsafe fn eq_hypercube_interleaved_stage_ext(
    out: *mut EF,
    input: *const EF,
    x_i: EF,
    step: u32,
) -> Result<(), CudaError> {
    check(_kb_eq_hypercube_interleaved_stage_ext(out, input, x_i, step))
}

/// Scalar multiplication of a vector in-place by `scalar`.
pub fn vector_scalar_multiply_ext(
    vec: &mut DeviceBuffer<EF>,
    scalar: EF,
) -> Result<(), CudaError> {
    // SAFETY: `vec` is allocated for `vec.len()` so scalar multiplication is safe.
    unsafe {
        check(_kb_vector_scalar_multiply_ext(
            vec.as_mut_ptr(),
            scalar,
            vec.len() as u32,
        ))
    }
}

/// Evaluates a `EF`-polynomial stored in coefficient form as column-major `F`-matrix at point `x`.
///
/// # Safety
/// - `base_coeffs` is in column-major form for the **base** field `F`.
/// - `base_coeffs.len() >= coeff_len * D_EF`, where `D_EF` is the degree of extension for `EF`.
pub unsafe fn eval_poly_ext_at_point_from_base(
    base_coeffs: &DeviceBuffer<F>,
    coeff_len: usize,
    x: EF,
) -> Result<EF, KernelError> {
    debug_assert!(base_coeffs.len() >= coeff_len * D_EF);
    let d_out = DeviceBuffer::<EF>::with_capacity(1);
    check(_kb_eval_poly_ext_at_point(
        base_coeffs.as_ptr(),
        coeff_len,
        x,
        d_out.as_mut_ptr(),
    ))
    .map_err(KernelError::Kernel)?;
    let out = d_out.to_host().map_err(KernelError::MemCopy)?;
    Ok(out[0])
}

/// Transposes a `DeviceBuffer<F>` as a column-major `height x D_EF` matrix into a single
/// `DeviceBuffer<EF>` of length `height`.
///
/// # Safety
/// - `input.len() == output.len() * D_EF`.
pub unsafe fn transpose_fp_to_fpext_vec(
    output: &mut DeviceBuffer<EF>,
    input: &DeviceBuffer<F>,
) -> Result<(), CudaError> {
    let height = output.len();
    debug_assert_eq!(height * D_EF, input.len());
    check(_kb_transpose_fp_to_fpext_vec(
        output.as_mut_ptr(),
        input.as_ptr(),
        height as u32,
    ))
}
