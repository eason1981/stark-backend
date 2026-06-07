//! Rust FFI bindings for the KoalaBear supra NTT library (`supra_ntt_kb`).
//!
//! All symbols are prefixed with `_kb_` to avoid link conflicts with the
//! BabyBear `supra_ntt` library.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use p3_koala_bear::KoalaBear;

/// Maximum log2 domain size supported by the KoalaBear NTT.
/// KoalaBear has 2-adicity 24 (P-1 = 2^24 * 127).
pub const MAX_CUDA_KB_NTT_LOG_DOMAIN_SIZE: u32 = 24;

// ── FFI declarations ──────────────────────────────────────────────────────────

#[link(name = "cuda_kb_all", kind = "static")]
extern "C" {
    fn _kb_generate_all_twiddles(twiddles: *mut std::ffi::c_void, inverse: bool) -> i32;
    fn _kb_generate_partial_twiddles(
        partial_twiddles: *mut std::ffi::c_void,
        inverse: bool,
    ) -> i32;
    fn _kb_bit_rev(
        d_out: *mut std::ffi::c_void,
        d_inp: *const std::ffi::c_void,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> i32;
    fn _kb_ct_mixed_radix_narrow(
        d_inout: *mut std::ffi::c_void,
        radix: u32,
        lg_domain_size: u32,
        stage: u32,
        iterations: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
    ) -> i32;
}

// ── Safe wrappers ─────────────────────────────────────────────────────────────

pub unsafe fn kb_generate_all_twiddles<T>(
    twiddles: &DeviceBuffer<T>,
    inverse: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_generate_all_twiddles(
        twiddles.as_mut_raw_ptr(),
        inverse,
    ))
}

/// Generic wrapper — `partial_twiddles` is interpreted by the kernel as a
/// `fr_t[WINDOW_NUM][WINDOW_SIZE]` array regardless of the Rust element type.
pub unsafe fn kb_generate_partial_twiddles<T>(
    partial_twiddles: &DeviceBuffer<T>,
    inverse: bool,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_generate_partial_twiddles(
        partial_twiddles.as_mut_raw_ptr(),
        inverse,
    ))
}

pub unsafe fn kb_bit_rev(
    d_out: &DeviceBuffer<KoalaBear>,
    d_inp: &DeviceBuffer<KoalaBear>,
    lg_domain_size: u32,
    padded_poly_size: u32,
    poly_count: u32,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_bit_rev(
        d_out.as_mut_raw_ptr(),
        d_inp.as_raw_ptr(),
        lg_domain_size,
        padded_poly_size,
        poly_count,
    ))
}

pub unsafe fn kb_ct_mixed_radix_narrow(
    d_inout: &DeviceBuffer<KoalaBear>,
    radix: u32,
    lg_domain_size: u32,
    stage: u32,
    iterations: u32,
    padded_poly_size: u32,
    poly_count: u32,
    is_intt: bool,
) -> Result<(), CudaError> {
    if lg_domain_size > MAX_CUDA_KB_NTT_LOG_DOMAIN_SIZE {
        return Err(CudaError::new(1));
    }
    CudaError::from_result(_kb_ct_mixed_radix_narrow(
        d_inout.as_mut_raw_ptr(),
        radix,
        lg_domain_size,
        stage,
        iterations,
        padded_poly_size,
        poly_count,
        is_intt,
    ))
}
