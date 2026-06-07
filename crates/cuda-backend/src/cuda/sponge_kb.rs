//! Rust bindings for the KoalaBear CUDA sponge grinding kernel.

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
};

use crate::sponge::{validate_gpu_grind_bits, GrindError, KbDeviceSpongeState};

extern "C" {
    fn _kb_sponge_grind(
        init_state: *const KbDeviceSpongeState,
        bits: u32,
        min_witness: u32,
        max_witness: u32,
        result: *mut u32,
    ) -> i32;
}

/// Launch the KoalaBear GPU grinding kernel.
///
/// # Safety
/// - `init_state` must point to valid device memory containing a `KbDeviceSpongeState`
pub unsafe fn kb_sponge_grind(
    init_state: *const KbDeviceSpongeState,
    bits: u32,
    max_witness: u32,
) -> Result<u32, GrindError> {
    validate_gpu_grind_bits(bits as usize)?;
    let mut d_result = DeviceBuffer::with_capacity(1);
    [u32::MAX].copy_to(&mut d_result)?;
    for start in (0..=max_witness).step_by(1 << bits) {
        CudaError::from_result(_kb_sponge_grind(
            init_state,
            bits,
            start,
            max_witness,
            d_result.as_mut_ptr(),
        ))?;
        let result = d_result.to_host()?[0];
        if result < u32::MAX {
            return Ok(result);
        }
    }
    Err(GrindError::WitnessNotFound)
}
