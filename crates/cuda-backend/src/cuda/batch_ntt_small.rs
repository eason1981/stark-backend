use std::{
    collections::BTreeSet,
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

use openvm_cuda_common::{
    common::{device_reset_epoch, get_device},
    d_buffer::DeviceBuffer,
    error::{check, CudaError},
};

use crate::prelude::F;

/// Maximum `l_skip` supported by the small-NTT CUDA kernel.
pub const MAX_SMALL_NTT_LEVEL: usize = 10;

/// Validate the GPU small-NTT domain size.
///
/// `l_skip == 0` is explicitly allowed and means the caller can take the size-1 no-op path
/// without launching the small-NTT kernel.
pub fn validate_gpu_l_skip(l_skip: usize) -> Result<(), CudaError> {
    if l_skip > MAX_SMALL_NTT_LEVEL {
        return Err(CudaError::new(1));
    }
    Ok(())
}

/// Size of the device NTT twiddle table (2^11 - 2 = 2046 elements for MAX_NTT_LEVEL=10)
pub const DEVICE_NTT_TWIDDLES_SIZE: usize = (1 << 11) - 2;

extern "C" {
    pub fn _batch_ntt_small(
        d_buffer: *mut F,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> i32;

    fn _generate_device_ntt_twiddles(d_twiddles: *mut c_void) -> i32;
}

#[cfg(feature = "koala-bear-poseidon2")]
extern "C" {
    // KoalaBear has its OWN DEVICE_NTT_TWIDDLES constant array (batch_ntt_small.cu is compiled
    // into the cuda_kb_all library too). It must be initialized separately from the BabyBear one.
    fn _kb_generate_device_ntt_twiddles(d_twiddles: *mut c_void) -> i32;
}

static INIT_DEVICE_NTT_TWIDDLES: OnceLock<Mutex<BTreeSet<(i32, u64)>>> = OnceLock::new();

/// Ensure device NTT twiddles are initialized in constant memory.
/// Safe to call multiple times - initialization happens once per device/reset epoch.
pub fn ensure_device_ntt_twiddles_initialized() -> Result<(), CudaError> {
    let device_key = (get_device()?, device_reset_epoch());
    let initialized =
        INIT_DEVICE_NTT_TWIDDLES.get_or_init(|| Mutex::new(BTreeSet::<(i32, u64)>::new()));
    let mut initialized = initialized.lock().unwrap();
    if initialized.contains(&device_key) {
        return Ok(());
    }

    {
        // Temporary staging buffer for the generated twiddles. The CUDA side copies from this
        // buffer into constant memory and synchronizes before returning, so it is safe to free
        // the buffer once `generate_device_ntt_twiddles` completes.
        let twiddles = DeviceBuffer::<F>::with_capacity(DEVICE_NTT_TWIDDLES_SIZE);
        unsafe {
            generate_device_ntt_twiddles(&twiddles)?;
        }
        // KoalaBear's DEVICE_NTT_TWIDDLES is a distinct constant array in the cuda_kb_all
        // library and must be initialized too — otherwise it stays zero and get_twiddle()
        // returns 0, collapsing the round0 zerocheck/logup small-NTT evals for l_skip>0.
        #[cfg(feature = "koala-bear-poseidon2")]
        {
            let kb_twiddles = DeviceBuffer::<F>::with_capacity(DEVICE_NTT_TWIDDLES_SIZE);
            unsafe {
                CudaError::from_result(_kb_generate_device_ntt_twiddles(
                    kb_twiddles.as_mut_raw_ptr(),
                ))?;
            }
        }
    }
    initialized.insert(device_key);
    Ok(())
}

/// Generate device NTT twiddles and copy to constant memory.
/// `d_twiddles` must have capacity for `DEVICE_NTT_TWIDDLES_SIZE` elements.
unsafe fn generate_device_ntt_twiddles(d_twiddles: &DeviceBuffer<F>) -> Result<(), CudaError> {
    CudaError::from_result(_generate_device_ntt_twiddles(d_twiddles.as_mut_raw_ptr()))
}

pub unsafe fn batch_ntt_small(
    buffer: &mut DeviceBuffer<F>,
    l_skip: usize,
    cnt_blocks: usize,
    is_intt: bool,
) -> Result<(), CudaError> {
    if l_skip == 0 || cnt_blocks == 0 {
        return Ok(());
    }
    validate_gpu_l_skip(l_skip)?;

    ensure_device_ntt_twiddles_initialized()?;
    check(_batch_ntt_small(
        buffer.as_mut_ptr(),
        l_skip,
        cnt_blocks,
        is_intt,
    ))
}
