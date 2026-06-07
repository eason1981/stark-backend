//! `GpuNttField` — dispatch trait for field-specific GPU NTT operations.
//!
//! Each implementation routes `batch_ntt` / `batch_ntt_small` to the correct
//! underlying CUDA kernel library (BabyBear → `supra_ntt`, KoalaBear →
//! `supra_ntt_kb`).

use std::{
    collections::BTreeSet,
    sync::{Mutex, OnceLock},
};

use openvm_cuda_common::{
    common::{device_reset_epoch, get_device},
    d_buffer::DeviceBuffer,
};

use crate::cuda::{batch_ntt_small, ntt};

/// Dispatch trait for GPU NTT over a specific field.
///
/// Implementors supply field-specific twiddle initialisation and the two NTT
/// entry points used by `stacked_pcs` and `whir`.
pub trait GpuNttField: Sized + Copy + Send + Sync + 'static + std::ops::Add<Output = Self> + std::ops::AddAssign {
    /// Returns true if CPU fallbacks should be used for operations that have
    /// known GPU arithmetic bugs for this field (e.g., KoalaBear with l_skip > 0).
    fn use_cpu_gkr() -> bool {
        false
    }

    /// Perform a column-wise batch NTT on `buffer`.
    ///
    /// `buffer` is column-major with columns of height `2^(log_trace_height + log_blowup)`.
    /// See [`crate::ntt::batch_ntt`] for full documentation.
    fn batch_ntt(
        buffer: &DeviceBuffer<Self>,
        log_trace_height: u32,
        log_blowup: u32,
        width: u32,
        bit_reverse: bool,
        is_intt: bool,
    );

    /// Perform a small-domain column-wise batch NTT on `buffer`.
    fn batch_ntt_small(
        buffer: &mut DeviceBuffer<Self>,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> Result<(), openvm_cuda_common::error::CudaError>;

    /// Perform an in-place bit-reversal permutation on `buffer`.
    fn bit_rev(
        d_out: &DeviceBuffer<Self>,
        d_inp: &DeviceBuffer<Self>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> Result<(), openvm_cuda_common::error::CudaError>;
}

// ── BabyBear ──────────────────────────────────────────────────────────────────

impl GpuNttField for p3_baby_bear::BabyBear {
    fn batch_ntt(
        buffer: &DeviceBuffer<Self>,
        log_trace_height: u32,
        log_blowup: u32,
        width: u32,
        bit_reverse: bool,
        is_intt: bool,
    ) {
        crate::ntt::batch_ntt(buffer, log_trace_height, log_blowup, width, bit_reverse, is_intt);
    }

    fn batch_ntt_small(
        buffer: &mut DeviceBuffer<Self>,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        unsafe { batch_ntt_small::batch_ntt_small(buffer, l_skip, cnt_blocks, is_intt) }
    }

    fn bit_rev(
        d_out: &DeviceBuffer<Self>,
        d_inp: &DeviceBuffer<Self>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        unsafe { ntt::bit_rev(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count) }
    }
}

// ── KoalaBear ─────────────────────────────────────────────────────────────────

#[cfg(feature = "koala-bear-poseidon2")]
mod kb_ntt {
    use super::*;
    use crate::cuda::ntt_kb;

    // KoalaBear NTT twiddle table constants (matching KB parameters.cuh).
    const KB_MAX_LG_DOMAIN_SIZE: usize = ntt_kb::MAX_CUDA_KB_NTT_LOG_DOMAIN_SIZE as usize;
    const KB_LG_WINDOW_SIZE: usize = (KB_MAX_LG_DOMAIN_SIZE + 4) / 5;
    const KB_WINDOW_SIZE: usize = 1 << KB_LG_WINDOW_SIZE;
    const KB_WINDOW_NUM: usize =
        (KB_MAX_LG_DOMAIN_SIZE + KB_LG_WINDOW_SIZE - 1) / KB_LG_WINDOW_SIZE;
    const KB_RADIX_TWIDDLES_SIZE: usize = 32 + 64 + 128 + 256 + 512;

    static KB_INIT_FORWARD: OnceLock<Mutex<BTreeSet<(i32, u64)>>> = OnceLock::new();
    static KB_INIT_INVERSE: OnceLock<Mutex<BTreeSet<(i32, u64)>>> = OnceLock::new();

    fn kb_ensure_initialized(
        inverse: bool,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        let device_key = (get_device()?, device_reset_epoch());
        let initialized = if inverse {
            &KB_INIT_INVERSE
        } else {
            &KB_INIT_FORWARD
        };
        let initialized = initialized.get_or_init(|| Mutex::new(BTreeSet::new()));
        let mut initialized = initialized.lock().unwrap();
        if initialized.contains(&device_key) {
            return Ok(());
        }

        {
            let partial_twiddles =
                DeviceBuffer::<[p3_koala_bear::KoalaBear; KB_WINDOW_SIZE]>::with_capacity(
                    KB_WINDOW_NUM,
                );
            let twiddles =
                DeviceBuffer::<p3_koala_bear::KoalaBear>::with_capacity(KB_RADIX_TWIDDLES_SIZE);
            unsafe {
                ntt_kb::kb_generate_all_twiddles(&twiddles, inverse)?;
                ntt_kb::kb_generate_partial_twiddles(&partial_twiddles, inverse)?;
            }
        }
        // Synchronize device to ensure twiddle tables are visible to all subsequent
        // kernel launches (including those on the NULL/default stream).
        // The initialization uses cudaStreamPerThread; NTT kernels use NULL stream.
        tracing::warn!("KB NTT twiddles initialized (inverse={})", inverse);
        openvm_cuda_common::stream::device_synchronize().map_err(|e| e)?;
        initialized.insert(device_key);
        Ok(())
    }

    struct KbNttImpl<'a> {
        buffer: &'a DeviceBuffer<p3_koala_bear::KoalaBear>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
        is_intt: bool,
        stage: u32,
    }

    impl<'a> KbNttImpl<'a> {
        fn new(
            buffer: &'a DeviceBuffer<p3_koala_bear::KoalaBear>,
            lg_domain_size: u32,
            padded_poly_size: u32,
            poly_count: u32,
            is_intt: bool,
        ) -> Self {
            kb_ensure_initialized(is_intt)
                .expect("failed to initialize KB CUDA NTT twiddle tables");
            Self {
                buffer,
                lg_domain_size,
                padded_poly_size,
                poly_count,
                is_intt,
                stage: 0,
            }
        }

        fn step(&mut self, iterations: u32) {
            assert!(iterations <= 10);
            let radix = if iterations < 6 { 6 } else { iterations };
            unsafe {
                ntt_kb::kb_ct_mixed_radix_narrow(
                    self.buffer,
                    radix,
                    self.lg_domain_size,
                    self.stage,
                    iterations,
                    self.padded_poly_size,
                    self.poly_count,
                    self.is_intt,
                )
                .expect("failed to launch KB CUDA mixed-radix NTT step");
            }
            self.stage += iterations;
        }
    }

    pub fn batch_ntt_kb(
        buffer: &DeviceBuffer<p3_koala_bear::KoalaBear>,
        log_trace_height: u32,
        log_blowup: u32,
        width: u32,
        bit_reverse: bool,
        is_intt: bool,
    ) {
        if log_trace_height == 0 {
            return;
        }
        assert!(
            log_trace_height <= ntt_kb::MAX_CUDA_KB_NTT_LOG_DOMAIN_SIZE,
            "CUDA KB batch_ntt supports log_trace_height <= {}",
            ntt_kb::MAX_CUDA_KB_NTT_LOG_DOMAIN_SIZE
        );

        let padded_poly_size = 1 << (log_trace_height + log_blowup);

        if bit_reverse {
            unsafe {
                ntt_kb::kb_bit_rev(buffer, buffer, log_trace_height, padded_poly_size, width)
                    .expect("failed to launch KB CUDA bit-reversal permutation");
            }
        }

        let mut _impl =
            KbNttImpl::new(buffer, log_trace_height, padded_poly_size, width, is_intt);
        if log_trace_height <= 10 {
            _impl.step(log_trace_height);
        } else if log_trace_height <= 17 {
            let step = log_trace_height / 2;
            _impl.step(step + log_trace_height % 2);
            _impl.step(step);
        } else if log_trace_height <= ntt_kb::MAX_CUDA_KB_NTT_LOG_DOMAIN_SIZE {
            let step = log_trace_height / 3;
            let rem = log_trace_height % 3;
            _impl.step(step);
            _impl.step(step);
            _impl.step(step + rem);
        } else {
            unreachable!("log_trace_height is bounded above");
        }
    }

    // KoalaBear small-NTT: The existing `_batch_ntt_small` kernel is BabyBear-specific
    // (it uses BB twiddles in constant memory).  Until a KB small-NTT kernel is
    // available we fall back to using the regular KB mixed-radix NTT.
    //
    // The buffer layout for `batch_ntt_small` is `cnt_blocks` consecutive chunks,
    // each of `2^l_skip` elements.  This is exactly the layout expected by
    // `batch_ntt_kb` with `width=cnt_blocks, log_blowup=0, padded_poly_size=2^l_skip`.
    pub fn batch_ntt_small_kb(
        buffer: &mut DeviceBuffer<p3_koala_bear::KoalaBear>,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
        use openvm_stark_backend::dft::Radix2BowersSerial;
        use p3_dft::TwoAdicSubgroupDft;
        if l_skip == 0 || cnt_blocks == 0 {
            return Ok(());
        }
        if l_skip > batch_ntt_small::MAX_SMALL_NTT_LEVEL {
            return Err(openvm_cuda_common::error::CudaError::new(1));
        }
        // CPU fallback: GPU KB NTT twiddle table has wrong values for small domains.
        // Use CPU DFT which produces correct results consistent with the WHIR expectations.
        let chunk_len = 1usize << l_skip;
        let dft = Radix2BowersSerial;
        let mut host_buf = buffer.to_host().map_err(|_| openvm_cuda_common::error::CudaError::new(1))?;
        // Standard NTT conventions (matching eval_to_coeff_rs_message and BB GPU NTT):
        // - is_intt=true  (INTT): domain evaluations → monomial coefficients → use idft
        // - is_intt=false (NTT):  monomial coefficients → domain evaluations → use dft
        for chunk in host_buf.chunks_exact_mut(chunk_len) {
            let result = if is_intt {
                dft.idft(chunk.to_vec())
            } else {
                dft.dft(chunk.to_vec())
            };
            chunk.copy_from_slice(&result);
        }
        host_buf.copy_to(buffer).map_err(|_| openvm_cuda_common::error::CudaError::new(1))?;
        Ok(())
    }

    /// Wrapper for ntt_kb::kb_bit_rev, accessible from the parent module.
    pub(super) fn kb_ntt_bit_rev(
        d_out: &DeviceBuffer<p3_koala_bear::KoalaBear>,
        d_inp: &DeviceBuffer<p3_koala_bear::KoalaBear>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        unsafe { ntt_kb::kb_bit_rev(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count) }
    }
}

#[cfg(feature = "koala-bear-poseidon2")]
impl GpuNttField for p3_koala_bear::KoalaBear {
    fn use_cpu_gkr() -> bool {
        true
    }

    fn batch_ntt(
        buffer: &DeviceBuffer<Self>,
        log_trace_height: u32,
        log_blowup: u32,
        width: u32,
        bit_reverse: bool,
        is_intt: bool,
    ) {
        kb_ntt::batch_ntt_kb(buffer, log_trace_height, log_blowup, width, bit_reverse, is_intt);
    }

    fn batch_ntt_small(
        buffer: &mut DeviceBuffer<Self>,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        kb_ntt::batch_ntt_small_kb(buffer, l_skip, cnt_blocks, is_intt)
    }

    fn bit_rev(
        d_out: &DeviceBuffer<Self>,
        d_inp: &DeviceBuffer<Self>,
        lg_domain_size: u32,
        padded_poly_size: u32,
        poly_count: u32,
    ) -> Result<(), openvm_cuda_common::error::CudaError> {
        unsafe { kb_ntt::kb_ntt_bit_rev(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count) }
    }
}
