//! Rust bindings for KoalaBear Poseidon2 Merkle tree CUDA kernels.

use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use p3_koala_bear::KoalaBear;

use crate::merkle_tree::BatchQueryMerkle;

pub(crate) type KbDigest = [KoalaBear; 8];

// Extension field: 4 KoalaBear elements (KoalaBear^4 in coefficient form).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KbFpExt {
    pub elems: [KoalaBear; 4],
}

// SAFETY: KbFpExt contains only [KoalaBear; 4] which is Send + Sync.
unsafe impl Send for KbFpExt {}
unsafe impl Sync for KbFpExt {}

extern "C" {
    fn _kb_poseidon2_compressing_row_hashes(
        out: *mut KbDigest,
        matrix: *const KoalaBear,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> i32;

    fn _kb_poseidon2_compressing_row_hashes_ext(
        out: *mut KbDigest,
        matrix: *const KbFpExt,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> i32;

    fn _kb_poseidon2_adjacent_compress_layer(
        output: *mut KbDigest,
        prev_layer: *const KbDigest,
        output_size: usize,
    ) -> i32;
}

pub unsafe fn kb_poseidon2_compressing_row_hashes(
    out: &mut DeviceBuffer<KbDigest>,
    matrix: &DeviceBuffer<KoalaBear>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_poseidon2_compressing_row_hashes(
        out.as_mut_ptr(),
        matrix.as_ptr(),
        width,
        query_stride,
        log_rows_per_query,
    ))
}

pub unsafe fn kb_poseidon2_compressing_row_hashes_ext(
    out: &mut DeviceBuffer<KbDigest>,
    matrix: &DeviceBuffer<KbFpExt>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_poseidon2_compressing_row_hashes_ext(
        out.as_mut_ptr(),
        matrix.as_ptr(),
        width,
        query_stride,
        log_rows_per_query,
    ))
}

pub unsafe fn kb_poseidon2_adjacent_compress_layer(
    output: &mut DeviceBuffer<KbDigest>,
    prev_layer: &DeviceBuffer<KbDigest>,
    output_size: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_kb_poseidon2_adjacent_compress_layer(
        output.as_mut_ptr(),
        prev_layer.as_ptr(),
        output_size,
    ))
}
