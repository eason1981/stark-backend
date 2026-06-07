use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};
use openvm_stark_backend::{StarkProtocolConfig, SystemParams};
#[cfg(feature = "baby-bear-bn254-poseidon2")]
use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::{
    BabyBearBn254Poseidon2Config, Digest as Bn254Digest,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, Digest as BabyBearPoseidon2Digest,
};
use p3_field::{PrimeField32, TwoAdicField};
use serde::{de::DeserializeOwned, Serialize};

#[cfg(feature = "baby-bear-bn254-poseidon2")]
use crate::{
    bn254_sponge::MultiFieldTranscriptGpu,
    cuda::bn254_merkle_tree::{
        bn254_poseidon2_adjacent_compress_layer, bn254_poseidon2_compressing_row_hashes,
        bn254_poseidon2_compressing_row_hashes_ext,
    },
};
use crate::{
    cuda::merkle_tree::{
        poseidon2_adjacent_compress_layer, poseidon2_compressing_row_hashes,
        poseidon2_compressing_row_hashes_ext,
    },
    merkle_tree::BatchQueryMerkle,
    ntt_field::GpuNttField,
    sponge::{DuplexSpongeGpu, GpuFiatShamirTranscript},  // GpuFiatShamirTranscript used in trait bounds below
    types::{EF, F},
};

#[cfg(feature = "koala-bear-poseidon2")]
use openvm_stark_sdk::config::koala_bear_poseidon2::KoalaBearPoseidon2Config;

/// Dispatch trait for GPU Merkle hash kernels.
///
/// Each implementation routes the three kernel entry points
/// (`compress_rows`, `compress_rows_ext`, `compress_layer`) to the
/// appropriate CUDA FFI wrappers, and declares the concrete `Digest` type
/// those kernels produce.
pub trait GpuMerkleHash: Copy + Clone + Send + Sync + 'static {
    type Digest: Copy
        + Clone
        + PartialEq
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + BatchQueryMerkle
        + 'static;

    /// The base field type this Merkle hash operates over.
    type BaseField: GpuNttField + PrimeField32 + TwoAdicField + Copy + Clone + Send + Sync + 'static;

    /// The extension field type (4 base-field elements).
    type ExtField: Copy + Clone + Send + Sync + Serialize + DeserializeOwned + std::fmt::Debug + 'static;

    /// Compress rows of a base-field matrix into digest leaves.
    ///
    /// # Safety
    ///
    /// `out` must be allocated with capacity `query_stride` and `matrix` must
    /// contain `width * query_stride * (1 << log_rows_per_query)` valid elements.
    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<Self::BaseField>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError>;

    /// Compress rows of an extension-field matrix into digest leaves.
    ///
    /// # Safety
    ///
    /// `out` must be allocated with capacity `query_stride` and `matrix` must
    /// contain `width * query_stride * (1 << log_rows_per_query)` valid elements.
    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<Self::ExtField>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError>;

    /// Compress adjacent pairs of digests to build an inner Merkle layer.
    ///
    /// # Safety
    ///
    /// `output` must be allocated with capacity `output_size`, `prev_layer` must
    /// contain at least `output_size * 2` valid elements, and the two buffers
    /// must not overlap.
    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
    ) -> Result<(), CudaError>;
}

/// Binding trait that couples a `StarkProtocolConfig`, a Merkle hash scheme,
/// and a transcript type into a single coherent GPU proving configuration.
pub trait GpuHashScheme: Copy + Clone + Send + Sync + 'static {
    type SC: StarkProtocolConfig<
        F = Self::BaseField,
        EF = Self::ExtField,
        Digest = Self::Digest,
    >;
    type Digest: Copy
        + Clone
        + PartialEq
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + BatchQueryMerkle
        + 'static;

    /// The base prime field for this proving configuration.
    type BaseField: GpuNttField + PrimeField32 + TwoAdicField + Copy + Clone + Send + Sync + 'static;

    /// The extension field (typically degree-4 extension of `BaseField`).
    type ExtField: p3_field::Field
        + p3_field::TwoAdicField
        + p3_field::ExtensionField<Self::BaseField>
        + p3_field::BasedVectorSpace<Self::BaseField>
        + openvm_stark_backend::poly_common::Squarable
        + Copy
        + Clone
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + std::fmt::Debug
        + std::fmt::Display
        + 'static;

    type Transcript: GpuFiatShamirTranscript<Self::SC> + Default + Clone + Send + Sync + 'static;
    type MerkleHash: GpuMerkleHash<
        Digest = Self::Digest,
        BaseField = Self::BaseField,
        ExtField = Self::ExtField,
    >;

    fn default_config(params: SystemParams) -> Self::SC;

    fn default_transcript() -> Self::Transcript;
}

// ---------------------------------------------------------------------------
// Poseidon2 / BabyBear concrete implementations
// ---------------------------------------------------------------------------

/// Poseidon2 Merkle hash over BabyBear — delegates to the existing CUDA FFI.
#[derive(Clone, Copy, Debug, Default)]
pub struct Poseidon2MerkleHash;

impl GpuMerkleHash for Poseidon2MerkleHash {
    type Digest = BabyBearPoseidon2Digest;
    type BaseField = F;
    type ExtField = EF;

    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<F>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError> {
        poseidon2_compressing_row_hashes(out, matrix, width, query_stride, log_rows_per_query)
    }

    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError> {
        poseidon2_compressing_row_hashes_ext(out, matrix, width, query_stride, log_rows_per_query)
    }

    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
    ) -> Result<(), CudaError> {
        poseidon2_adjacent_compress_layer(output, prev_layer, output_size)
    }
}

/// BabyBear Poseidon2 hash scheme — the default scheme implemented in this crate.
#[derive(Clone, Copy, Debug, Default)]
pub struct BabyBearPoseidon2HashScheme;

impl GpuHashScheme for BabyBearPoseidon2HashScheme {
    type SC = BabyBearPoseidon2Config;
    type Digest = BabyBearPoseidon2Digest;
    type BaseField = F;
    type ExtField = EF;
    type Transcript = DuplexSpongeGpu;
    type MerkleHash = Poseidon2MerkleHash;

    fn default_config(params: SystemParams) -> Self::SC {
        Self::SC::default_from_params(params)
    }

    fn default_transcript() -> Self::Transcript {
        Self::Transcript::default()
    }
}

pub type DefaultHashScheme = BabyBearPoseidon2HashScheme;

// ---------------------------------------------------------------------------
// BN254 Poseidon2 concrete implementations
// ---------------------------------------------------------------------------

#[cfg(feature = "baby-bear-bn254-poseidon2")]
/// BN254 Poseidon2 Merkle hash — delegates to the BN254 CUDA FFI.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bn254Poseidon2MerkleHash;

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl GpuMerkleHash for Bn254Poseidon2MerkleHash {
    // `Bn254Digest` from stark-sdk = `[Bn254Scalar; 1]`, which is the same concrete type as
    // `Bn254Digest` in `cuda::bn254_merkle_tree` — both are type aliases for `[p3_bn254::Bn254;
    // 1]`.
    type Digest = Bn254Digest;
    type BaseField = F;
    type ExtField = EF;

    unsafe fn compress_rows(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<F>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError> {
        bn254_poseidon2_compressing_row_hashes(out, matrix, width, query_stride, log_rows_per_query)
    }

    unsafe fn compress_rows_ext(
        out: &mut DeviceBuffer<Self::Digest>,
        matrix: &DeviceBuffer<EF>,
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
    ) -> Result<(), CudaError> {
        bn254_poseidon2_compressing_row_hashes_ext(
            out,
            matrix,
            width,
            query_stride,
            log_rows_per_query,
        )
    }

    unsafe fn compress_layer(
        output: &mut DeviceBuffer<Self::Digest>,
        prev_layer: &DeviceBuffer<Self::Digest>,
        output_size: usize,
    ) -> Result<(), CudaError> {
        bn254_poseidon2_adjacent_compress_layer(output, prev_layer, output_size)
    }
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
/// BabyBear + BN254 Poseidon2 hash scheme (Groth16-friendly transcript).
#[derive(Clone, Copy, Debug, Default)]
pub struct BabyBearBn254Poseidon2HashScheme;

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl GpuHashScheme for BabyBearBn254Poseidon2HashScheme {
    type SC = BabyBearBn254Poseidon2Config;
    type Digest = Bn254Digest;
    type BaseField = F;
    type ExtField = EF;
    type Transcript = MultiFieldTranscriptGpu;
    type MerkleHash = Bn254Poseidon2MerkleHash;

    fn default_config(params: SystemParams) -> Self::SC {
        Self::SC::default_from_params(params)
    }

    fn default_transcript() -> Self::Transcript {
        Self::Transcript::default()
    }
}

// ─── KoalaBear Poseidon2 hash scheme ────────────────────────────────────────

#[cfg(feature = "koala-bear-poseidon2")]
mod kb_hash_scheme {
    use super::*;
    use crate::cuda::merkle_tree_kb::{
        kb_poseidon2_adjacent_compress_layer, kb_poseidon2_compressing_row_hashes,
        kb_poseidon2_compressing_row_hashes_ext, KbFpExt,
    };
    use crate::sponge::KoalaBearDuplexSpongeGpu;
    use openvm_stark_sdk::config::koala_bear_poseidon2::{
        Digest as KbDigest, EF as KbEF, F as KbF,
    };

    /// KoalaBear Poseidon2 Merkle hash — uses kb31_t CUDA kernels.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct KoalaBearPoseidon2MerkleHash;

    impl GpuMerkleHash for KoalaBearPoseidon2MerkleHash {
        type Digest = KbDigest;
        type BaseField = KbF;
        // KbEF = BinomialExtensionField<KoalaBear, 4> — same memory layout as KbFpExt [KoalaBear; 4]
        type ExtField = KbEF;

        unsafe fn compress_rows(
            out: &mut DeviceBuffer<Self::Digest>,
            matrix: &DeviceBuffer<KbF>,
            width: usize,
            query_stride: usize,
            log_rows_per_query: usize,
        ) -> Result<(), CudaError> {
            kb_poseidon2_compressing_row_hashes(out, matrix, width, query_stride, log_rows_per_query)
        }

        unsafe fn compress_rows_ext(
            out: &mut DeviceBuffer<Self::Digest>,
            matrix: &DeviceBuffer<KbEF>,
            width: usize,
            query_stride: usize,
            log_rows_per_query: usize,
        ) -> Result<(), CudaError> {
            // KbEF = BinomialExtensionField<KoalaBear, 4> has the same memory layout as
            // KbFpExt = struct { elems: [KoalaBear; 4] } — both are 4 × 4-byte = 16 bytes.
            // Create a non-owning DeviceBuffer<KbFpExt> view over the same device pointer.
            // ManuallyDrop prevents the drop impl from freeing device memory we don't own.
            let ffi_view = std::mem::ManuallyDrop::new(
                DeviceBuffer::<KbFpExt>::from_raw_parts(
                    matrix.as_raw_ptr() as *mut KbFpExt,
                    matrix.len(),
                )
            );
            kb_poseidon2_compressing_row_hashes_ext(
                out,
                &*ffi_view,
                width,
                query_stride,
                log_rows_per_query,
            )
        }

        unsafe fn compress_layer(
            output: &mut DeviceBuffer<Self::Digest>,
            prev_layer: &DeviceBuffer<Self::Digest>,
            output_size: usize,
        ) -> Result<(), CudaError> {
            kb_poseidon2_adjacent_compress_layer(output, prev_layer, output_size)
        }
    }

    /// KoalaBear Poseidon2 GPU hash scheme.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct KoalaBearPoseidon2HashScheme;

    impl GpuHashScheme for KoalaBearPoseidon2HashScheme {
        type SC = KoalaBearPoseidon2Config;
        type Digest = KbDigest;
        type BaseField = KbF;
        type ExtField = KbEF;
        type Transcript = KoalaBearDuplexSpongeGpu;
        type MerkleHash = KoalaBearPoseidon2MerkleHash;

        fn default_config(params: SystemParams) -> Self::SC {
            Self::SC::default_from_params(params)
        }

        fn default_transcript() -> Self::Transcript {
            Self::Transcript::default()
        }
    }
}

#[cfg(feature = "koala-bear-poseidon2")]
pub use kb_hash_scheme::{KoalaBearPoseidon2HashScheme, KoalaBearPoseidon2MerkleHash};

/// Marker trait for KoalaBear-native GPU hash schemes.
///
/// Used as a disjointness hint for Rust's coherence checker: types implementing
/// this marker are treated as KB-native, preventing overlap with the BB-native
/// `MultiRapProver`/`OpeningProver` impls that use `GpuHashScheme<BaseField=F>`.
#[cfg(feature = "koala-bear-poseidon2")]
pub trait KoalaBearNativeHash: GpuHashScheme {}

#[cfg(feature = "koala-bear-poseidon2")]
impl KoalaBearNativeHash for KoalaBearPoseidon2HashScheme {}
