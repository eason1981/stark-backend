use std::marker::PhantomData;

use itertools::Itertools;
use openvm_cuda_common::memory_manager::MemTracker;
use openvm_stark_backend::{
    poly_common::Squarable,
    proof::*,
    prover::{
        DeviceMultiStarkProvingKey, MultiRapProver, OpeningProver, ProverBackend, ProverDevice,
        ProvingContext, TraceCommitter,
    },
};
use tracing::instrument;

use crate::{
    base::DeviceMatrix,
    cuda::field_kernels::{BabyBearKernels, FieldKernels},
    hash_scheme::{DefaultHashScheme, GpuHashScheme},
    logup_zerocheck::prove_zerocheck_and_logup_gpu,
    merkle_tree::{MerkleProofQueryDigest, MerkleTreeConstructor},
    prelude::{D_EF},
    sponge::GpuFiatShamirTranscript,
    stacked_pcs::{stacked_commit, StackedPcsDataGpu},
    stacked_reduction::prove_stacked_opening_reduction_gpu,
    whir::prove_whir_opening_gpu,
    AirDataGpu, GpuDevice, ProverError,
};

/// Associates a `GpuHashScheme` with the correct `FieldKernels` implementation
/// for the GPU proving pipeline (stacked reduction + WHIR).
///
/// Concrete impls:
/// - `DefaultHashScheme` (BabyBear) → `BabyBearKernels`
/// - `KoalaBearPoseidon2HashScheme` → `KoalaBearKernels`
pub trait FieldKernelsFor<HS: GpuHashScheme> {
    type FK: FieldKernels<Val = HS::BaseField, ValExt = HS::ExtField>;
}

impl FieldKernelsFor<DefaultHashScheme> for GpuDevice {
    type FK = BabyBearKernels;
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
impl FieldKernelsFor<crate::hash_scheme::BabyBearBn254Poseidon2HashScheme> for GpuDevice {
    type FK = BabyBearKernels;
}

#[cfg(feature = "koala-bear-poseidon2")]
impl FieldKernelsFor<crate::hash_scheme::KoalaBearPoseidon2HashScheme> for GpuDevice {
    type FK = crate::cuda::field_kernels::KoalaBearKernels;
}

/// Generic GPU prover backend parameterised by a hash scheme `HS`.
///
/// Use the [`GpuBackend`] type alias to refer to the concrete BabyBear-Poseidon2
/// backend without spelling out the generic parameter.
#[derive(Clone, Copy)]
pub struct GenericGpuBackend<HS: GpuHashScheme>(PhantomData<HS>);

impl<HS: GpuHashScheme> Default for GenericGpuBackend<HS> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Concrete GPU backend using the default BabyBear-Poseidon2 hash scheme.
pub type GpuBackend = GenericGpuBackend<DefaultHashScheme>;

impl<HS: GpuHashScheme> ProverBackend for GenericGpuBackend<HS> {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;  // 4 for both BB and KB

    type Val = HS::BaseField;
    type Challenge = HS::ExtField;
    type Commitment = HS::Digest;
    type Matrix = DeviceMatrix<HS::BaseField>;
    type PcsData = StackedPcsDataGpu<HS::BaseField, HS::Digest>;
    type OtherAirData = AirDataGpu<HS::BaseField>;
}

impl<HS: GpuHashScheme> TraceCommitter<GenericGpuBackend<HS>> for GpuDevice
where
    HS::MerkleHash: MerkleTreeConstructor,
{
    type Error = ProverError;

    fn commit(
        &self,
        traces: &[&DeviceMatrix<HS::BaseField>],
    ) -> Result<(HS::Digest, StackedPcsDataGpu<HS::BaseField, HS::Digest>), Self::Error> {
        stacked_commit::<HS::BaseField, HS::MerkleHash>(
            self.config.l_skip,
            self.config.n_stack,
            self.config.log_blowup,
            self.config.k_whir(),
            traces,
            self.prover_config,
        )
    }
}

impl<HS: GpuHashScheme, TS: GpuFiatShamirTranscript<HS::SC>>
    ProverDevice<GenericGpuBackend<HS>, TS> for GpuDevice
where
    GpuDevice: FieldKernelsFor<HS>,
    HS::MerkleHash: MerkleTreeConstructor,
    HS::Digest: MerkleProofQueryDigest,
{
    type Error = ProverError;
}

impl<HS: GpuHashScheme, TS: GpuFiatShamirTranscript<HS::SC>>
    MultiRapProver<GenericGpuBackend<HS>, TS> for GpuDevice
where
    GpuDevice: FieldKernelsFor<HS>,
{
    type PartialProof = (GkrProof<HS::SC>, BatchConstraintProof<HS::SC>);
    type Artifacts = Vec<HS::ExtField>;
    type Error = ProverError;

    #[allow(clippy::type_complexity)]
    #[instrument(name = "prover.rap_constraints", skip_all, fields(phase = "prover"))]
    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        ctx: &ProvingContext<GenericGpuBackend<HS>>,
        common_main_pcs_data: &StackedPcsDataGpu<HS::BaseField, HS::Digest>,
    ) -> Result<((GkrProof<HS::SC>, BatchConstraintProof<HS::SC>), Vec<HS::ExtField>), Self::Error>
    {
        let mem = MemTracker::start_and_reset_peak("prover.rap_constraints");
        let save_memory = self.prover_config.zerocheck_save_memory;
        let monomial_num_y_threshold = if self.config.log_blowup == 1 { 512 } else { 64 };
        let (gkr_proof, batch_constraint_proof, r) =
            prove_zerocheck_and_logup_gpu::<
                <GpuDevice as FieldKernelsFor<HS>>::FK, HS, TS
            >(
                transcript,
                mpk,
                ctx,
                save_memory,
                monomial_num_y_threshold,
                self.sm_count,
            )?;
        mem.emit_metrics();
        Ok(((gkr_proof, batch_constraint_proof), r))
    }
}

impl<HS: GpuHashScheme, TS: GpuFiatShamirTranscript<HS::SC>>
    OpeningProver<GenericGpuBackend<HS>, TS> for GpuDevice
where
    GpuDevice: FieldKernelsFor<HS>,
    HS::MerkleHash: MerkleTreeConstructor,
    HS::Digest: MerkleProofQueryDigest,
{
    type OpeningProof = (StackingProof<HS::SC>, WhirProof<HS::SC>);
    type OpeningPoints = Vec<HS::ExtField>;
    type Error = ProverError;

    #[instrument(name = "prover.openings", skip_all, fields(phase = "prover"))]
    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        ctx: ProvingContext<GenericGpuBackend<HS>>,
        common_main_pcs_data: StackedPcsDataGpu<HS::BaseField, HS::Digest>,
        r: Vec<HS::ExtField>,
    ) -> Result<Self::OpeningProof, Self::Error> {
        let mut mem = MemTracker::start_and_reset_peak("prover.openings");
        let params = self.config();
        #[cfg(debug_assertions)]
        {
            let total_stacked_width: usize = std::iter::once(common_main_pcs_data.layout().width())
                .chain(ctx.per_trace.iter().flat_map(|(air_idx, air_ctx)| {
                    mpk.per_air[*air_idx]
                        .preprocessed_data
                        .iter()
                        .map(|committed| committed.data.layout().width())
                        .chain(
                            air_ctx
                                .cached_mains
                                .iter()
                                .map(|committed| committed.data.layout().width()),
                        )
                }))
                .sum();
            debug_assert!(
                total_stacked_width <= mpk.params.w_stack,
                "total stacked width across commits ({total_stacked_width}) exceeds w_stack ({})",
                mpk.params.w_stack
            );
        }
        let (stacking_proof, u_prisma, stacked_per_commit) =
            prove_stacked_opening_reduction_gpu::<
                <GpuDevice as FieldKernelsFor<HS>>::FK, HS, TS
            >(
                self,
                transcript,
                mpk,
                ctx,
                common_main_pcs_data,
                &r,
            )?;

        let (&u0, u_rest) = u_prisma.split_first().unwrap();
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        let whir_proof = prove_whir_opening_gpu::<
            <GpuDevice as FieldKernelsFor<HS>>::FK, HS, TS
        >(params, transcript, stacked_per_commit, &u_cube)?;
        mem.emit_metrics();
        mem.reset_peak();
        Ok((stacking_proof, whir_proof))
    }
}

// ── KoalaBear stub impls ───────────────────────────────────────────────────
// These satisfy the `ProverDevice` bound for `KoalaBearPoseidon2GpuEngine`
// while the full KB GPU proving kernels (logup/GKR, WHIR, stacked reduction)
// are being wired to `cuda_kb_all`. They will panic at runtime if called.
// All KB impls (MultiRapProver, OpeningProver, ProverDevice) are handled by the generic
// impls using FieldKernelsFor<KoalaBearPoseidon2HashScheme> → KoalaBearKernels
