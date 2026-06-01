use std::{
    io::{self, Read, Write},
    marker::PhantomData,
    sync::OnceLock,
};

use openvm_stark_backend::{
    codec::{
        decode_extension_field32, decode_prime_field32, encode_extension_field32,
        encode_prime_field32, DecodableConfig, EncodableConfig,
    },
    hasher::Hasher,
    p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation},
    prover::{Coordinator, CpuColMajorBackend, ReferenceDevice},
    transcript::duplex_sponge,
    FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams, TranscriptLog,
};
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_poseidon2::{ExternalLayerConstants, Poseidon2};

const RATE: usize = 8;
const WIDTH: usize = 16;
pub const CHUNK: usize = 8;
pub const DIGEST_SIZE: usize = CHUNK;

type Perm = Poseidon2KoalaBear<WIDTH>;
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_SIZE>;
type Compress<P> = TruncatedPermutation<P, 2, CHUNK, WIDTH>;
type PermHasher<P> = Hasher<F, Digest, Hash<P>, Compress<P>>;
type SC = KoalaBearPoseidon2Config;

pub type F = KoalaBear;
pub type EF = BinomialExtensionField<KoalaBear, 4>;
pub const D_EF: usize = 4;
pub type Digest = [F; DIGEST_SIZE];
pub type DuplexSponge = duplex_sponge::DuplexSponge<F, Perm, WIDTH, RATE>;
pub type DuplexSpongeRecorder = duplex_sponge::DuplexSpongeRecorder<F, Perm, WIDTH, RATE>;
pub type DuplexSpongeValidator = duplex_sponge::DuplexSpongeValidator<F, Perm, WIDTH, RATE>;

/// Initial external round constants for the 16-width KoalaBear Poseidon2.
///
/// Same u32 values as the HorizenLabs BabyBear constants (rows 0–3 of RC_16_30),
/// reinterpreted as KoalaBear field elements. Matches pico's `pico_poseidon2kb_init()`.
pub const KOALABEAR_RC16_EXTERNAL_INITIAL: [[KoalaBear; 16]; 4] = KoalaBear::new_2d_array([
    [
        0x69cbb6af, 0x46ad93f9, 0x60a00f4e, 0x6b1297cd, 0x23189afe, 0x732e7bef, 0x72c246de,
        0x2c941900, 0x0557eede, 0x1580496f, 0x3a3ea77b, 0x54f3f271, 0x0f49b029, 0x47872fe1,
        0x221e2e36, 0x1ab7202e,
    ],
    [
        0x487779a6, 0x3851c9d8, 0x38dc17c0, 0x209f8849, 0x268dcee8, 0x350c48da, 0x5b9ad32e,
        0x0523272b, 0x3f89055b, 0x01e894b2, 0x13ddedde, 0x1b2ef334, 0x7507d8b4, 0x6ceeb94e,
        0x52eb6ba2, 0x50642905,
    ],
    [
        0x05453f3f, 0x06349efc, 0x6922787c, 0x04bfff9c, 0x768c714a, 0x3e9ff21a, 0x15737c9c,
        0x2229c807, 0x0d47f88c, 0x097e0ecc, 0x27eadba0, 0x2d7d29e4, 0x3502aaa0, 0x0f475fd7,
        0x29fbda49, 0x018afffd,
    ],
    [
        0x0315b618, 0x6d4497d1, 0x1b171d9e, 0x52861abd, 0x2e5d0501, 0x3ec8646c, 0x6e5f250a,
        0x148ae8e6, 0x17f5fa4a, 0x3e66d284, 0x0051aa3b, 0x483f7913, 0x2cfe5f15, 0x023427ca,
        0x2cc78315, 0x1e36ea47,
    ],
]);

/// Terminal external round constants for the 16-width KoalaBear Poseidon2.
///
/// Rows 24–27 of RC_16_30 are zeros (the original table has only 21 meaningful rows).
/// Matches pico's `pico_poseidon2kb_init()`.
pub const KOALABEAR_RC16_EXTERNAL_FINAL: [[KoalaBear; 16]; 4] =
    KoalaBear::new_2d_array([[0; 16], [0; 16], [0; 16], [0; 16]]);

/// Internal round constants for the 16-width KoalaBear Poseidon2 (20 rounds).
///
/// Col 0 of rows 4–23 of RC_16_30. Rows 4–16 are the BB partial constants;
/// rows 17–20 are the BB terminal external col-0 values; rows 21–23 are zero.
/// Matches pico's `pico_poseidon2kb_init()`.
pub const KOALABEAR_RC16_INTERNAL: [KoalaBear; 20] = KoalaBear::new_array([
    // rows 4–16: BB partial round constants (col 0)
    0x5a8053c0, 0x693be639, 0x3858867d, 0x19334f6b, 0x128f0fd8, 0x4e2b1ccb, 0x61210ce0, 0x3c318939,
    0x0b5b2f22, 0x2edb11d5, 0x213effdf, 0x0cac4606, 0x241af16d,
    // rows 17–20: BB terminal external col-0 values
    0x7290a80d, 0x5b8fe9c4, 0x58eb1611, 0x59986d19, // rows 21–23: zero
    0, 0, 0,
]);

/// Build the default 16-width KoalaBear Poseidon2 permutation.
pub fn default_koalabear_poseidon2_16() -> Poseidon2KoalaBear<16> {
    Poseidon2::new(
        ExternalLayerConstants::new(
            KOALABEAR_RC16_EXTERNAL_INITIAL.to_vec(),
            KOALABEAR_RC16_EXTERNAL_FINAL.to_vec(),
        ),
        KOALABEAR_RC16_INTERNAL.to_vec(),
    )
}

#[derive(Clone, Debug, derive_new::new)]
pub struct KoalaBearPoseidon2Config {
    params: SystemParams,
    hasher: PermHasher<Perm>,
}

impl StarkProtocolConfig for KoalaBearPoseidon2Config {
    type F = F;
    type EF = EF;
    type Digest = Digest;
    type Hasher = PermHasher<Perm>;

    fn params(&self) -> &SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

impl KoalaBearPoseidon2Config {
    pub fn new_from_perm(params: SystemParams, perm: Perm) -> Self {
        let hasher = Hasher::new(
            PaddingFreeSponge::new(perm.clone()),
            TruncatedPermutation::new(perm),
        );
        Self { params, hasher }
    }

    pub fn default_from_params(params: SystemParams) -> Self {
        let perm = default_koalabear_poseidon2_16();
        Self::new_from_perm(params, perm)
    }
}

impl EncodableConfig for KoalaBearPoseidon2Config {
    fn encode_base_field<W: Write>(val: &F, writer: &mut W) -> io::Result<()> {
        encode_prime_field32(val, writer)
    }

    fn encode_extension_field<W: Write>(val: &EF, writer: &mut W) -> io::Result<()> {
        encode_extension_field32::<Self::F, _, _>(val, writer)
    }

    fn encode_digest<W: Write>(digest: &Self::Digest, writer: &mut W) -> io::Result<()> {
        for val in digest {
            encode_prime_field32(val, writer)?;
        }
        Ok(())
    }
}

impl DecodableConfig for KoalaBearPoseidon2Config {
    fn decode_base_field<R: Read>(reader: &mut R) -> io::Result<F> {
        decode_prime_field32(reader)
    }

    fn decode_extension_field<R: Read>(reader: &mut R) -> io::Result<EF> {
        decode_extension_field32::<F, _, _>(reader)
    }

    fn decode_digest<R: Read>(reader: &mut R) -> io::Result<Digest> {
        let mut result = Digest::default();
        for val in &mut result {
            *val = decode_prime_field32(reader)?;
        }
        Ok(result)
    }
}

pub struct KoalaBearPoseidon2RefEngine<TS = DuplexSponge> {
    device: ReferenceDevice<SC>,
    _transcript: PhantomData<TS>,
}

impl<TS> StarkEngine for KoalaBearPoseidon2RefEngine<TS>
where
    TS: FiatShamirTranscript<SC> + From<Perm>,
{
    type SC = SC;
    type PB = CpuColMajorBackend<SC>;
    type PD = ReferenceDevice<SC>;
    type TS = TS;

    fn new(params: SystemParams) -> Self {
        let config = KoalaBearPoseidon2Config::default_from_params(params);
        Self {
            device: ReferenceDevice::new(config),
            _transcript: PhantomData,
        }
    }

    fn config(&self) -> &SC {
        self.device.config()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        TS::from(default_koalabear_poseidon2_16())
    }

    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(CpuColMajorBackend::new(), self.device.clone(), transcript)
    }
}

#[cfg(feature = "cpu-backend")]
mod cpu_engine {
    use openvm_cpu_backend::{CpuBackend, CpuDevice};

    use super::*;

    #[derive(Clone, Debug)]
    pub struct KbCpuTranscript {
        inner: openvm_stark_backend::p3_challenger::DuplexChallenger<
            KoalaBear,
            Poseidon2KoalaBear<WIDTH>,
            WIDTH,
            RATE,
        >,
    }

    impl From<Perm> for KbCpuTranscript {
        fn from(perm: Perm) -> Self {
            Self {
                inner: openvm_stark_backend::p3_challenger::DuplexChallenger::new(perm),
            }
        }
    }

    impl FiatShamirTranscript<KoalaBearPoseidon2Config> for KbCpuTranscript {
        #[inline]
        fn observe(&mut self, value: KoalaBear) {
            openvm_stark_backend::p3_challenger::CanObserve::observe(&mut self.inner, value);
        }

        #[inline]
        fn sample(&mut self) -> KoalaBear {
            openvm_stark_backend::p3_challenger::CanSample::sample(&mut self.inner)
        }

        fn observe_commit(&mut self, digest: [KoalaBear; RATE]) {
            for x in digest {
                openvm_stark_backend::p3_challenger::CanObserve::observe(&mut self.inner, x);
            }
        }

        fn grind(&mut self, bits: usize) -> KoalaBear {
            openvm_stark_backend::p3_challenger::GrindingChallenger::grind(&mut self.inner, bits)
        }
    }

    pub struct KoalaBearPoseidon2CpuEngine<TS = KbCpuTranscript> {
        device: CpuDevice<SC>,
        _transcript: PhantomData<TS>,
    }

    impl<TS> StarkEngine for KoalaBearPoseidon2CpuEngine<TS>
    where
        TS: FiatShamirTranscript<SC> + From<Poseidon2KoalaBear<WIDTH>>,
    {
        type SC = SC;
        type PB = CpuBackend<SC>;
        type PD = CpuDevice<SC>;
        type TS = TS;

        fn new(params: SystemParams) -> Self {
            let config = KoalaBearPoseidon2Config::default_from_params(params);
            Self {
                device: CpuDevice::new(config),
                _transcript: PhantomData,
            }
        }

        fn config(&self) -> &SC {
            self.device.config()
        }

        fn device(&self) -> &Self::PD {
            &self.device
        }

        fn initial_transcript(&self) -> Self::TS {
            TS::from(default_koalabear_poseidon2_16())
        }

        fn prover_from_transcript(
            &self,
            transcript: TS,
        ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
            Coordinator::new(CpuBackend::new(), self.device.clone(), transcript)
        }
    }
}

#[cfg(feature = "cpu-backend")]
pub use cpu_engine::{KbCpuTranscript, KoalaBearPoseidon2CpuEngine};

pub fn poseidon2_perm() -> &'static Poseidon2KoalaBear<WIDTH> {
    static PERM: OnceLock<Poseidon2KoalaBear<WIDTH>> = OnceLock::new();
    PERM.get_or_init(default_koalabear_poseidon2_16)
}

pub fn poseidon2_compress_with_capacity(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> ([F; CHUNK], [F; CHUNK]) {
    let mut state = [F::ZERO; WIDTH];
    state[..CHUNK].copy_from_slice(&left);
    state[CHUNK..].copy_from_slice(&right);
    poseidon2_perm().permute_mut(&mut state);
    (
        state[..CHUNK].try_into().unwrap(),
        state[CHUNK..].try_into().unwrap(),
    )
}

pub fn default_duplex_sponge() -> DuplexSponge {
    DuplexSponge::from(poseidon2_perm().clone())
}

pub fn default_duplex_sponge_recorder() -> DuplexSpongeRecorder {
    DuplexSpongeRecorder::from(poseidon2_perm().clone())
}

pub fn default_duplex_sponge_validator(
    logs: TranscriptLog<F, [F; WIDTH]>,
) -> DuplexSpongeValidator {
    DuplexSpongeValidator::new(poseidon2_perm().clone(), logs)
}
