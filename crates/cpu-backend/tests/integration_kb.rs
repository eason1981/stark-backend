//! KoalaBear CPU integration tests (method A — standalone, without backend-tests suite).
//!
//! Validates that `KoalaBearPoseidon2CpuEngine` can keygen, prove, and verify
//! basic AIR circuits end-to-end.

use openvm_stark_backend::{
    test_utils::{FibFixture, InteractionsFixture11, TestFixture},
    StarkEngine,
};
use openvm_stark_sdk::config::{
    kb_app_params_with_100_bits_security,
    koala_bear_poseidon2::{DuplexSponge, KoalaBearPoseidon2CpuEngine},
};

type KbEngine = KoalaBearPoseidon2CpuEngine<DuplexSponge>;

fn make_engine() -> KbEngine {
    KbEngine::new(kb_app_params_with_100_bits_security(11))
}

#[test]
fn kb_fib_roundtrip() {
    let engine = make_engine();
    let fib = FibFixture::new(0, 1, 1 << 4); // 16 rows
    let (vk, proof) = fib.keygen_and_prove(&engine);
    engine.verify(&vk, &proof).expect("kb fib verify failed");
}

#[test]
fn kb_interactions_roundtrip() {
    let engine = make_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    engine
        .verify(&vk, &proof)
        .expect("kb interactions verify failed");
}

#[test]
fn kb_fib_multi_shard() {
    let engine = make_engine();
    // Two separate FibFixtures — different AIR instances, each independently proved
    for log_n in [3, 4, 5] {
        let fib = FibFixture::new(0, 1, 1 << log_n);
        let (vk, proof) = fib.keygen_and_prove(&engine);
        engine
            .verify(&vk, &proof)
            .unwrap_or_else(|e| panic!("kb fib 2^{log_n} verify failed: {e}"));
    }
}
