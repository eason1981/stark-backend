use std::process::exit;

use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if !cuda_available() {
        eprintln!("cargo:warning=CUDA is not available");
        exit(1);
    }

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BABY_BEAR_BN254_POSEIDON2");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_KOALA_BEAR_POSEIDON2");
    // Recompile CUDA kernels when cuda-common headers change (e.g. koala_bear_ext.hpp).
    if let Ok(common_include) = std::env::var("DEP_CUDA_COMMON_INCLUDE") {
        println!("cargo:rerun-if-changed={}", common_include);
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .include("cuda/include");

    common.emit_link_directives();

    let mut builder = common
        .clone()
        .library_name("cuda-backend")
        .watch("cuda")
        .include("cuda/include");

    // Collect .cu files, excluding feature-gated files unless the feature is enabled.
    let bn254_enabled = std::env::var("CARGO_FEATURE_BABY_BEAR_BN254_POSEIDON2").is_ok();
    let kb_enabled = std::env::var("CARGO_FEATURE_KOALA_BEAR_POSEIDON2").is_ok();
    for entry in glob::glob("cuda/src/**/*.cu").expect("failed to glob cuda/src/**/*.cu") {
        let path = entry.expect("glob error");
        let fname = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
        if !bn254_enabled && fname == "bn254_poseidon2.cu" {
            continue;
        }
        // Skip KoalaBear-specific files unless feature is enabled
        if !kb_enabled && (fname == "sponge_kb.cu" || fname == "merkle_tree_kb.cu") {
            continue;
        }
        builder = builder.file(path);
    }

    builder.build();

    // BabyBear NTT library (supra_ntt).
    common
        .clone()
        .library_name("supra_ntt")
        .include("cuda/supra/include")
        .files_from_glob("cuda/supra/*.cu")
        .build();

    // KoalaBear full CUDA library (cuda_kb_all) — only when feature is enabled.
    //
    // Combines NTT + proving kernels into ONE library so that CUDA device symbols
    // (DEVICE_NTT_TWIDDLES, etc.) are defined in the same link unit that uses them.
    // Splitting them causes nvlink "undefined reference" errors during device-link.
    //
    // Include order rules:
    //   1. cuda/kb_include MUST come BEFORE DEP_CUDA_COMMON_INCLUDE so that
    //      #include "fp.h" resolves to the KB version (kb31_t) not BB.
    //      This ensures kb_symbol_prefix.h is included in all files that include fp.h.
    //   2. DEP_CUDA_COMMON_INCLUDE added last (cuda-common/include).
    //
    // Additionally, nvcc -include force-includes kb_symbol_prefix.h for ALL files
    // (including ntt_params.cu and ntt.cu which do NOT include fp.h) so that:
    //   - __constant__ TWIDDLE arrays get renamed to _KB variants
    //   - All extern "C" NTT launchers get the _kb_ prefix
    if kb_enabled {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let kb_prefix = format!("{}/cuda/kb_include/kb_symbol_prefix.h", manifest_dir);

        // Build a fresh builder (NOT common.clone()) so that kb_include is FIRST.
        CudaBuilder::new()
            .library_name("cuda_kb_all")
            // KB field overrides: MUST come before DEP_CUDA_COMMON_INCLUDE
            .include("cuda/kb_include")        // fp.h = kb31_t, includes kb_symbol_prefix.h
            .include("cuda/supra_kb/include")  // KB NTT params + ntt.cuh wrapper
            .include("cuda/supra/include")     // ntt.cuh helpers
            .include("cuda/include")           // remaining standard headers
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")  // AFTER kb_include
            // Force-include kb_symbol_prefix.h for files that do NOT include fp.h
            // (e.g. ntt_params.cu, ntt.cu). This ensures __constant__ twiddle arrays
            // and NTT launchers get KB-prefixed names, avoiding link-time duplicates.
            .flag("-include")
            .flag(&kb_prefix)
            // NTT files (defines twiddle tables needed by proving kernels)
            .file("cuda/supra/ntt.cu")
            .file("cuda/supra/ntt_params.cu")
            .file("cuda/supra_kb/ntt_bitrev_kb.cu")
            // batch_ntt_small.cu defines DEVICE_NTT_TWIDDLES used by logup_round0 etc.
            .file("cuda/src/batch_ntt_small.cu")
            // Proving kernel files
            .file("cuda/src/logup_zerocheck/logup_round0.cu")
            .file("cuda/src/logup_zerocheck/zerocheck_round0.cu")
            .file("cuda/src/logup_zerocheck/batch_mle.cu")
            .file("cuda/src/logup_zerocheck/mle.cu")
            .file("cuda/src/logup_zerocheck/gkr.cu")
            .file("cuda/src/logup_zerocheck/gkr_input.cu")
            .file("cuda/src/logup_zerocheck/utils.cu")
            .file("cuda/src/logup_zerocheck/batch_mle_monomial.cu")
            .file("cuda/src/whir.cu")
            .file("cuda/src/sumcheck.cu")
            .file("cuda/src/poly.cu")
            .file("cuda/src/matrix.cu")
            .file("cuda/src/mle_interpolate.cu")
            .file("cuda/src/stacked_reduction.cu")
            .watch("cuda/kb_include")
            .watch("cuda/supra_kb")
            .build();
    }
}
