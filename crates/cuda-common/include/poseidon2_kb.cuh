/*
 * KoalaBear Poseidon2 permutation for GPU.
 *
 * Parameters (matching openvm_stark_sdk::config::koala_bear_poseidon2):
 *   - Field: KoalaBear (p = 2^31 - 2^24 + 1)
 *   - Width:  16
 *   - S-box:  x^3  (alpha = 3)
 *   - External rounds: 8  (4 initial + 4 terminal)
 *   - Internal rounds: 20
 *   - Round constants: same hex values as BabyBear initial (from OPENVM_RC_16_30_U32),
 *     terminal = all zeros, internal = 20 values.
 *
 * Internal matrix diagonal V (matching KoalaBearInternalLayerParameters in Plonky3):
 *   [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
 */

#pragma once

#include "ff/koala_bear.hpp"

namespace poseidon2_kb {

using Fp = kb31_t;

// ── Round constants (standard/canonical form; kb31_t constructor converts to Montgomery) ──

// 4 initial external rounds × 16 elements
// Same values as BabyBear INITIAL_ROUND_CONSTANTS (from OPENVM_RC_16_30_U32 rows 0-3)
static __device__ __constant__ Fp INITIAL_ROUND_CONSTANTS[64] = {
    // row 0
    Fp{0x69cbb6af}, Fp{0x46ad93f9}, Fp{0x60a00f4e}, Fp{0x6b1297cd},
    Fp{0x23189afe}, Fp{0x732e7bef}, Fp{0x72c246de}, Fp{0x2c941900},
    Fp{0x0557eede}, Fp{0x1580496f}, Fp{0x3a3ea77b}, Fp{0x54f3f271},
    Fp{0x0f49b029}, Fp{0x47872fe1}, Fp{0x221e2e36}, Fp{0x1ab7202e},
    // row 1
    Fp{0x487779a6}, Fp{0x3851c9d8}, Fp{0x38dc17c0}, Fp{0x209f8849},
    Fp{0x268dcee8}, Fp{0x350c48da}, Fp{0x5b9ad32e}, Fp{0x0523272b},
    Fp{0x3f89055b}, Fp{0x01e894b2}, Fp{0x13ddedde}, Fp{0x1b2ef334},
    Fp{0x7507d8b4}, Fp{0x6ceeb94e}, Fp{0x52eb6ba2}, Fp{0x50642905},
    // row 2
    Fp{0x05453f3f}, Fp{0x06349efc}, Fp{0x6922787c}, Fp{0x04bfff9c},
    Fp{0x768c714a}, Fp{0x3e9ff21a}, Fp{0x15737c9c}, Fp{0x2229c807},
    Fp{0x0d47f88c}, Fp{0x097e0ecc}, Fp{0x27eadba0}, Fp{0x2d7d29e4},
    Fp{0x3502aaa0}, Fp{0x0f475fd7}, Fp{0x29fbda49}, Fp{0x018afffd},
    // row 3
    Fp{0x0315b618}, Fp{0x6d4497d1}, Fp{0x1b171d9e}, Fp{0x52861abd},
    Fp{0x2e5d0501}, Fp{0x3ec8646c}, Fp{0x6e5f250a}, Fp{0x148ae8e6},
    Fp{0x17f5fa4a}, Fp{0x3e66d284}, Fp{0x0051aa3b}, Fp{0x483f7913},
    Fp{0x2cfe5f15}, Fp{0x023427ca}, Fp{0x2cc78315}, Fp{0x1e36ea47},
};

// 4 terminal external rounds × 16 elements = all zeros
// (KOALABEAR_RC16_EXTERNAL_FINAL is all zeros)
static __device__ __constant__ Fp TERMINAL_ROUND_CONSTANTS[64] = {
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
    Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0},
};

// 20 internal round constants (only index 0 of each state is used)
// From KOALABEAR_RC16_INTERNAL (20 values)
static __device__ __constant__ Fp INTERNAL_ROUND_CONSTANTS[20] = {
    Fp{0x5a8053c0}, Fp{0x693be639}, Fp{0x3858867d}, Fp{0x19334f6b},
    Fp{0x128f0fd8}, Fp{0x4e2b1ccb}, Fp{0x61210ce0}, Fp{0x3c318939},
    Fp{0x0b5b2f22}, Fp{0x2edb11d5}, Fp{0x213effdf}, Fp{0x0cac4606},
    Fp{0x241af16d}, Fp{0x7290a80d}, Fp{0x5b8fe9c4}, Fp{0x58eb1611},
    Fp{0x59986d19}, Fp{0},         Fp{0},         Fp{0},
};

#define CELLS     16
#define CELLS_RATE 8
#define CELLS_OUT  8
#define ROUNDS_FULL     8
#define ROUNDS_HALF_FULL 4
#define ROUNDS_PARTIAL  20  // KoalaBear: 20 internal rounds (vs BabyBear: 13)

// S-box: x^3 for KoalaBear (alpha = 3)
static __device__ __forceinline__ Fp sbox_d3(Fp x) {
    Fp x2 = x * x;
    return x2 * x;
}

static __device__ void do_full_sboxes_kb(Fp *cells) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = sbox_d3(cells[i]);
    }
}

static __device__ void do_partial_sbox_kb(Fp *cells) {
    cells[0] = sbox_d3(cells[0]);
}

// Field helpers: kb31_t uses mont32_t but lacks doubled()/halve() methods
// doubled(x) = x + x;  halve(x) = x * inv2 where inv2 = (P+1)/2 = 1065353217
static __device__ __forceinline__ Fp kb_dbl(Fp x)  { return x + x; }
// KoalaBear inv2 = (2130706433+1)/2 = 1065353217  →  Fp{1065353217} converts to Montgomery
// IMPORTANT: must use int literal (not uint32_t) so kb31_t(int) constructor is called,
// which performs the Montgomery conversion val = (1065353217 << 32) % P.
// FIX_APPLIED_V2: kb_hlv now uses int literal for correct Montgomery form
static __device__ __forceinline__ Fp kb_hlv(Fp x)  { return x * Fp{1065353217}; }

// Multiply a 4-element vector by circulant MDS [2,3,1,1] (same as BabyBear)
static __device__ void multiply_by_4x4_circulant_kb(Fp *x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];
    x[3] = t01233 + kb_dbl(x[0]);
    x[1] = t01123 + kb_dbl(x[2]);
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

static __device__ void multiply_by_m_ext_kb(Fp *old_cells) {
    Fp tmp_sums[4] = {Fp{0}, Fp{0}, Fp{0}, Fp{0}};
    for (uint i = 0; i < CELLS / 4; i++) {
        multiply_by_4x4_circulant_kb(old_cells + i * 4);
        for (uint j = 0; j < 4; j++) {
            tmp_sums[j] = tmp_sums[j] + old_cells[i * 4 + j];
        }
    }
    for (uint i = 0; i < CELLS; i++) {
        old_cells[i] = old_cells[i] + tmp_sums[i % 4];
    }
}

static __device__ void add_round_constants_full_kb(const Fp *rc, Fp *cells, uint round) {
    for (uint i = 0; i < CELLS; i++) {
        cells[i] = cells[i] + rc[round * CELLS + i];
    }
}

// Helper: x / 2^n via repeated halving (uses kb_hlv for kb31_t compatibility)
static __device__ __forceinline__ Fp halve_n(Fp x, int n) {
    for (int i = 0; i < n; i++) x = kb_hlv(x);
    return x;
}

// Internal matrix diagonal:
// V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24]
static __device__ __forceinline__ void internal_layer_mat_mul_kb(Fp *cells) {
    // Compute sum = cells[1] + cells[2] + ... + cells[15]
    Fp part_sum = cells[1];
    for (uint i = 2; i < CELLS; i++) {
        part_sum = part_sum + cells[i];
    }
    Fp full_sum = part_sum + cells[0];

    // cells[0] gets coefficient -2: part_sum - cells[0]
    cells[0] = part_sum - cells[0];

    // cells[1]: +1  → += full_sum
    cells[1] = cells[1] + full_sum;
    // cells[2]: +2  → x+x + full_sum
    cells[2] = kb_dbl(cells[2]) + full_sum;
    // cells[3]: +1/2 → halve + full_sum
    cells[3] = kb_hlv(cells[3]) + full_sum;
    // cells[4]: +3  → dbl+orig + full_sum
    cells[4] = full_sum + kb_dbl(cells[4]) + cells[4];
    // cells[5]: +4  → dbl(dbl) + full_sum
    cells[5] = full_sum + kb_dbl(kb_dbl(cells[5]));
    // cells[6]: -1/2 → full_sum - halve
    cells[6] = full_sum - kb_hlv(cells[6]);
    // cells[7]: -3  → full_sum - (dbl+orig)
    cells[7] = full_sum - (kb_dbl(cells[7]) + cells[7]);
    // cells[8]: -4  → full_sum - dbl(dbl)
    cells[8] = full_sum - kb_dbl(kb_dbl(cells[8]));
    // cells[9]:  +1/2^8 → halve 8 times + full_sum
    cells[9] = halve_n(cells[9], 8) + full_sum;
    // cells[10]: +1/8 = 1/2^3 → halve 3 times + full_sum
    cells[10] = halve_n(cells[10], 3) + full_sum;
    // cells[11]: +1/2^24 → halve 24 times + full_sum
    cells[11] = halve_n(cells[11], 24) + full_sum;
    // cells[12]: -1/2^8 → full_sum - halve_8
    cells[12] = full_sum - halve_n(cells[12], 8);
    // cells[13]: -1/8 = -1/2^3 → full_sum - halve_3
    cells[13] = full_sum - halve_n(cells[13], 3);
    // cells[14]: -1/16 = -1/2^4 → full_sum - halve_4
    cells[14] = full_sum - halve_n(cells[14], 4);
    // cells[15]: -1/2^24 → full_sum - halve_24
    cells[15] = full_sum - halve_n(cells[15], 24);
}

static __device__ void full_round_half_kb(const Fp *rc, Fp *cells, uint round) {
    add_round_constants_full_kb(rc, cells, round);
    do_full_sboxes_kb(cells);
    multiply_by_m_ext_kb(cells);
}

static __device__ void partial_round_kb(Fp *cells, uint round) {
    cells[0] = cells[0] + INTERNAL_ROUND_CONSTANTS[round];
    do_partial_sbox_kb(cells);
    internal_layer_mat_mul_kb(cells);
}

// Full KoalaBear Poseidon2 permutation (width=16)
static __device__ void poseidon2_mix_kb(Fp *cells) {
    // Initial linear layer
    multiply_by_m_ext_kb(cells);

    // 4 initial external (full) rounds
    for (uint i = 0; i < ROUNDS_HALF_FULL; i++) {
        full_round_half_kb(INITIAL_ROUND_CONSTANTS, cells, i);
    }

    // 20 internal (partial) rounds
    for (uint i = 0; i < ROUNDS_PARTIAL; i++) {
        partial_round_kb(cells, i);
    }

    // 4 terminal external (full) rounds
    for (uint r = 0; r < ROUNDS_HALF_FULL; r++) {
        full_round_half_kb(TERMINAL_ROUND_CONSTANTS, cells, r);
    }
}

} // namespace poseidon2_kb
