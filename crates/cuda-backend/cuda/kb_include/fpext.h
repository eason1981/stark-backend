/*
 * KoalaBear extension field override for GPU proving kernels.
 *
 * Mirrors cuda-common/include/fpext.h but wraps kb31_4_t instead of bb31_4_t.
 * Included via -I cuda/kb_include BEFORE -I (cuda-common include path) so that
 * FpExt = kb31_4_t for all KB kernel code.
 *
 * KoalaBear^4 = KoalaBear[x] / (x^4 - 3)
 */

#pragma once

#include "fp.h"   // KB override of fp.h => Fp = kb31_t

/// FpExt: element of GF(p^4) = GF(p)[X] / (X^4 - 3).
/// Represented as  elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*X^3
/// where elems[i] are Fp = kb31_t elements.
struct FpExt {
    union {
        Fp       elems[4];
        kb31_4_t rep;
    };

    /// Default constructor: zero element
    __device__ FpExt() : rep(kb31_t{0u}) {}

    /// Initialize from uint32_t
    __device__ explicit FpExt(uint32_t x) : rep(kb31_t{x}) {}

    /// Convert from Fp to FpExt
    __device__ explicit FpExt(Fp x) : rep(static_cast<kb31_t>(x)) {}

    /// Explicitly construct an FpExt from four Fp parts
    __device__ FpExt(Fp a, Fp b, Fp c, Fp d) {
        elems[0] = a;
        elems[1] = b;
        elems[2] = c;
        elems[3] = d;
    }

    __device__ FpExt operator+=(FpExt rhs) {
        rep += rhs.rep;
        return *this;
    }
    __device__ FpExt operator-=(FpExt rhs) {
        rep -= rhs.rep;
        return *this;
    }
    __device__ FpExt operator+(FpExt rhs) const {
        FpExt result = *this;
        result += rhs;
        return result;
    }
    __device__ FpExt operator-(FpExt rhs) const {
        FpExt result = *this;
        result -= rhs;
        return result;
    }
    __device__ FpExt operator-(Fp rhs) const {
        FpExt result = *this;
        result.elems[0] -= rhs;
        return result;
    }
    __device__ FpExt operator-() const { return FpExt() - *this; }

    __device__ FpExt operator*=(Fp rhs) {
        rep *= static_cast<kb31_t>(rhs);
        return *this;
    }
    __device__ FpExt operator*(Fp rhs) const {
        FpExt result = *this;
        result *= rhs;
        return result;
    }
    __device__ FpExt operator*=(FpExt rhs) {
        rep *= rhs.rep;
        return *this;
    }
    __device__ FpExt operator*(FpExt rhs) const {
        FpExt result = *this;
        result *= rhs;
        return result;
    }

    __device__ bool operator==(FpExt rhs) const { return rep == rhs.rep; }
    __device__ bool operator!=(FpExt rhs) const { return rep != rhs.rep; }

    __device__ Fp constPart() const { return elems[0]; }
};

/// Overload for case where LHS is Fp
__device__ inline FpExt operator*(Fp a, FpExt b) { return b * a; }

/// Raise an FpExt to a power
__device__ inline FpExt pow(FpExt x, uint32_t n) {
    FpExt r; r.rep = x.rep ^ n;
    return r;
}

template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
__device__ inline FpExt pow(FpExt x, I n) {
    return pow(x, static_cast<uint32_t>(n));
}

/// Compute the multiplicative inverse of an FpExt using kb31_4_t::reciprocal().
__device__ inline FpExt inv(FpExt in) {
    FpExt result;
    result.rep = in.rep.reciprocal();
    return result;
}

static_assert(sizeof(FpExt) == 16, "FpExt must be 16 bytes");
static_assert(sizeof(FpExt) == sizeof(kb31_4_t),   "FpExt and kb31_4_t sizes must match");
static_assert(alignof(FpExt) == alignof(kb31_4_t), "FpExt and kb31_4_t align must match");
