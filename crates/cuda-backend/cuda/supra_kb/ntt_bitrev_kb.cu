/*
 * KoalaBear bit-reversal kernel.
 *
 * This file compiles the same bit_rev_impl template as ntt_bitrev.cu, but
 * uses fr_t = kb31_t and exports the _kb_ prefixed entry points.
 * BB-specific functions (_bit_rev_ext, _bit_rev_frac_ext, _bit_rev_frac_ext_build_k2)
 * are stubbed out since the KB extension field type kb31_4_t is not yet available
 * in this library.
 */

#include <cstdint>
#include "launcher.cuh"
#include "ntt/ntt.cuh"

// ── Templated bit-reversal kernel (copied from ntt_bitrev.cu) ─────────────────

template<unsigned int Z_COUNT>
__device__ __forceinline__ unsigned subgroup_sync_mask(uint32_t idx)
{
    if constexpr (Z_COUNT >= WARP_SIZE) {
        return 0xffffffffu;
    } else {
        uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
        uint32_t subgroup_base = lane - idx;
        return (((uint32_t)1 << Z_COUNT) - 1u) << subgroup_base;
    }
}

template<typename T>
struct bit_rev_args {
    T*       d_out;
    const T* d_inp;
    uint32_t lg_domain_size;
    uint32_t padded_poly_size;
    uint32_t poly_count;
};

template<typename T, uint32_t Z_COUNT>
__global__ void bit_rev_permutation(T* d_out, const T* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    const uint32_t poly_idx = blockIdx.y + blockIdx.z * gridDim.y;
    if (poly_idx >= poly_count) return;

    const T* inp = d_inp + (size_t)poly_idx * padded_poly_size;
    T* out = d_out + (size_t)poly_idx * padded_poly_size;

    const uint32_t domain_size = 1u << lg_domain_size;
    const uint32_t tid = (threadIdx.x + blockDim.x * blockIdx.x) * Z_COUNT;
    if (tid >= domain_size) return;

    uint32_t rev = __brev(tid) >> (32 - lg_domain_size);
    if (tid < rev) {
        T tmp[Z_COUNT];
        for (uint32_t z = 0; z < Z_COUNT; z++) {
            tmp[z] = inp[tid + z];
        }
        T tmp2[Z_COUNT];
        for (uint32_t z = 0; z < Z_COUNT; z++) {
            tmp2[z] = inp[rev + z];
        }
        for (uint32_t z = 0; z < Z_COUNT; z++) {
            out[tid + z] = tmp2[z];
        }
        for (uint32_t z = 0; z < Z_COUNT; z++) {
            out[rev + z] = tmp[z];
        }
    } else if (tid == rev && d_out != d_inp) {
        for (uint32_t z = 0; z < Z_COUNT; z++) {
            out[tid + z] = inp[tid + z];
        }
    }
}

template<typename T>
static int bit_rev_impl(T* d_out, const T* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    if (lg_domain_size > MAX_LG_DOMAIN_SIZE || poly_count == 0)
        return cudaErrorInvalidValue;

    const uint32_t domain_size = 1u << lg_domain_size;
    const uint32_t block_size = 256;
    const uint32_t num_blocks = (domain_size / 1 + block_size - 1) / block_size;

    const uint32_t MAX_Y = 65535u;
    const uint32_t grid_y = poly_count < MAX_Y ? poly_count : MAX_Y;
    const uint32_t grid_z = (poly_count + grid_y - 1) / grid_y;

    bit_rev_permutation<T, 1><<<dim3(num_blocks, grid_y, grid_z), block_size>>>(
        d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
    return CHECK_KERNEL();
}

// ── Exported entry points ─────────────────────────────────────────────────────

extern "C" int _kb_bit_rev(fr_t* d_out, const fr_t* d_inp,
    uint32_t lg_domain_size, uint32_t padded_poly_size, uint32_t poly_count)
{
    return bit_rev_impl(d_out, d_inp, lg_domain_size, padded_poly_size, poly_count);
}

// KB extension-field and frac-ext bit-reversal are not yet implemented.
// These stubs satisfy the linker; the Rust side never calls them for KB.
extern "C" int _kb_bit_rev_ext(void*, const void*, uint32_t, uint32_t, uint32_t)
{
    return cudaErrorNotSupported;
}

extern "C" int _kb_bit_rev_frac_ext(void*, const void*, uint32_t, uint32_t, uint32_t)
{
    return cudaErrorNotSupported;
}

extern "C" int _kb_bit_rev_frac_ext_build_k2(void*, uint32_t, fr_t)
{
    return cudaErrorNotSupported;
}
