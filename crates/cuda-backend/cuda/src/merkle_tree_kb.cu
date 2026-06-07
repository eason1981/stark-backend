/**
 * KoalaBear Poseidon2 Merkle tree commitment kernels.
 *
 * Replicates merkle_tree.cu but uses kb31_t (KoalaBear) and poseidon2_kb::poseidon2_mix_kb.
 * Exported symbols are prefixed with _kb_ to avoid link conflicts with BabyBear versions.
 */

#include "ff/koala_bear.hpp"
#include "launcher.cuh"
#include "poseidon2_kb.cuh"
#include <cstddef>
#include <cstdint>

using Fp = kb31_t;

// poseidon2_kb.cuh defines CELLS=16, CELLS_OUT=8, CELLS_RATE=8 as preprocessor macros
// (they're #define, not namespace members, so `using` is not needed)

struct kb_digest_t {
    Fp cells[CELLS_OUT];  // [KoalaBear; 8]
};

// KoalaBear extension field: 4 base-field elements
struct KbFpExt {
    Fp elems[4];
};

// ── Row hash + merkle combine kernel (base field) ──────────────────────────

__global__ void kb_poseidon2_compressing_row_hashes_kernel(
    kb_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[];
    Fp *shared = reinterpret_cast<Fp *>(smem);
    const uint32_t shared_stride = blockDim.x * (blockDim.y >> 1);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const uint32_t row = leaf_idx * query_stride + stride_idx;

    size_t used = 0;
    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);

    if (stride_idx < query_stride) {
        for (int col = 0; col < (int)width; col++) {
            cells[used++] = matrix[col * height + row];
            if (used == CELLS_RATE) {
                poseidon2_kb::poseidon2_mix_kb(cells);
                used = 0;
            }
        }
        if (used != 0) poseidon2_kb::poseidon2_mix_kb(cells);
    }

    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        auto shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;
        if ((leaf_idx & mask) == (1u << layer)) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                shared[i * shared_stride + shared_offset] = cells[i];
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            poseidon2_kb::poseidon2_mix_kb(cells);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++)
            out[stride_idx].cells[i] = cells[i];
    }
}

// ── Row hash kernel (extension field) ─────────────────────────────────────

__global__ void kb_poseidon2_compressing_row_hashes_ext_kernel(
    kb_digest_t *out,
    const KbFpExt *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[];
    Fp *shared = reinterpret_cast<Fp *>(smem);
    const uint32_t shared_stride = blockDim.x * (blockDim.y >> 1);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const uint32_t row = leaf_idx * query_stride + stride_idx;

    size_t used = 0;
    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);

    if (stride_idx < query_stride) {
        for (int col = 0; col < (int)width; col++) {
            KbFpExt ext = matrix[col * height + row];
            // Absorb 4 base-field elements per extension field element
            for (int k = 0; k < 4; k++) {
                cells[used++] = ext.elems[k];
                if (used == CELLS_RATE) {
                    poseidon2_kb::poseidon2_mix_kb(cells);
                    used = 0;
                }
            }
        }
        if (used != 0) poseidon2_kb::poseidon2_mix_kb(cells);
    }

    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        auto shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;
        if ((leaf_idx & mask) == (1u << layer)) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                shared[i * shared_stride + shared_offset] = cells[i];
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            poseidon2_kb::poseidon2_mix_kb(cells);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++)
            out[stride_idx].cells[i] = cells[i];
    }
}

// ── Adjacent compress layer ────────────────────────────────────────────────

__global__ void kb_poseidon2_strided_compress_layer_kernel(
    kb_digest_t *output,
    const kb_digest_t *prev_layer,
    size_t output_size,
    size_t stride
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    // Non-overlapping pair indexing: pair (2*idx, 2*idx+1) for stride=1
    // Generalized: x = idx / stride, y = idx % stride → (2*x*stride+y, (2*x+1)*stride+y)
    uint32_t x = idx / stride;
    uint32_t y = idx % stride;
    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) cells[i] = prev_layer[2 * x * stride + y].cells[i];
#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) cells[CELLS_OUT + i] = prev_layer[(2 * x + 1) * stride + y].cells[i];

    poseidon2_kb::poseidon2_mix_kb(cells);

#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) output[idx].cells[i] = cells[i];
}

// ── Extern C launchers ─────────────────────────────────────────────────────

static inline size_t div_ceil_kb(size_t a, size_t b) { return (a + b - 1) / b; }

extern "C" int _kb_poseidon2_compressing_row_hashes(
    kb_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    if (log_rows_per_query > 10) return cudaErrorInvalidValue;
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    size_t shared_stride = block.x * div_ceil_kb(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    kb_poseidon2_compressing_row_hashes_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query);
    return CHECK_KERNEL();
}

extern "C" int _kb_poseidon2_compressing_row_hashes_ext(
    kb_digest_t *out,
    const KbFpExt *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    if (log_rows_per_query > 10) return cudaErrorInvalidValue;
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    size_t shared_stride = block.x * div_ceil_kb(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    kb_poseidon2_compressing_row_hashes_ext_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query);
    return CHECK_KERNEL();
}

extern "C" int _kb_poseidon2_adjacent_compress_layer(
    kb_digest_t *output,
    const kb_digest_t *prev_layer,
    size_t output_size
) {
    auto [grid, block] = kernel_launch_params(output_size);
    kb_poseidon2_strided_compress_layer_kernel<<<grid, block>>>(
        output, prev_layer, output_size, 1);
    return CHECK_KERNEL();
}
