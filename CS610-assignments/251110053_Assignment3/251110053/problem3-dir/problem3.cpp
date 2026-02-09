#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace std;
using HR = chrono::high_resolution_clock;

// Grid sizes
const uint32_t NX = 128;
const uint32_t NY = 128;
const uint32_t NZ = 128;
const uint64_t TOTAL_SIZE = (uint64_t)NX * NY * NZ;
const uint32_t N_ITERATIONS = 100;
const uint64_t INITIAL_VAL = 1000000ULL;

// Scalar baseline
void scalar_3d_gradient(const uint64_t* A, uint64_t* B) {
    const int stride_i = (NY * NZ);
    for (int i = 1; i < (int)NX - 1; ++i)
        for (int j = 0; j < (int)NY; ++j)
            for (int k = 0; k < (int)NZ; ++k) {
                int base_idx = (i * NY * NZ) + j * NZ + k;
                uint64_t A_right = A[base_idx + stride_i];
                uint64_t A_left  = A[base_idx - stride_i];
                B[base_idx] = A_right - A_left;
            }
}

// SSE4.1 version (128-bit vectors, 2x uint64_t per vector)
void sse4_3d_gradient(const uint64_t* A, uint64_t* B) {
    const int stride_i = (NY * NZ);
    const int V = 2; // 2 x 64-bit per SSE vector
    for (int i = 1; i < (int)NX - 1; ++i)
        for (int j = 0; j < (int)NY; ++j) {
            int k = 0;
            for (; k + V < (int)NZ; k += V) {
                int base_idx = (i * NY * NZ) + j * NZ + k;
                const uint64_t* p_right = A + base_idx + stride_i;
                const uint64_t* p_left  = A + base_idx - stride_i;
                __m128i vr = _mm_loadu_si128((const __m128i*)p_right);
                __m128i vl = _mm_loadu_si128((const __m128i*)p_left);
                __m128i vres = _mm_sub_epi64(vr, vl);
                _mm_storeu_si128((__m128i*)(B + base_idx), vres);
            }
            // Scalar cleanup
            for (; k < (int)NZ; ++k) {
                int base_idx = (i * NY * NZ) + j * NZ + k;
                uint64_t A_right = A[base_idx + stride_i];
                uint64_t A_left  = A[base_idx - stride_i];
                B[base_idx] = A_right - A_left;
            }
        }
}

// AVX2 (256-bit, 4x uint64_t per vector)
void avx2_3d_gradient(const uint64_t* A, uint64_t* B) {
    const int stride_i = (NY * NZ);
    const int V = 4; // 4 x 64-bit per AVX2 vector
    for (int i = 1; i < (int)NX - 1; ++i)
        for (int j = 0; j < (int)NY; ++j) {
            int k = 0;
            for (; k + V < (int)NZ; k += V) {
                int base_idx = (i * NY * NZ) + j * NZ + k;
                const uint64_t* p_right = A + base_idx + stride_i;
                const uint64_t* p_left  = A + base_idx - stride_i;
                __m256i vr = _mm256_loadu_si256((const __m256i*)p_right);
                __m256i vl = _mm256_loadu_si256((const __m256i*)p_left);
                __m256i vres = _mm256_sub_epi64(vr, vl);
                _mm256_storeu_si256((__m256i*)(B + base_idx), vres);
            }
            // Scalar cleanup
            for (; k < (int)NZ; ++k) {
                int base_idx = (i * NY * NZ) + j * NZ + k;
                uint64_t A_right = A[base_idx + stride_i];
                uint64_t A_left  = A[base_idx - stride_i];
                B[base_idx] = A_right - A_left;
            }
        }
}

// Utility: checksum
uint64_t compute_checksum(const uint64_t* grid) {
    uint64_t sum = 0;
    for (int i = 1; i < (int)NX - 1; i++)
        for (int j = 0; j < (int)NY; j++)
            for (int k = 0; k < (int)NZ; k++)
                sum += grid[i * NY * NZ + j * NZ + k];
    return sum;
}

int main() {
    uint64_t* i_grid  = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    uint64_t* o_scalar = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    uint64_t* o_sse4   = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));
    uint64_t* o_avx2   = (uint64_t*)aligned_alloc(32, TOTAL_SIZE * sizeof(uint64_t));

    for (uint64_t i = 0; i < NX; i++)
        for (uint64_t j = 0; j < NY; j++)
            for (uint64_t k = 0; k < NZ; k++)
                i_grid[i * NY * NZ + j * NZ + k] = INITIAL_VAL + i + 2*j + 3*k;

    memset(o_scalar, 0, TOTAL_SIZE * sizeof(uint64_t));
    memset(o_sse4,   0, TOTAL_SIZE * sizeof(uint64_t));
    memset(o_avx2,   0, TOTAL_SIZE * sizeof(uint64_t));

    auto start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter)
        scalar_3d_gradient(i_grid, o_scalar);
    auto end = HR::now();
    auto scalar_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    auto scalar_sum = compute_checksum(o_scalar);

    start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter)
        sse4_3d_gradient(i_grid, o_sse4);
    end = HR::now();
    auto sse_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    auto sse_sum = compute_checksum(o_sse4);

    start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter)
        avx2_3d_gradient(i_grid, o_avx2);
    end = HR::now();
    auto avx_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    auto avx_sum = compute_checksum(o_avx2);

    cout << fixed << setprecision(2);
    cout << "Scalar: " << scalar_ms << " ms  | Checksum " << scalar_sum << endl;
    cout << "SSE4  : " << sse_ms    << " ms  | Checksum " << sse_sum << endl;
    cout << "AVX2  : " << avx_ms    << " ms  | Checksum " << avx_sum << endl;
    cout << "Speedup SSE4 vs Scalar: " << (double)scalar_ms/sse_ms << "x\n";
    cout << "Speedup AVX2 vs Scalar: " << (double)scalar_ms/avx_ms << "x\n";

    free(i_grid);
    free(o_scalar);
    free(o_sse4);
    free(o_avx2);
    return 0;
}
