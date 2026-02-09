#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <iterator>
#include <cstdint>

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;
const uint64_t N = (1ULL << 31);
#define block 512   

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void cte_sum(uint64_t* a, uint64_t* blocksum, uint64_t n) {
  int tid = threadIdx.x;
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + tid;
  int bdx = blockDim.x;
  if (gid >= n) return;
  __shared__ uint64_t temp[block];
  temp[tid] = a[gid];
  __syncthreads();
  for (int s = 1; s < bdx; s <<= 1) {
    uint64_t x = 0;
    if (tid >= s) x = temp[tid - s];
    __syncthreads();
    if (tid >= s) temp[tid] += x;
    __syncthreads();
  }
  a[gid] = temp[tid];
  if (tid == bdx - 1) blocksum[blockIdx.x] = temp[tid];
}

__global__ void block_offset(uint64_t* data, const uint64_t* blockOffsets, uint64_t n) {
  uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  uint64_t offset = blockOffsets[blockIdx.x];
  data[gid] += offset;
}

__host__ void check_result(const uint64_t* w_ref, const uint64_t* w_opt, const uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if (w_ref[i] != w_opt[i]) {
      cout << "Differences found between arrays at " << i << "\n";
      assert(false);
    }
  }
  cout << "No differences found between base and test versions\n";
}

__host__ void inclusive_prefix_sum(const uint64_t* input, uint64_t* output) {
  output[0] = input[0];
  for (uint64_t i = 1; i < N; i++)
    output[i] = output[i - 1] + input[i];
}

int main() {
  auto* h_input = new uint64_t[N];
  std::fill_n(h_input, N, 1ULL);
  auto* h_output_cpu = new uint64_t[N];
  inclusive_prefix_sum(h_input, h_output_cpu);
  cout << "CPU last element = " << h_output_cpu[N - 1] << "\n";

  size_t freeMem, totalMem;
  cudaCheckError(cudaMemGetInfo(&freeMem, &totalMem));

  size_t usable = freeMem / 16;
  size_t chunk_elems = usable / sizeof(uint64_t);
  if (chunk_elems < block) chunk_elems = block;
  if (chunk_elems > N) chunk_elems = N;

  uint64_t chunks = (N + chunk_elems - 1) / chunk_elems;
  cout << "Chunks: " << chunks << "\n";

  uint64_t *da, *dblock;
  cudaCheckError(cudaMalloc(&da, chunk_elems * sizeof(uint64_t)));

  int maxGrid = (chunk_elems + block - 1) / block;
  cudaCheckError(cudaMalloc(&dblock, maxGrid * sizeof(uint64_t)));

  uint64_t *hblock = new uint64_t[maxGrid];
  uint64_t *hboffset = new uint64_t[maxGrid];
  auto* out = new uint64_t[N];

  cudaEvent_t start, end,start_kernel,end_kernel;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  uint64_t processed = 0;
  uint64_t running_offset = 0;
  float cpe_part_kernel = 0.0, cpe_kernel=0.0, cpe_total=0.0;
  while (processed < N) {
    uint64_t remain = N - processed;
    uint64_t this_chunk = (remain < chunk_elems) ? remain : chunk_elems;
    cudaMemcpy(da, h_input + processed, this_chunk * sizeof(uint64_t), cudaMemcpyHostToDevice);
    int grid = (this_chunk + block - 1) / block;
    cudaEventRecord(start_kernel);
    cte_sum<<<grid, block>>>(da, dblock, this_chunk);
    cudaDeviceSynchronize();
    cudaMemcpy(hblock, dblock, grid * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    hboffset[0] = 0;
    for (int i = 1; i < grid; ++i) hboffset[i] = hboffset[i - 1] + hblock[i - 1];
    for (int i = 0; i < grid; ++i) hboffset[i] += running_offset;
    cudaMemcpy(dblock, hboffset, grid * sizeof(uint64_t), cudaMemcpyHostToDevice);
    block_offset<<<grid, block>>>(da, dblock, this_chunk);
    cudaDeviceSynchronize();
    cudaEventRecord(end_kernel);
    cudaEventSynchronize(end_kernel);
    cudaEventElapsedTime(&cpe_part_kernel,start_kernel,end_kernel);
    cpe_kernel+=cpe_part_kernel;
    cudaMemcpy(out + processed, da, this_chunk * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    running_offset = out[processed + this_chunk - 1];
    processed += this_chunk;
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&cpe_total, start, end);
  //cout << "Kernel time (copy then execute): " << cpe_kernel << " ms\n";
  cout << "Total time (copy then execute): " << cpe_total << " ms\n";

  check_result(h_output_cpu, out, N);

  cudaFree(da);
  cudaFree(dblock);
  delete[] out;
  delete[] hblock;
  delete[] hboffset;

  da = nullptr;
  dblock = nullptr;

  cudaMallocManaged(&da, N * sizeof(uint64_t));
  std::copy(h_input, h_input + N, da);

  int grid = (int)((N + block - 1) / block);
  cudaMallocManaged(&dblock, (size_t)grid * sizeof(uint64_t));

  int device = 0;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(da, N * sizeof(uint64_t), device, nullptr);
  cudaMemPrefetchAsync(dblock, (size_t)grid * sizeof(uint64_t), device, nullptr);
  cudaDeviceSynchronize();

  cout << "________UVM kernel________\n";
  cudaEventRecord(start);
  cte_sum<<<grid, block>>>(da, dblock, N);
  cudaDeviceSynchronize();
  cudaMemPrefetchAsync(dblock, (size_t)grid * sizeof(uint64_t), cudaCpuDeviceId, nullptr);
  cudaDeviceSynchronize();
  uint64_t prev = 0;
  for (int i = 0; i < grid; ++i) {
    uint64_t s = dblock[i];
    dblock[i] = prev;
    prev += s;
  }
  cudaMemPrefetchAsync(dblock, (size_t)grid * sizeof(uint64_t), device, nullptr);
  cudaDeviceSynchronize();
  block_offset<<<grid, block>>>(da, dblock, N);
  cudaDeviceSynchronize();
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float uvm_time = 0;
  cudaEventElapsedTime(&uvm_time, start, end);
  cout << "Total time (uvm): " << uvm_time << "\n";

  cudaMemPrefetchAsync(da, N * sizeof(uint64_t), cudaCpuDeviceId, nullptr);
  cudaDeviceSynchronize();
  check_result(h_output_cpu, da, N);

  cudaFree(da);
  cudaFree(dblock);
  delete[] h_input;
  delete[] h_output_cpu;

  cout << "UVM prefix sum completed.\n";
  return 0;
}

