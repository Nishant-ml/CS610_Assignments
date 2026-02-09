#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <cmath>
#include <limits>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

const uint64_t N = 64;

// =========== CPU Reference Implementation ===========
__host__ void cpuStencil(const double *input, double *output) {
  for (uint64_t i = 1; i < N - 1; i++) {
    for (uint64_t j = 1; j < N - 1; j++) {
      for (uint64_t k = 1; k < N - 1; k++) {
        output[i * N * N + j * N + k] =
            0.8 * (input[(i - 1) * N * N + j * N + k] +
                   input[(i + 1) * N * N + j * N + k] +
                   input[i * N * N + (j - 1) * N + k] +
                   input[i * N * N + (j + 1) * N + k] +
                   input[i * N * N + j * N + (k - 1)] +
                   input[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

// =========== Naive GPU Kernel ===========
__global__ void gpuStencilKernel(const double *input, double *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
    output[i * N * N + j * N + k] =
        0.8 * (input[(i - 1) * N * N + j * N + k] +
               input[(i + 1) * N * N + j * N + k] +
               input[i * N * N + (j - 1) * N + k] +
               input[i * N * N + (j + 1) * N + k] +
               input[i * N * N + j * N + (k - 1)] +
               input[i * N * N + j * N + (k + 1)]);
  }
}

// =========== Results Checker ===========
__host__ void compareResults(const double* ref, const double* test, uint64_t size) {
  double maxDiff = 0.0;
  int mismatchCount = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        uint64_t idx = i + N * j + N * N * k;
        double diff = std::fabs(ref[idx] - test[idx]);
        if (diff > THRESHOLD) {
          mismatchCount++;
          maxDiff = std::max(maxDiff, diff);
        }
      }
    }
  }

  if (mismatchCount > 0) {
    cout << mismatchCount << " mismatches found, max diff = " << maxDiff << endl;
  } else {
    cout << "No differences found between base and test versions" << endl;
  }
}

// optional utility
void printMatrix(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k)
        printf("%lf, ", A[i * N * N + j * N + k]);
      printf("\n");
    }
    printf("\n");
  }
}

double getTime() {
  struct timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + t.tv_usec * 1e-6;
}

int main() {
  uint64_t numElements = N * N * N;

  cout << "Running CPU stencil and naive CUDA stencil kernel...\n\n";

  // Allocate host memory
  auto *hostInput  = new double[numElements];
  auto *hostOutputRef = new double[numElements];
  auto *hostOutputGPU = new double[numElements];

  for (uint64_t i = 0; i < numElements; i++)
    hostInput[i] = static_cast<double>(rand()) / RAND_MAX;

  // =============== CPU Execution ===============
  double startCPU = getTime();
  cpuStencil(hostInput, hostOutputRef);
  double endCPU = getTime();
  cout << "CPU time = " << (endCPU - startCPU) * 1000 << " ms\n";

  // =============== GPU Execution ===============
  cudaEvent_t startEvent, endEvent, kernelStart, kernelEnd;
  float kernelTime = 0, totalGPUTime = 0;

  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);
  cudaEventCreate(&kernelStart);
  cudaEventCreate(&kernelEnd);

  double *devInput, *devOutput;
  cudaCheckError(cudaMalloc(&devInput, numElements * sizeof(double)));
  cudaCheckError(cudaMalloc(&devOutput, numElements * sizeof(double)));

  cudaEventRecord(startEvent);

  cudaCheckError(cudaMemcpy(devInput, hostInput,
      numElements * sizeof(double), cudaMemcpyHostToDevice));

  dim3 block(8, 8, 8);
  dim3 grid((N + block.x - 1) / block.x,
            (N + block.y - 1) / block.y,
            (N + block.z - 1) / block.z);

  cudaEventRecord(kernelStart);
  gpuStencilKernel<<<grid, block>>>(devInput, devOutput);
  cudaEventRecord(kernelEnd);

  cudaCheckError(cudaDeviceSynchronize());
  cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd);

  cudaCheckError(cudaMemcpy(hostOutputGPU, devOutput,
      numElements * sizeof(double), cudaMemcpyDeviceToHost));

  cudaEventRecord(endEvent);
  cudaEventSynchronize(endEvent);   // <-- FIX FOR 0 ms BUG
  cudaEventElapsedTime(&totalGPUTime, startEvent, endEvent);

  cout << "Kernel only time = " << kernelTime << " ms\n";
  cout << "Total GPU time (incl copy) = " << totalGPUTime << " ms\n";

  compareResults(hostOutputRef, hostOutputGPU, N);

  cudaFree(devInput);
  cudaFree(devOutput);

  delete[] hostInput;
  delete[] hostOutputRef;
  delete[] hostOutputGPU;

  return EXIT_SUCCESS;
}

