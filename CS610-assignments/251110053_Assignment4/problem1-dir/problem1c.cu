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
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = 64;
const uint64_t TILE = 4;
const uint64_t WORKER_THREADS = 4;

// CPU Reference
__host__ void cpuStencil(const double* input, double* output) {
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

// Basic GPU kernel
__global__ void naiveKernel(const double *input, double *output) {
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

// Optimized shared-memory kernel
__global__ void sharedTileKernel(const double* input, double* output) {
    __shared__ double tile[TILE + 2][TILE + 2][TILE + 2];

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int x = blockIdx.x * TILE + tx - 1;
    int y = blockIdx.y * TILE + ty - 1;
    int z = blockIdx.z * TILE + tz - 1;

    if (x >= 0 && x < N && y >= 0 && y < N && z >= 0 && z < N)
        tile[tx][ty][tz] = input[x * N * N + y * N + z];
    else
        tile[tx][ty][tz] = 0.0;

    __syncthreads();

    uint64_t w = (WORKER_THREADS < TILE ? WORKER_THREADS : TILE);
    uint64_t stride = TILE / w;

    if (tx < w && ty < w && tz < w) {
      x = blockIdx.x * TILE + tx * stride;
      y = blockIdx.y * TILE + ty * stride;
      z = blockIdx.z * TILE + tz * stride;

      int xEnd = ((x + stride < N - 1) ? (x + stride) : (N - 1));
      int yEnd = ((y + stride < N - 1) ? (y + stride) : (N - 1));
      int zEnd = ((z + stride < N - 1) ? (z + stride) : (N - 1));
      #pragma unroll
      for(int i = x; i < xEnd; i++) {
        for(int j = y; j < yEnd; j++) {
          for(int k = z; k < zEnd; k++) {
            int xs = i - blockIdx.x * TILE + 1;
            int ys = j - blockIdx.y * TILE + 1;
            int zs = k - blockIdx.z * TILE + 1;

            if(i > 0 && j > 0 && k > 0) {
              output[i*N*N + j*N + k] =
                0.8 * (tile[xs-1][ys][zs] + tile[xs+1][ys][zs]
                     + tile[xs][ys-1][zs] + tile[xs][ys+1][zs]
                     + tile[xs][ys][zs-1] + tile[xs][ys][zs+1]);
            }
          }
        }
      }
    }
}

__host__ void compareResults(const double* reference, const double* test, uint64_t size) {
  double maxDiff = 0.0;
  int mismatchCount = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        uint64_t idx = i*N*N + j*N + k;
        double diff = std::fabs(reference[idx] - test[idx]);
        if (diff > THRESHOLD) {
          mismatchCount++;
          maxDiff = std::max(maxDiff, diff);
        }
      }
    }
  }

  if (mismatchCount > 0) {
    cout << mismatchCount << " differences found. Max diff = " << maxDiff << endl;
  } else {
    cout << "No differences found between base and test versions" << endl;
  }
}

void printMatrix(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k)
        printf("%lf,", A[i * N * N + j * N + k]);
      printf("      ");
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
  uint64_t totalElements = N * N * N;

  cout << "__________shared-memory kernel + Loop Transformations_________\n";

  auto *hostInput = new double[totalElements];
  auto *hostReferenceOutput = new double[totalElements];
  auto *hostGPUOutput = new double[totalElements];

  for(uint64_t i = 0; i < totalElements; i++)
    hostInput[i] = static_cast<double>(rand()) / RAND_MAX;

  double begin = getTime();
  cpuStencil(hostInput, hostReferenceOutput);
  double end = getTime();
  cout << "CPU stencil time = " << (end - begin) * 1000 << " ms\n";

  // GPU timing setup
  cudaEvent_t start, endEvent, startKernel, endKernel;
  float kernelTime = 0, totalGPUTime = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&endEvent);
  cudaEventCreate(&startKernel);
  cudaEventCreate(&endKernel);

  double *deviceInput, *deviceOutput;
  cudaCheckError(cudaMalloc(&deviceInput, totalElements * sizeof(double)));
  cudaCheckError(cudaMalloc(&deviceOutput, totalElements * sizeof(double)));

  cudaEventRecord(start);
  cudaCheckError(cudaMemcpy(deviceInput, hostInput, totalElements * sizeof(double), cudaMemcpyHostToDevice));

  dim3 block(TILE+2, TILE+2, TILE+2);
  dim3 grid((N + TILE - 1)/TILE, (N + TILE - 1)/TILE, (N + TILE - 1)/TILE);

  cudaEventRecord(startKernel);
  sharedTileKernel<<<grid, block>>>(deviceInput, deviceOutput);
  cudaEventRecord(endKernel);
  cudaCheckError(cudaDeviceSynchronize());

  cudaEventElapsedTime(&kernelTime, startKernel, endKernel);
  cudaCheckError(cudaMemcpy(hostGPUOutput, deviceOutput, totalElements * sizeof(double), cudaMemcpyDeviceToHost));
  cudaEventRecord(endEvent);
  cudaCheckError(cudaDeviceSynchronize());
  cudaEventElapsedTime(&totalGPUTime, start, endEvent);

  cout << "Shared memory kernel time = " << kernelTime << " ms\n";
  cout << "Total GPU time (incl copy) = " << totalGPUTime << " ms\n";

  compareResults(hostReferenceOutput, hostGPUOutput, N);

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  delete[] hostInput;
  delete[] hostReferenceOutput;
  delete[] hostGPUOutput;

  return EXIT_SUCCESS;
}

