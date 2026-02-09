#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <cmath>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (64);
const uint64_t TILE = 4;
const uint64_t WORKING_THREADS = 4;

// TODO: Edit the function definition as required
__global__ void kernel1(const double *in, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
    out[i * N * N + j * N + k] = 0.8 * (in[(i - 1) * N * N + j * N + k] +
                                        in[(i + 1) * N * N + j * N + k] +
                                        in[i * N * N + (j - 1) * N + k] +
                                        in[i * N * N + (j + 1) * N + k] +
                                        in[i * N * N + j * N + (k - 1)] +
                                        in[i * N * N + j * N + (k + 1)]);
  }
}

__global__ void kernel2(const double* in, double* out) {
    __shared__ double tile[TILE + 2][TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x = blockIdx.x * TILE + tx - 1;
    int y = blockIdx.y * TILE + ty - 1;
    int z = blockIdx.z * TILE + tz - 1;

    if (x >= 0 && x < N && y >= 0 && y < N && z >= 0 && z < N) {
        tile[tx][ty][tz] = in[x * N * N + y * N + z];
    } 
    else{
        tile[tx][ty][tz] = 0.0f;
    }

    __syncthreads();

    uint64_t w = min(WORKING_THREADS, TILE);
    uint64_t stride = TILE/w;
    if(tx < w and ty < w and tz < w){
      x = blockIdx.x * TILE + tx * stride;
      y = blockIdx.y * TILE + ty * stride;
      z = blockIdx.z * TILE + tz * stride;
      #pragma unroll
      for(int i=x; i < min(N-1, x + stride); i++){
        for(int j=y; j < min(N-1, y + stride); j++){
          for(int k=z; k < min(N-1, z + stride); k++){
            int xs = i - blockIdx.x * TILE + 1;
            int ys = j - blockIdx.y * TILE + 1;
            int zs = k - blockIdx.z * TILE + 1;
            if(i > 0 and j > 0 and k > 0){
              out[i*N*N + j*N + k] = 0.8 * (tile[xs-1][ys][zs] + tile[xs+1][ys][zs]
                                            + tile[xs][ys-1][zs] + tile[xs][ys+1][zs]
                                            + tile[xs][ys][zs-1] + tile[xs][ys][zs+1]
                                            );
            }
          }
        }
      }
    }
}

// TODO: Edit the function definition as required
__host__ void stencil(const double *in, double *out) {
  for (uint64_t i =1; i<N -1; i++) {
    for (uint64_t j =1; j<N -1; j++) {
      for (uint64_t k =1; k<N -1; k++) {
        out[i * N * N + j * N + k] = 0.8 * (in[(i - 1) * N * N + j * N + k] +
                                            in[(i + 1) * N * N + j * N + k] +
                                            in[i * N * N + (j - 1) * N + k] +
                                            in[i * N * N + (j + 1) * N + k] +
                                            in[i * N * N + j * N + (k - 1)] +
                                            in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (fabs(this_diff) > maxdiff) {
            maxdiff = fabs(this_diff);
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N * N;
  std::cout << "The program will run CPU version and a pinned + shared memory CUDA Kernel for stencil\n#############Results##############\n";

  // Initialisation
  auto *h_out = new double[SIZE];
  
  double *host_alloc_in;
  double *host_alloc_out;
  cudaCheckError(cudaHostAlloc(&host_alloc_in, SIZE*sizeof(double), cudaHostAllocDefault));
  cudaCheckError(cudaHostAlloc(&host_alloc_out, SIZE*sizeof(double), cudaHostAllocDefault));
  for(int i=0; i<SIZE; i++){
    host_alloc_in[i] = static_cast<double>(rand())/RAND_MAX;;
  }

  double clkbegin = rtclock();
  stencil(host_alloc_in, h_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  cudaEvent_t start_kernel, end_kernel;
  float kernel_time, overall_time;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&end_kernel);

  double *d_in, *d_out;
  cudaCheckError(cudaMalloc(&d_in, SIZE * sizeof(double)));
  cudaCheckError(cudaMalloc(&d_out, SIZE * sizeof(double)));

  cudaEventRecord(start);
  cudaCheckError(cudaMemcpy(d_in, host_alloc_in, SIZE * sizeof(double), cudaMemcpyHostToDevice));

  dim3 block2(TILE+2, TILE+2, TILE+2);
  dim3 grid2((N + TILE - 1)/TILE, (N + TILE - 1)/TILE, (N + TILE - 1)/TILE);
  
  cudaEventRecord(start_kernel);
  kernel2<<<grid2, block2>>> (d_in, d_out);
  cudaEventRecord(end_kernel);
  cudaCheckError(cudaDeviceSynchronize());
  cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);

  cudaCheckError(cudaMemcpy(host_alloc_out, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  cudaEventRecord(end);

  cudaCheckError(cudaDeviceSynchronize());

  cudaEventElapsedTime(&overall_time, start, end);
  std::cout << "Only Kernel time: " << kernel_time << "ms\n";
  std::cout << "Overall time: " << overall_time << "ms\n";

  check_result(host_alloc_out, h_out, N);
  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(host_alloc_in);
  cudaFreeHost(host_alloc_out);

  // TODO: Free memory
  delete [] h_out;

  return EXIT_SUCCESS;
}
