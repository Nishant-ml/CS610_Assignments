#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <cmath>
#include <assert.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())
#define N 64
#define FILTER_SIZE 3

using std::cerr;
using std::cout;
using std::endl;
const uint64_t TILE = 8;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void cpu_convolution_2D(const float* arrIn, float* arrOut, int dim) {

    int radius = FILTER_SIZE / 2;

    for (int px = 0; px < dim; px++) {
        for (int py = 0; py < dim; py++) {

            float accum = 0.0;
            int tot = FILTER_SIZE * FILTER_SIZE;

            for (int dx = -radius; dx <= radius; ++dx) {
                for (int dy = -radius; dy <= radius; ++dy) {

                    int qx = px + dx;
                    int qy = py + dy;

                    if (qx >= 0 && qx < dim && qy >= 0 && qy < dim)
                        accum += arrIn[qx * dim + qy];
                }
            }
            arrOut[px * dim + py] = accum / tot;
        }
    }
}

void cpu_convolution_3D(const float* arrIn, float* arrOut, int dim) {

    int radius = FILTER_SIZE / 2;

    for (int px = 0; px < dim; px++) {
        for (int py = 0; py < dim; py++) {
            for (int pz = 0; pz < dim; pz++) {

                float accum = 0.0;
                int tot = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE;

                for (int dx = -radius; dx <= radius; ++dx) {
                    for (int dy = -radius; dy <= radius; ++dy) {
                        for (int dz = -radius; dz <= radius; ++dz) {

                            int qx = px + dx;
                            int qy = py + dy;
                            int qz = pz + dz;

                            if (qx >= 0 && qx < dim && qy >= 0 && qy < dim && qz >= 0 && qz < dim)
                                accum += arrIn[qx * dim * dim + qy * dim + qz];
                        }
                    }
                }
                arrOut[px * dim * dim + py * dim + pz] = accum / tot;
            }
        }
    }
}

__global__ void kernel2D_basic(const float* arrIn, float* arrOut, int dim) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = FILTER_SIZE / 2;
    float sum = 0.0f;
    int tot = FILTER_SIZE * FILTER_SIZE;

    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {

            int qx = px + dx;
            int qy = py + dy;

            if (qx >= 0 && qx < dim && qy >= 0 && qy < dim)
                sum += arrIn[qx * dim + qy];
        }
    }

    if (px < dim && py < dim)
        arrOut[px * dim + py] = sum / tot;
}

__global__ void kernel2D_opt(const float* arrIn, float* arrOut, int dim) {

    __shared__ double tile[TILE + FILTER_SIZE][TILE + FILTER_SIZE];

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int radius = FILTER_SIZE / 2;

    int gx = blockIdx.x * TILE + lx - radius;
    int gy = blockIdx.y * TILE + ly - radius;

    if (gx >= 0 && gx < dim && gy >= 0 && gy < dim)
        tile[lx][ly] = arrIn[gx * dim + gy];
    else
        tile[lx][ly] = 0.0f;

    __syncthreads();

    if (lx >= radius && lx < TILE + radius &&
        ly >= radius && ly < TILE + radius) {

        float accum = 0.0f;
        int total = FILTER_SIZE * FILTER_SIZE;

        for (int dx = -radius; dx <= radius; dx++)
            for (int dy = -radius; dy <= radius; dy++)
                accum += tile[lx + dx][ly + dy];

        arrOut[gx * dim + gy] = accum / total;
    }
}

__global__ void kernel3D_basic(const float* arrIn, float* arrOut, int dim) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z * blockDim.z + threadIdx.z;

    int radius = FILTER_SIZE / 2;
    float accum = 0.0f;
    int total = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++)
            for (int dz = -radius; dz <= radius; dz++) {

                int qx = px + dx;
                int qy = py + dy;
                int qz = pz + dz;

                if (qx >= 0 && qx < dim &&
                    qy >= 0 && qy < dim &&
                    qz >= 0 && qz < dim)

                    accum += arrIn[qx * dim * dim + qy * dim + qz];
            }

    if (px < dim && py < dim && pz < dim)
        arrOut[px * dim * dim + py * dim + pz] = accum / total;
}

__global__ void kernel3D_opt(const float* arrIn, float* arrOut, int dim) {

    __shared__ double shTile[(TILE + FILTER_SIZE) *
                             (TILE + FILTER_SIZE) *
                             (TILE + FILTER_SIZE)];

    int tileDim = TILE + FILTER_SIZE;
    int tileArea = tileDim * tileDim;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lz = threadIdx.z;

    int radius = FILTER_SIZE / 2;

    int gx = blockIdx.x * TILE + lx - radius;
    int gy = blockIdx.y * TILE + ly - radius;
    int gz = blockIdx.z * TILE + lz - radius;

    if (gx >= 0 && gx < dim &&
        gy >= 0 && gy < dim &&
        gz >= 0 && gz < dim) {

        shTile[lz * tileArea + ly * tileDim + lx] =
            arrIn[gx * dim * dim + gy * dim + gz];
    }
    else {
        shTile[lz * tileArea + ly * tileDim + lx] = 0.0f;
    }

    __syncthreads();

    if (lx >= radius && lx < TILE + radius &&
        ly >= radius && ly < TILE + radius &&
        lz >= radius && lz < TILE + radius) {

        float accum = 0.0f;
        int tot = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE;

        for (int dx = -radius; dx <= radius; dx++)
            for (int dy = -radius; dy <= radius; dy++)
                for (int dz = -radius; dz <= radius; dz++) {

                    int sx = lx + dx;
                    int sy = ly + dy;
                    int sz = lz + dz;

                    accum += shTile[sz * tileArea + sy * tileDim + sx];
                }

        arrOut[gx * dim * dim + gy * dim + gz] = accum / tot;
    }
}


void check_result(const float* ref, const float* opt, int size) {
    int numdiffs = 0;
    float maxdiff = 0.0f;

    for (int i = 0; i < size; i++) {
        float this_diff = std::fabs(ref[i] - opt[i]);
        if (this_diff > THRESHOLD) {
            numdiffs++;
            if (this_diff > maxdiff) maxdiff = this_diff;
        }
    }

    if (numdiffs > 0)
        cout << numdiffs << " diffs; max = " << maxdiff << endl;
    else
        cout << "No differences between gpu and cpu results.\n";
}




void print2D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "n";
  }
}

void print3D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "n";
    }
    cout << "n";
  }
}

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) {
        cout << "Error return from gettimeofday: " << stat << "\n";
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int driver(int dim) {
    int size = pow(N, dim);
    float* h_input = new float[size];
    float* h_output_cpu = new float[size]();  
    float* h_output_gpu = new float[size]();  

    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    double clkbegin = rtclock();
    if (dim == 2)
        cpu_convolution_2D(h_input, h_output_cpu, N);
    else if (dim == 3)
        cpu_convolution_3D(h_input, h_output_cpu, N);
    double clkend = rtclock();
    cout << "CPU convolution time: " << (clkend - clkbegin) * 1000 << " ms\n";

    float *d_input, *d_output;
    cudaCheckError(cudaMalloc((void**)&d_input, size * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_output, size * sizeof(float)));

    dim3 block2d(8,8);
    dim3 grid2d((N + block2d.x - 1) / block2d.x,
                (N + block2d.y - 1) / block2d.y);

    dim3 block3d(8, 8, 8);
    dim3 grid3d((N + block3d.x - 1) / block3d.x,
                (N + block3d.y - 1) / block3d.y,
                (N + block3d.z - 1) / block3d.z);

    dim3 block2do(TILE+FILTER_SIZE-1, TILE+FILTER_SIZE-1);
    dim3 grid2do((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    dim3 block3do(TILE+FILTER_SIZE-1, TILE+FILTER_SIZE-1, TILE+FILTER_SIZE-1);
    dim3 grid3do((N + TILE - 1) / TILE,
                 (N + TILE - 1) / TILE,
                 (N + TILE - 1) / TILE);

    cudaEvent_t start, end;
    cudaEvent_t start_kernel, end_kernel;
    float kernel_time, overall_time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    

    cudaEventRecord(start);
    cudaCheckError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    if (dim == 2) {
        cudaEventRecord(start_kernel);
        kernel2D_basic<<<grid2d, block2d>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);

    } else if (dim == 3) {
        cudaEventRecord(start_kernel);
        kernel3D_basic<<<grid3d, block3d>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    }

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&overall_time, start, end);

    cout << "Basic Kernel time: " << kernel_time << "ms\n";
    cout << "Basic Kernel time including memory transfers: " << overall_time << "ms\n";

    check_result(h_output_cpu, h_output_gpu, size);

    cudaEventDestroy(start_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(end);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);

    cudaEventRecord(start);
    cudaCheckError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    if (dim == 2) {
        cudaEventRecord(start_kernel);
        kernel2D_opt<<<grid2do, block2do>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);

    } else if (dim == 3) {
        cudaEventRecord(start_kernel);
        kernel3D_opt<<<grid3do, block3do>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    }

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&overall_time, start, end);

    cout << "Optimized Kernel time: " << kernel_time << "ms\n";
    cout << "Optimized Kernel time including memory transfers: " << overall_time << "ms\n";

    check_result(h_output_cpu, h_output_gpu, size);

    cudaFree(d_output);
    cudaFree(d_input);
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    return EXIT_SUCCESS;
}


int main() {
    cout << "____________2D Convolution____________\n";
    assert(driver(2) == EXIT_SUCCESS);
    cout << "____________3D Convolution____________\n";
    assert(driver(3) == EXIT_SUCCESS);

    return EXIT_SUCCESS;
}
