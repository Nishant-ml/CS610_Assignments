#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

#define ITER_CHUNK_SIZE (1 << 25)
#define NUM_VAR 10
#define THRESHOLD (std::numeric_limits<double>::epsilon())

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void kernel_tiled(double *constraints_g,
                             long long *loop_iter_g,
                             double *dev_a_g,
                             double *dev_b_g,
                             long long *dev_output_x,
                             int *dev_output_count,
                             long long chunk_start,
                             long long chunk_end) {
    int tx = threadIdx.x;
    int x = blockIdx.x * blockDim.x + tx;  // Global thread ID
    long long total_threads = (long long)gridDim.x * blockDim.x;

    // Shared-memory tiles
    __shared__ double constraints_s[NUM_VAR];
    __shared__ long long loop_iter_s[NUM_VAR];
    __shared__ double dev_a_s[120]; // 10 * 12
    __shared__ double dev_b_s[30];  // 10 * 3

    // Load constraints and loop_iter
    if (tx < NUM_VAR) {
        constraints_s[tx] = constraints_g[tx];
        loop_iter_s[tx]   = loop_iter_g[tx];
    }
    // Load a (120 doubles)
    if (tx < 120) {
        dev_a_s[tx] = dev_a_g[tx];
    }
    // Load b (30 doubles)
    if (tx < 30) {
        dev_b_s[tx] = dev_b_g[tx];
    }
    __syncthreads();

    long long iter_per_thread =
        (chunk_end - chunk_start + total_threads - 1) / total_threads;

    long long start_iter = chunk_start + iter_per_thread * x;

    long long tmp_end = start_iter + iter_per_thread - 1;
    long long end_iter = (chunk_end - 1 < tmp_end) ? (chunk_end - 1) : tmp_end;

    double x_array[NUM_VAR];
    double q[NUM_VAR] = {0.0};
    long long iter_no[NUM_VAR];

    for (long long iter = start_iter; iter <= end_iter; iter++) {
        long long tmp_iter = iter;

        for (int i = NUM_VAR - 1; i >= 0; i--) {
            iter_no[i] = tmp_iter % loop_iter_s[i];
            tmp_iter /= loop_iter_s[i];
            x_array[i] = dev_b_s[3 * i] + iter_no[i] * dev_b_s[3 * i + 2];
        }

        bool is_valid = true;
        for (int i = 0; i < NUM_VAR; i++) {
            q[i] = 0.0;
            #pragma unroll
            for (int j = 0; j < NUM_VAR; j++) {
                q[i] += dev_a_s[i * 12 + j] * x_array[j];
            }
            q[i] -= dev_a_s[i * 12 + 10];
            is_valid &= (fabs(q[i]) <= constraints_s[i]);
        }
        if (is_valid) {
          int old = atomicAdd(dev_output_count, 1);
          dev_output_x[old] = iter;
        }
    }
}


int main() {
    double a[120], b[30];
    int i, j;

    FILE* fp = fopen("./disp.txt", "r");
    if (fp == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (i = 0; !feof(fp) && i < 120; i++) {
        if (!fscanf(fp, "%lf", &a[i])) {
            printf("Error reading disp.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);

    FILE* fpq = fopen("./grid.txt", "r");
    if (fpq == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (j = 0; !feof(fpq) && j < 30; j++) {
        if (!fscanf(fpq, "%lf", &b[j])) {
            printf("Error reading grid.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(fpq);

    double kk = 0.3;
    double constraints[NUM_VAR];
    for (i = 0; i < NUM_VAR; i++) {
        constraints[i] = kk * a[11 + i * 12];
    }

    long long loop_iter[NUM_VAR+1];
    long long total_iter = 1;
    for (i = 0; i < NUM_VAR; i++) {
        loop_iter[i] = floor((b[3 * i + 1] - b[3 * i]) / b[3 * i + 2]);
        total_iter *= loop_iter[i];
    }
    loop_iter[NUM_VAR] = total_iter;

    long long result_cnt = 0;
    long long *host_output_x = new long long[ITER_CHUNK_SIZE]();

    double *d_a, *d_b, *d_constraints;
    long long *d_loop_iter;
    int *d_output_count;
    long long *d_output_x;

    cudaCheckError(cudaMalloc(&d_a, 120 * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_b, 30 * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_constraints, NUM_VAR * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_loop_iter, (NUM_VAR+1) * sizeof(long long)));
    cudaCheckError(cudaMalloc(&d_output_count, sizeof(int)));
    cudaCheckError(cudaMalloc(&d_output_x, ITER_CHUNK_SIZE * sizeof(long long)));

    cudaCheckError(cudaMemcpy(d_a, a, 120 * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, b, 30 * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_constraints, constraints, NUM_VAR * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_loop_iter, loop_iter, (NUM_VAR+1) * sizeof(long long), cudaMemcpyHostToDevice));

    int block = 512;
    int grid = (1 << 16);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    ofstream output_file("results-tiled.txt");
    output_file << setprecision(6) << fixed;

    for (long long chunk_start = 0; chunk_start < total_iter; chunk_start += ITER_CHUNK_SIZE) {
        long long chunk_end = std::min(chunk_start + (long long)ITER_CHUNK_SIZE, total_iter);

        cudaCheckError(cudaMemset(d_output_count, 0, sizeof(int)));
        kernel_tiled<<<grid, block>>>(d_constraints, d_loop_iter, d_a, d_b,
                                      d_output_x, d_output_count,
                                      chunk_start, chunk_end);
        cudaCheckError(cudaDeviceSynchronize());

        int output_count;
        cudaCheckError(cudaMemcpy(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));

        result_cnt += output_count;
        if (output_count > 0) {
            cudaCheckError(cudaMemcpy(host_output_x, d_output_x, output_count * sizeof(long long),
                                      cudaMemcpyDeviceToHost));
            sort(host_output_x, host_output_x + output_count);
            for (int k = 0; k < output_count; k ++) {
                double x_array[NUM_VAR];
                long long tmp_iter = host_output_x[k];
                for(int i = NUM_VAR - 1; i >= 0; i--){
                    x_array[i] = b[3 * i] + (tmp_iter % loop_iter[i]) * b[3 * i + 2];
                    tmp_iter /= loop_iter[i];
                }
                for (int l = 0; l < NUM_VAR; l++) {
                    output_file << x_array[l];
                    if(l == NUM_VAR-1)
                      output_file << std::endl;
                    else
                      output_file << "\t";
                }
            }
        }
    }

    output_file.close();

    cudaEventRecord(end);
    cudaCheckError(cudaDeviceSynchronize());
    float kernel_time = 0.0;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel time " << kernel_time * 1e-3 << "s\n";
    std::cout << "Result pnts " << result_cnt << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_constraints);
    cudaFree(d_loop_iter);
    cudaFree(d_output_count);
    cudaFree(d_output_x);
    delete [] host_output_x;

    return EXIT_SUCCESS;
}

