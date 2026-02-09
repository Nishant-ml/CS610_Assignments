#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define NSEC_SEC_MUL (1.0e9)
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

__global__ void solverKernel(double *gp_lim, long long *gp_dims, double *gp_matA, double *gp_matB, long long *gp_hits, int *gp_hitCount, long long cr_begin, long long cr_end) {
    int lx = threadIdx.x;
    int gx = blockIdx.x * blockDim.x + lx;
    long long tthreads = gridDim.x * blockDim.x;
    long long seg = (cr_end - cr_begin + tthreads - 1) / tthreads;
    long long st = cr_begin + seg * gx;
    long long en = min(cr_end - 1, st + seg - 1);

    double vals[NUM_VAR];
    double chk[NUM_VAR] = {0.0};
    long long idxv[NUM_VAR];

    for (long long it = st; it <= en; it++) {
        long long tmp = it;
        for (int i = NUM_VAR - 1; i >= 0; i--) {
            idxv[i] = tmp % gp_dims[i];
            tmp /= gp_dims[i];
            vals[i] = gp_matB[3 * i] + idxv[i] * gp_matB[3 * i + 2];
        }

        bool ok = true;
        for (int i = 0; i < NUM_VAR; i++) {
            chk[i] = 0.0;
            for (int j = 0; j < NUM_VAR; j++) {
                chk[i] += gp_matA[i * 12 + j] * vals[j];
            }
            chk[i] -= gp_matA[i * 12 + 10];
            ok &= (fabs(chk[i]) <= gp_lim[i]);
        }
        if (ok) {
          int old = atomicAdd(gp_hitCount, 1);
          gp_hits[old] = it;
        }
    }
}

int main() {
    double h_matA[120], h_matB[30];
    int i, j;

    FILE* f0 = fopen("./disp.txt", "r");
    if (f0 == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (i = 0; !feof(f0) && i < 120; i++) {
        if (!fscanf(f0, "%lf", &h_matA[i])) {
            printf("Error reading disp.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(f0);

    FILE* f1 = fopen("./grid.txt", "r");
    if (f1 == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (j = 0; !feof(f1) && j < 30; j++) {
        if (!fscanf(f1, "%lf", &h_matB[j])) {
            printf("Error reading grid.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(f1);

    double ratio = 0.3;
    double h_thresholds[NUM_VAR];
    for (i = 0; i < NUM_VAR; i++) {
        h_thresholds[i] = ratio * h_matA[11 + i * 12];
    }

    long long h_dims[NUM_VAR+1];
    long long totalIters = 1;
    for (i = 0; i < NUM_VAR; i++) {
        h_dims[i] = floor((h_matB[3 * i + 1] - h_matB[3 * i]) / h_matB[3 * i + 2]);
        totalIters *= h_dims[i];
    }
    h_dims[NUM_VAR] = totalIters;
    long long solutions = 0;
    long long *h_solBuffer = new long long[ITER_CHUNK_SIZE]();

    double *dA, *dB, *dThresh;
    long long *dDims;
    int *dCount;
    long long *dSol;

    cudaMalloc(&dA, 120 * sizeof(double));
    cudaMalloc(&dB, 30 * sizeof(double));
    cudaMalloc(&dThresh, NUM_VAR * sizeof(double));
    cudaMalloc(&dDims, (NUM_VAR+1) * sizeof(long long));
    cudaMalloc(&dCount, sizeof(int));
    cudaMalloc(&dSol, ITER_CHUNK_SIZE * sizeof(long long));

    cudaMemcpy(dA, h_matA, 120 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, h_matB, 30 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dThresh, h_thresholds, NUM_VAR * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dDims, h_dims, (NUM_VAR+1) * sizeof(long long), cudaMemcpyHostToDevice);

    int blksz = 512;
    int grsz = (1 << 16);

    cudaEvent_t eStart, eEnd;
    cudaEventCreate(&eStart);
    cudaEventCreate(&eEnd);
    cudaEventRecord(eStart);

    ofstream outFile("results-va.txt");
    outFile << setprecision(6) << fixed;

    for (long long cs = 0; cs < totalIters; cs += ITER_CHUNK_SIZE) {
        long long ce = min(cs + ITER_CHUNK_SIZE, totalIters);
        cudaMemset(dCount, 0, sizeof(int));
        solverKernel<<<grsz, blksz>>>(dThresh, dDims, dA, dB, dSol, dCount, cs, ce);
        cudaDeviceSynchronize();

        int outCount;
        cudaMemcpy(&outCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);

        solutions += outCount;
        if (outCount > 0) {
            cudaMemcpy(h_solBuffer, dSol, outCount * sizeof(long long), cudaMemcpyDeviceToHost);
            sort(h_solBuffer, h_solBuffer + outCount);
            for (int k = 0; k < outCount; k ++) {
                double locals[NUM_VAR];
                long long tmp = h_solBuffer[k];
                for(int i=NUM_VAR-1; i>=0; i--){
                    locals[i] = h_matB[3 * i] + (tmp % h_dims[i]) * h_matB[3 * i + 2];
                    tmp /= h_dims[i];
                }
                for (int l = 0; l < NUM_VAR; l++) {
                    outFile << locals[l];
                    if(l == NUM_VAR-1)
                      outFile << std::endl;
                    else
                      outFile << "\t";
                }
            }
        }
    }

    outFile.close();

    cudaEventRecord(eEnd);
    cudaDeviceSynchronize();
    float t = 0.0;
    cudaEventElapsedTime(&t, eStart, eEnd);
    std::cout << "Kernel time " << t * 1e-3 << "s\n";
    std::cout << "Result pnts " << solutions << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dThresh);
    cudaFree(dDims);
    cudaFree(dCount);
    cudaFree(dSol);
    delete [] h_solBuffer;
    return EXIT_SUCCESS;
}

