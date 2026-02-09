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

__global__ void kernel(double *g_limits, long long *g_shape, double *g_matA, double *g_matB, long long *g_hits, int *g_hitCount, long long seg_begin, long long seg_end) {
    int t = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + t;
    long long totalT = gridDim.x * blockDim.x;
    long long span = (seg_end - seg_begin + totalT - 1) / totalT;
    long long i0 = seg_begin + span * gid;
    long long i1 = min(seg_end - 1, i0 + span - 1);

    double vals[NUM_VAR];
    double checkVec[NUM_VAR] = {0.0};
    long long idxMap[NUM_VAR];

    for (long long idx = i0; idx <= i1; idx++) {
        long long tmp = idx;

        for (int p = NUM_VAR - 1; p >= 0; p--) {
            idxMap[p] = tmp % g_shape[p];
            tmp /= g_shape[p];
            vals[p] = g_matB[3 * p] + idxMap[p] * g_matB[3 * p + 2];
        }

        bool ok = true;
        for (int r = 0; r < NUM_VAR; r++) {
            checkVec[r] = 0.0;
            for (int c = 0; c < NUM_VAR; c++) {
                checkVec[r] += g_matA[r * 12 + c] * vals[c];
            }
            checkVec[r] -= g_matA[r * 12 + 10];
            ok &= (fabs(checkVec[r]) <= g_limits[r]);
        }
        if (ok) {
          int old = atomicAdd(g_hitCount, 1);
          g_hits[old] = idx;
        }
    }
}

int main() {
    double *h_A, *h_B;
    int p, q;
    cudaCheckError(cudaMallocManaged(&h_A, 120*sizeof(double)));
    cudaCheckError(cudaMallocManaged(&h_B, 30*sizeof(double)));

    FILE* fdisp = fopen("./disp.txt", "r");
    if (fdisp == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (p = 0; !feof(fdisp) && p < 120; p++) {
        if (!fscanf(fdisp, "%lf", &h_A[p])) {
            printf("Error reading disp.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(fdisp);

    FILE* fgrid = fopen("./grid.txt", "r");
    if (fgrid == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (q = 0; !feof(fgrid) && q < 30; q++) {
        if (!fscanf(fgrid, "%lf", &h_B[q])) {
            printf("Error reading grid.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(fgrid);

    double kscale = 0.3;
    double *g_bounds;
    cudaCheckError(cudaMallocManaged(&g_bounds, NUM_VAR * sizeof(double)));
    for (p = 0; p < NUM_VAR; p++) {
        g_bounds[p] = kscale * h_A[11 + p * 12];
    }

    long long *g_dim;
    cudaCheckError(cudaMallocManaged(&g_dim, (NUM_VAR+1)*sizeof(double)));
    long long totalRange = 1;
    for (p = 0; p < NUM_VAR; p++) {
        g_dim[p] = floor((h_B[3 * p + 1] - h_B[3 * p]) / h_B[3 * p + 2]);
        totalRange *= g_dim[p];
    }
    g_dim[NUM_VAR] = totalRange;

    int dev;
    cudaGetDevice(&dev);
    cudaMemPrefetchAsync(h_A, 120 * sizeof(double), dev);
    cudaMemPrefetchAsync(h_B, 30 * sizeof(double), dev);
    cudaCheckError(cudaMemAdvise(h_A, 120 * sizeof(double), cudaMemAdviseSetReadMostly, dev));
    cudaCheckError(cudaMemAdvise(h_B, 30 * sizeof(double), cudaMemAdviseSetReadMostly, dev));
    cudaCheckError(cudaMemAdvise(g_bounds, NUM_VAR * sizeof(double), cudaMemAdviseSetReadMostly, dev));
    cudaCheckError(cudaMemAdvise(g_dim, (NUM_VAR + 1) * sizeof(long long), cudaMemAdviseSetReadMostly, dev));

    long long hitCount = 0;
    int *d_counter;
    long long *d_results;
    cudaMallocManaged(&d_counter, sizeof(int));
    cudaMallocManaged(&d_results, ITER_CHUNK_SIZE * sizeof(long long));

    int threads = 512;
    int blocks = (1 << 16);

    cudaEvent_t ts, te;
    cudaEventCreate(&ts);
    cudaEventCreate(&te);
    cudaEventRecord(ts);

    ofstream fout("results-vc.txt");
    fout << setprecision(6) << fixed;

    for (long long segStart = 0; segStart < totalRange; segStart += ITER_CHUNK_SIZE) {
        long long segEnd = min(segStart + ITER_CHUNK_SIZE, totalRange);

        *d_counter = 0;
        kernel<<<blocks, threads>>>(g_bounds, g_dim, h_A, h_B, d_results, d_counter, segStart, segEnd);
        cudaDeviceSynchronize();

        hitCount += (*d_counter);
        if (*d_counter > 0) {
            sort(d_results, d_results + (*d_counter));
            for (int t = 0; t < (*d_counter); t ++) {
                double xvals[NUM_VAR];
                long long tmp = d_results[t];
                for(int p=NUM_VAR-1; p>=0; p--){
                    xvals[p] = h_B[3 * p] + (tmp % g_dim[p]) * h_B[3 * p + 2];
                    tmp /= g_dim[p];
                }
                for (int r = 0; r < NUM_VAR; r++) {
                    fout << xvals[r];
                    if(r == NUM_VAR-1)
                      fout << std::endl;
                    else
                      fout << "\t";
                }
            }
        }
    }

    fout.close();

    cudaEventRecord(te);
    cudaDeviceSynchronize();
    float ktime = 0.0;
    cudaEventElapsedTime(&ktime, ts, te);
    std::cout << "Kernel time " << ktime * 1e-3 << "s\n";
    std::cout << "Result pnts " << hitCount << std::endl;

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(g_bounds);
    cudaFree(g_dim);
    cudaFree(d_counter);
    cudaFree(d_results);
    return EXIT_SUCCESS;
}

