#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <omp.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::milliseconds;

const uint64_t TIMESTEPS = 100;
const double W_OWN = (1.0 / 7.0);
const double W_NEIGHBORS = (1.0 / 7.0);
const uint64_t NX = 66;
const uint64_t NY = 66;
const uint64_t NZ = 66;
const uint64_t TOTAL_SIZE = NX * NY * NZ;
const static double EPSILON = std::numeric_limits<double>::epsilon();

void stencil_3d_7pt(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// UNROLLING
void stencil_3d_7pt_unroll(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      int k = 1;
      for (; k+3 < NZ - 1; k += 4) {
        #pragma GCC unroll 4
        for (int u = 0; u < 4; ++u) {
          double neighbors_sum = 0.0;
          neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k+u];
          neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k+u];
          neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k+u];
          neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k+u];
          neighbors_sum += curr[i * NY * NZ + j * NZ + (k+u + 1)];
          neighbors_sum += curr[i * NY * NZ + j * NZ + (k+u - 1)];
          next[i * NY * NZ + j * NZ + (k+u)] =
              W_OWN * curr[i * NY * NZ + j * NZ + (k+u)] +
              W_NEIGHBORS * neighbors_sum;
        }
      }
      for (; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// OMP
void stencil_3d_7pt_omp(const double* curr, double* next) {
  #pragma omp parallel for collapse(2)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

// OMP + UNROLL
void stencil_3d_7pt_omp_unroll(const double* curr, double* next) {
  #pragma omp parallel for collapse(2)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      int k = 1;
      for (; k+3 < NZ - 1; k += 4) {
        #pragma GCC unroll 4
        for (int u = 0; u < 4; ++u) {
          double neighbors_sum = 0.0;
          neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k+u];
          neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k+u];
          neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k+u];
          neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k+u];
          neighbors_sum += curr[i * NY * NZ + j * NZ + (k+u + 1)];
          neighbors_sum += curr[i * NY * NZ + j * NZ + (k+u - 1)];
          next[i * NY * NZ + j * NZ + (k+u)] =
              W_OWN * curr[i * NY * NZ + j * NZ + (k+u)] +
              W_NEIGHBORS * neighbors_sum;
        }
      }
      for (; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

int main() {
  auto* grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto* grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  // SCALAR
  double* current_grid = grid1;
  double* next_grid = grid2;
  auto start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_3d_7pt(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  double final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  double total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) total_sum += current_grid[i];
  cout << "Scalar kernel time: " << duration << " ms" << endl;
  cout << "Final value at center: " << final << "\n";
  cout << "Total sum : " << total_sum << "\n";

  // UNROLL
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_3d_7pt_unroll(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) total_sum += current_grid[i];
  cout << "Unrolled kernel time: " << duration << " ms" << endl;
  cout << "Final value at center: " << final << "\n";
  cout << "Total sum : " << total_sum << "\n";

  // OMP
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_3d_7pt_omp(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) total_sum += current_grid[i];
  cout << "OMP kernel time: " << duration << " ms" << endl;
  cout << "Final value at center: " << final << "\n";
  cout << "Total sum : " << total_sum << "\n";

  // OMP + UNROLL
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;
  current_grid = grid1;
  next_grid = grid2;
  start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_3d_7pt_omp_unroll(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) total_sum += current_grid[i];
  cout << "OMP+Unroll kernel time: " << duration << " ms" << endl;
  cout << "Final value at center: " << final << "\n";
  cout << "Total sum : " << total_sum << "\n";

  delete[] grid1;
  delete[] grid2;
  return EXIT_SUCCESS;
}