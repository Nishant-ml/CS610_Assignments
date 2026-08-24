# CS610: Programming for Performance — Course Projects & Implementations

Comprehensive implementations, performance profiling, and parallel optimizations completed as part of the **CS610 (Programming for Performance)** coursework at IIT Kanpur. This repository spans cache memory hierarchy analysis, multithreaded concurrency and synchronization, loop transformations and SIMD vectorization (SSE4.1/AVX2/OpenMP), and massively parallel GPU computing with CUDA and Unified Virtual Memory (UVM).

---

## Repository Structure

```text
.
├── Assignment_1/
│   ├── problem1_cache_analysis/       # Analytical cache miss models (strided access)
│   ├── problem2_matmul_analysis/       # Analytical models for kij/jki matrix multiplication
│   └── problem3_producer_consumer/     # Multithreaded bounded FIFO queue in C++17
├── Assignment_2/
│   ├── problem1_convolution_3d/        # Naive vs. L2-cache blocked 3D convolution (PAPI profiling)
│   ├── problem2_cache_coherence/       # perf c2c false/true sharing elimination
│   └── problem3_custom_locks/          # Custom CAS/atomic locks (Filter, Bakery, Spin, Ticket, Array Q)
├── Assignment_3/
│   ├── problem1_stencil_openmp/        # 3D Stencil optimization (loop unrolling + OpenMP collapse)
│   ├── problem2_avx2_prefix_sum/       # In-register AVX2 inclusive scan vectorization
│   ├── problem3_gradient_simd/         # 3D Gradient computation using SSE4.1 & AVX2 intrinsics
│   └── problem4_loop_restructuring/    # LICM + Dynamic OpenMP parallelization on 10D loop nest
└── Assignment_4/
    ├── problem1_cuda_stencil_3d/       # 7-Point 3D Stencil (Shared Memory Tiling, Pinned Memory)
    ├── problem2_cuda_prefix_sum/       # Large-scale prefix sum with UVM & GPU oversubscription
    ├── problem3_cuda_10d_nest/         # 10D loop mapping, chunking, and memory advisories
    └── problem4_cuda_convolution/      # 2D/3D Convolution with Constant Memory & Shared Tiling

```

---

## Overview of Assignments

### Assignment 1: Memory Hierarchy Modeling & Multithreaded Synchronization

* **Strided Cache Miss Analysis:** Evaluated cache conflicts on an 8-way set-associative 128 KB cache ($128\text{ B}$ line, 4 B word size, LRU replacement) across strides $1, 16, 32, 64, 2\text{K}, 8\text{K}$. Verified that small strides induce heavy conflict and capacity misses ($1,024,000$ misses across 1,000 iterations), whereas stride $8\text{K}$ isolates accesses exclusively into Set 0, fitting all lines across the 8 ways with zero evictions ($4$ total misses).


* **Matrix Multiplication Locality ($kij$ vs. $jki$):** Modeled row-major spatial locality across Direct-Mapped and Fully Associative caches. Demonstrated how column traversals in $jki$ lead to severe line thrashing ($2^{30}$ misses in direct-mapped), while fully associative caching reduces misses to $2^{26}$ for array $A$ and $2^{16}$ for arrays $B$ and $C$ by preserving row-stride blocks.


* **Concurrent FIFO Buffer Pipeline:** Developed a multi-producer, multi-consumer framework in C++17. Synchronized $T$ producers reading $L \in [L_{\min}, L_{\max}]$ lines from an input file and writing atomically to a capacity-$M$ bounded shared buffer, while $\max(1, T/2)$ consumer threads write to disk. Coordinated blocking conditions via `std::mutex` and `std::condition_variable` to eliminate busy waiting and race conditions.



---

### Assignment 2: Cache Locality, Coherence Debugging, and Custom Locks

* **3D Convolution Tiling & PAPI Hardware Counters:** Implemented naive and cache-blocked 3D convolution over $64\times64\times64$ grids. Autotuned sub-volume block sizes for private L2 cache, achieving minimum execution time at block size 32 ($55.54\text{ ms}$ vs. $61.98\text{ ms}$ naive baseline), validated with PAPI L1 data cache counters.


* **False & True Sharing Remediation (`perf c2c`):** Diagnosed multithreaded cache bouncing in a shared word/line counter using Linux `perf c2c`:


* *False Sharing:* Padded per-thread tracking structs to separate 64-byte cache lines (`thread_id * 8`), reducing runtime from $1.781\text{ s}$ to $1.213\text{ s}$.


* *True Sharing:* Replaced fine-grained lock acquisitions with thread-local counters merged once at termination, bringing execution down to $0.090\text{ s}$ and eliminating HITMs entirely ($2633 \to 0$).




* **Custom Lock Implementations:** Built Filter, Bakery, Spin, Ticket, and Array-based Queue locks using x86_64 atomic and compare-and-swap (CAS) operations. Scaled evaluations from 1 to 64 threads against `pthread_mutex`, demonstrating the low-contention efficiency of spinlocks ($233\ \mu\text{s}$) alongside the heavy coherence overhead of ticket/bakery implementations at high thread counts.



---

### Assignment 3: Loop Optimization, SIMD Intrinsics, and OpenMP Parallelization

* **3D Stencil Optimization:** Combined manual unrolling on the innermost $k$-loop with OpenMP loop collapsing (`#pragma omp parallel for collapse(2)` across $i, j$). Achieved an overall $>10\times$ speedup, reducing runtimes from $37.5\text{ ms}$ (scalar) to $3.5\text{ ms}$ (OMP + unroll).


* **AVX2-Vectorized Inclusive Prefix Sum:** Engineered an in-register scan vectorizing 8 integers per 256-bit register (`__m256i`) using intra-register shift-and-add operations followed by cross-block scalar offset propagation.


* **3D Gradient SIMD Vectorization:** Vectorized finite-difference calculations using 128-bit SSE4.1 (`__m128i`) and 256-bit AVX2 (`__m256i`) instructions with 32-byte memory alignment (`aligned_alloc`). AVX2 achieved a $1.34\times$ speedup ($99.0\text{ ms}$) over the scalar baseline ($133.0\text{ ms}$).


* **10D Loop Nest Optimization:** Refactored a deeply nested 10D loop space in two stages:


1. *Sequential Optimization:* Applied Loop-Invariant Code Motion (LICM), dropping runtime from $287.99\text{ s}$ to $148.58\text{ s}$.


2. *Parallel Optimization:* Used OpenMP loop collapsing (`collapse(8)`), dynamic scheduling, atomic updates, and `#pragma omp ordered` file writing to achieve an execution time of $10.47\text{ s}$ ($>27\times$ speedup).





---

### Assignment 4: GPU Acceleration with CUDA & Unified Virtual Memory (UVM)

* **7-Point 3D Stencil on GPU:** Benchmarked naive CUDA kernels against shared-memory tiled versions ($\text{TILE} \in \{1, 2, 4, 8\}$) and pinned memory transfers (`cudaHostAlloc`). Tiling scaled kernel execution from $1.63\text{ ms}$ down to $0.16\text{ ms}$, while pinned memory lowered end-to-end latency to $2.28\text{ ms}$.


* **Prefix Sum & GPU Memory Oversubscription:** Implemented CUDA parallel prefix scan under explicit host-device transfers and Unified Virtual Memory (`cudaMallocManaged` with `cudaMemPrefetchAsync`). For large arrays ($N = 2^{31}$ elements) that exceed physical VRAM, UVM with on-demand driver paging ran in $148.34\text{ ms}$ compared to $17,515.5\text{ ms}$ for chunked explicit copying.


* **High-Dimensional 10D Loop Space Acceleration:** Mapped the 10D iteration space to CUDA threads using linearized chunking. Evaluated Baseline CUDA, Shared-Memory Tiled, UVM (with `cudaMemAdvise`), and Thrust primitives. Optimized UVM completed in $99.2\text{ s}$, outperforming Thrust ($121.1\text{ s}$) and Naive CUDA ($125.2\text{ s}$) while maintaining 100% output verification.


* **2D & 3D Convolution Optimizations:** Implemented convolution algorithms leveraging 2D/3D shared-memory tile caching, constant memory (`__constant__`) for convolution filters, and loop unrolling. Reduced 2D pure kernel execution time from $1.44\text{ ms}$ to $0.026\text{ ms}$, and 3D kernel compute time from $0.26\text{ ms}$ to $0.21\text{ ms}$.



---

## Build & Execution Instructions

Each assignment directory contains standalone `Makefile` targets.

### Compilation Prerequisites

* **C++ Compiler:** `g++` supporting C++17 standard.


* **GPU Compiler:** `nvcc` compatible with target compute architecture (e.g., `sm_70`, `sm_80`, `sm_86`).


* **Profiling & Instrumentation Tools:** `PAPI` library (`-lpapi`), `Linux perf` (`perf c2c`), and `nvprof` / NVIDIA Nsight Compute.


* **Parallel APIs:** OpenMP (`-fopenmp`), POSIX Threads (`-pthread`).



### Common Commands

```bash
# Compile and run Assignment 1 (Multithreaded Producer-Consumer)
cd Assignment_1/problem3_producer_consumer
g++ -std=c++17 -pthread 251110053.cpp -o prob3
./prob3 input.txt <num_producers> <Lmin> <Lmax> <buffer_size> output.txt

# Compile and run Assignment 2 (perf c2c Cache Analysis)
cd Assignment_2/problem2_cache_coherence
make
perf c2c record ./prob2_padded <num_threads> input.txt
perf c2c report

# Compile and run Assignment 3 (SIMD & OpenMP Kernels)
cd Assignment_3/
make problem1   # 3D Stencil (OpenMP)
make problem3   # 3D Gradient (SSE4.1 / AVX2)
make problem4   # 10D Loop Nest (LICM + OpenMP)

# Compile and run Assignment 4 (CUDA Kernels)
cd Assignment_4/
nvcc -O3 -arch=sm_80 problem1.cu -o prob1
nvprof ./prob1

```
