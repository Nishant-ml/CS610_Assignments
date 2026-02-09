# Compilers
CC     = gcc
CXX    = g++
NVCC   = nvcc

# Compilation Flags
CFLAGS    = -O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS  = -O2 -std=c++17
CUDAFLAGS = -O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx

# Default target
all: problem1 problem2 problem3 problem4

##########################################
# Problem 1 Builds
##########################################
problem1: problem1a.out problem1b.out problem1c.out problem1d.out

problem1a.out: problem1a.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

problem1b.out: problem1b.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

problem1c.out: problem1c.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

problem1d.out: problem1d.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

##########################################
# Problem 2 Builds
##########################################
problem2: problem2.out

problem2.out: problem2.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

##########################################
# Problem 3 Builds
##########################################
problem3: problem3-v0.out problem3a.out problem3b.out problem3c.out problem3d.out
        

# Baseline CPU version
problem3-v0.out: problem3-v0.c
        $(CC) $(CFLAGS) $< -o $@

# CUDA versions
problem3a.out: problem3a.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

problem3b.out: problem3b.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

problem3c.out: problem3c.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

problem3d.out: problem3d.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@

##########################################
# Problem 4 Builds
##########################################
problem4: problem4.out

problem4.out: problem4.cu
        $(NVCC) $(CUDAFLAGS) $< -o $@


clean:
        rm -f *.out *.o compile_commands.json

~                                                                                                                                                                                                                                                                                                                                                                                           
~                        
