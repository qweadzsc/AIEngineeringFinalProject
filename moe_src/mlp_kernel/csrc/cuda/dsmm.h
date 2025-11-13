#pragma once
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define PRINT(name, content) \
    do { \
        print(name); \
        print(" : "); \
        print(content); \
        print("\n"); \
    } while(0)


void sddmm_api(const half * x, const half * up, const half * gate, half * r,
               const long long * row, const long long * col, half * val,
               int m, int n, int k, int e, int p, int mn);

void spmm_api(const half * x, const half * down, half * r,
              const long long * row, const long long * col, const half * val, const half * w,
              int m, int n, int k, int e, int p, int mn);
