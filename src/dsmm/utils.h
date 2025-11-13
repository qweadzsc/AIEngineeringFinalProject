#pragma once

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>

#include <vector>
#include <algorithm>
#include <random>

#define half0 (static_cast<cute::half_t>(0.f))


template <typename T>
void cpu_rand_data(T * c, int low=-100, int high=100, float alpha=0.01f) {
    auto t = * c;
    using ValueType = typename T::value_type;
    int n = size(t);
    for (int i = 0; i < n; ++i) {
        float v = ((rand() % (high-low)) + low) * alpha;
        t(i) = ValueType(v);
    }
}

void rand_col_offset(long long * v, int max_num, int nnz_c, int seed=42) {
    if (max_num <= 0) return;
    std::vector<long long> base(max_num);
    for (int i = 0; i < max_num; ++i) {
        base[i] = i;
    }
    // std::random_device rd;
    std::mt19937 g(seed);
    std::shuffle(base.begin(), base.end(), g);
    std::sort(base.begin(), base.begin() + nnz_c);
    for (int i = 0; i < nnz_c; ++i) {
        v[i] = base[i];
    }
}

template <typename T>
__global__ void gpu_compare_kernel(const T *x, const T *y, int n,
                                   float threshold, int *count,
                                   float *max_error) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n) {
        return;
    }

    float v0 = x[idx];
    float v1 = y[idx];

    float diff = fabs(v0 - v1);
    if (diff > threshold) {
        atomicAdd(count, 1);

        // for positive floating point, there int representation is in the same
        // order.
        int int_diff = *((int *)(&diff));
        atomicMax((int *)max_error, int_diff);
    }
}

template <typename T>
void gpu_compare(const T *x, const T *y, int n, float threshold) {
    int *num_count;
    float *max_error;
    cudaMalloc(&num_count, sizeof(int));
    cudaMalloc(&max_error, sizeof(float));
    cudaMemset(num_count, 0, sizeof(int));
    cudaMemset(max_error, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
    int num = 0;
    float error = 0;
    cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (num == 0) {
        printf("check ok, max_error = %f\n", error);
    } else {
        float p = (100.f * num) / n;
        printf("===============================\n");
        printf("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n, error);
        printf("===============================\n");
    }
}

__global__ void gpu_hmul_relu(half * lhs, const half * rhs, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float lhsn = lhs[idx];
        float rhsn = rhs[idx];
        lhs[idx] = static_cast<half>((lhsn>0&&rhsn>0)?(lhsn*rhsn):0);
    }
}

template <typename T>
__global__ void gpu_compare_kernel(const T * val, const long long * row, const long long * col,
                                   const T * y, int nnz, float threshold, int *count,
                                   float *max_error, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= nnz) {
        return;
    }

    long long rid = 0;
    while (row[rid+1] <= idx) {
        ++rid;
    }

    float v0 = val[idx];
    long long cid = col[idx];
    float v1 = y[rid+cid*n];

    float diff = fabs(v0 - v1);
    if (diff > threshold) {
        atomicAdd(count, 1);

        // for positive floating point, there int representation is in the same
        // order.
        int int_diff = *((int *)(&diff));
        atomicMax((int *)max_error, int_diff);
    }
}

template <typename T>
void gpu_compare_sp(const T * val, const long long * row, const long long * col,
                 const T * y, int nnz, int n, float threshold) {
    int *num_count;
    float *max_error;
    cudaMalloc(&num_count, sizeof(int));
    cudaMalloc(&max_error, sizeof(float));
    cudaMemset(num_count, 0, sizeof(int));
    cudaMemset(max_error, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((nnz + 255) / 256);
    gpu_compare_kernel<<<grid, block>>>(val, row, col, y, nnz, threshold, num_count, max_error, n);
    int num = 0;
    float error = 0;
    cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (num == 0) {
        printf("check ok, max_error = %f\n", error);
    } else {
        float p = (100.f * num) / nnz;
        printf("===============================\n");
        printf("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, nnz, error);
        printf("===============================\n");
    }
}

template<typename T, int k>
__global__ void gpu_add(T * x, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float a = 0.f;
    T * p = x + idx;
#pragma unroll
    for (int i = 0; i < k; ++i) {
        a += *p;
        p += n;
    }
    x[idx] = a;
}

template<typename T>
__device__ void swap(T &lhs, T &rhs) {
    T temp = lhs;
    lhs = rhs;
    rhs = temp;
}

template <typename T>
__global__ void gpu_compare_kernel(const T * val, const long long * col,
                                   const T * y, int nnz, float threshold, int *count,
                                   float *max_error, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= nnz) {
        return;
    }

    int ix, iy, iz;
    ix = idx % 4;
    iy = (idx/4) % 8;
    iz = idx / 32;

    float v0 = val[idx];
    long long cid = col[idx];
    long long rid = cid + 80 * iz;
    float v1 = y[iy+rid*n];

    float diff = fabs(v0 - v1);
    if (diff > threshold) {
        atomicAdd(count, 1);

        // for positive floating point, there int representation is in the same
        // order.
        int int_diff = *((int *)(&diff));
        atomicMax((int *)max_error, int_diff);
    }
}

template <typename T>
void gpu_compare_sp(const T * val, const long long * col,
                 const T * y, int nnz, int n, float threshold) {
    int *num_count;
    float *max_error;
    cudaMalloc(&num_count, sizeof(int));
    cudaMalloc(&max_error, sizeof(float));
    cudaMemset(num_count, 0, sizeof(int));
    cudaMemset(max_error, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((nnz + 255) / 256);
    gpu_compare_kernel<<<grid, block>>>(val, col, y, nnz, threshold, num_count, max_error, n);
    int num = 0;
    float error = 0;
    cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (num == 0) {
        printf("check ok, max_error = %f\n", error);
    } else {
        float p = (100.f * num) / nnz;
        printf("===============================\n");
        printf("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, nnz, error);
        printf("===============================\n");
    }
}
