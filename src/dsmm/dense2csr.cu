#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define cudaCheck(call)                                                                     \
    do {                                                                                    \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaStatus != cudaSuccess) {                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "            \
                      << cudaGetErrorString(cudaStatus) << std::endl;                        \
            return EXIT_FAILURE;                                                            \
        }                                                                                   \
    } while (0)


struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    int* row_ptr;
    int* col_idx;
    float* values;
};


__global__ void countNonzerosKernel(const float* data, int* row_counts, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int count = 0;
    for (int col = 0; col < cols; ++col) {
        if (data[row * cols + col] != 0.0f) {
            count++;
        }
    }
    row_counts[row] = count;
}

__global__ void buildCSRKernel(const float* data, int* row_ptr, int* col_idx, float* values, 
                               int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int start = row_ptr[row];
    // int end = row_ptr[row + 1];
    int idx = 0;
    for (int col = 0; col < cols; ++col) {
        float val = data[row * cols + col];
        if (val != 0.0f) {
            col_idx[start + idx] = col;
            values[start + idx] = val;
            idx++;
        }
    }
}

bool denseToCSR(const float* host_data, int rows, int cols, CSRMatrix& csr) {
    int* d_row_counts;
    cudaCheck(cudaMalloc(&d_row_counts, sizeof(int) * rows));

    float* d_data;
    cudaCheck(cudaMalloc(&d_data, sizeof(float) * rows * cols));
    cudaCheck(cudaMemcpy(d_data, host_data, sizeof(float)*rows*cols, cudaMemcpyHostToDevice));
    
    dim3 block_size(1, 256);
    dim3 grid_size(1, (rows + block_size.y - 1) / block_size.y);
    countNonzerosKernel<<<grid_size, block_size>>>(d_data, d_row_counts, rows, cols);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    int* host_row_counts = new int[rows];
    cudaCheck(cudaMemcpy(host_row_counts, d_row_counts, sizeof(int)*rows, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(d_row_counts));

    csr.rows = rows;
    csr.cols = cols;
    csr.row_ptr = new int[rows + 1];
    csr.row_ptr[0] = 0;
    csr.nnz = 0;
    for (int i = 0; i < rows; ++i) {
        csr.nnz += host_row_counts[i];
        csr.row_ptr[i + 1] = csr.nnz;
    }

    int* d_row_ptr;
    cudaCheck(cudaMalloc(&d_row_ptr, sizeof(int) * (rows + 1)));
    cudaCheck(cudaMemcpy(d_row_ptr, csr.row_ptr, sizeof(int)*(rows+1), cudaMemcpyHostToDevice));

    csr.col_idx = new int[csr.nnz];
    csr.values = new float[csr.nnz];
    int* d_col_idx;
    float* d_values;
    cudaCheck(cudaMalloc(&d_col_idx, sizeof(int) * csr.nnz));
    cudaCheck(cudaMalloc(&d_values, sizeof(float) * csr.nnz));

    block_size.y = 256;
    grid_size.y = (rows + block_size.y - 1) / block_size.y;
    buildCSRKernel<<<grid_size, block_size>>>(d_data, d_row_ptr, d_col_idx, d_values, rows, cols);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(csr.col_idx, d_col_idx, sizeof(int)*csr.nnz, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(csr.values, d_values, sizeof(float)*csr.nnz, cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_data));
    cudaCheck(cudaFree(d_row_ptr));
    cudaCheck(cudaFree(d_col_idx));
    cudaCheck(cudaFree(d_values));

    delete[] host_row_counts;
    return true;
}

__global__ void prefix_sum(int *x, int *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int st = n*idx/256;
    int ed = n*(idx+1)/256;

    __shared__ int *temp;

    temp[st+1] = x[st];
    for (int i = st+1; i < ed; ++i) {
        temp[i+1] = temp[i] + x[i];
    }
    __syncthreads();

    int res;
#pragma unroll
    for (unsigned k = 1; k < 256; k *= 2) {
        if (idx & k == 0) continue;
        ed = n*(idx+1-k)/256;
        res = temp[ed];
        for (int i = st+1; i <= ed; ++i) {
            temp[i] += res;
        }
    }

    for (int i = st+1; i <= ed; ++i) {
        y[i] = temp[i];
    }
}

int main() {

    int x[268], y[269];
    int *dx, *dy;
    int choices[6] = {0, 1, 2, 4, 8, 16};
    for (int i = 0; i < 268; ++i) {
        x[i] = choices[rand() % 6];
        y[i] = 0;
    } y[268] = 0;
    cudaMalloc((void **)&dx, 268 * sizeof(int));
    cudaMalloc((void **)&dy, 269 * sizeof(int));
    cudaMemcpy(x, dx, 268 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dy, 269 * sizeof(int), cudaMemcpyDeviceToHost);

    int run_time_a = 21;
    float eps, total = 0.f;
    cudaEvent_t start, end;
    dim3 block(256), grid(1);
    cudaFuncSetAttribute(prefix_sum,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 2048);
    for (int run = 0; run < run_time_a; ++run) {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
        cudaCheck(cudaEventRecord(start, 0));
        prefix_sum<<<grid, block, 2048>>>(dx, dy, 268);
        cudaCheck(cudaEventRecord(end, 0));
        cudaCheck(cudaEventSynchronize(end));
        cudaCheck(cudaEventElapsedTime(&eps, start, end));
        total += eps;
    }
    std::cout << total / run_time_a << std::endl;

    return 0;

    const int rows = 8576;
    const int cols = 512;
    const float pz = 0.5f;
    constexpr unsigned int seed = 42;
    constexpr int run_time = 21;
    float avg_time = 0.0f;

    std::mt19937 rng(seed);
    std::bernoulli_distribution zero_dist(pz);
    std::uniform_real_distribution<float> value_dist(0.0f, 1.0f);
    float* host_data = new float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        host_data[i] = zero_dist(rng) ? 0.0f : value_dist(rng);
    }

    cudaEvent_t start_event, end_event;
    cudaCheck(cudaEventCreate(&start_event));
    cudaCheck(cudaEventCreate(&end_event));
    float* time_list = new float[run_time];

    CSRMatrix csr_warmup;
    denseToCSR(host_data, rows, cols, csr_warmup);
    delete[] csr_warmup.row_ptr;
    delete[] csr_warmup.col_idx;
    delete[] csr_warmup.values;

    for (int i = 0; i < run_time; ++i) {
        CSRMatrix csr;
        cudaCheck(cudaEventRecord(start_event, 0));
        
        denseToCSR(host_data, rows, cols, csr);
        
        cudaCheck(cudaEventRecord(end_event, 0));
        cudaCheck(cudaEventSynchronize(end_event));
        cudaCheck(cudaEventElapsedTime(&time_list[i], start_event, end_event));

        delete[] csr.row_ptr;
        delete[] csr.col_idx;
        delete[] csr.values;
    }

    float total_time = 0.0f;
    for (int i = 1; i < run_time; ++i) {
        total_time += time_list[i];
    }
    avg_time = total_time / (run_time - 1);

    std::cout << "Performance Statistics:" << std::endl;
    std::cout << "Matrix Size: " << rows << "x" << cols << ", Sparsity: " << pz << std::endl;
    std::cout << "Run Times (ms):";
    for (int i = 0; i < run_time; ++i) {
        std::cout << " " << time_list[i];
    }
    std::cout << "\nAverage Time (Excluding First Run): " << avg_time << " ms" << std::endl;

    delete[] host_data;
    delete[] time_list;
    cudaCheck(cudaEventDestroy(start_event));
    cudaCheck(cudaEventDestroy(end_event));
    return 0;
}
