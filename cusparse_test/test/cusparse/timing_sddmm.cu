#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <iomanip>
#include <random>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>


__global__ void fillMatrixWithRandom(float* matrix, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        curandState_t state;
        curand_init(clock64(), idx, 0, &state);
        matrix[idx] = static_cast<float>(curand_uniform(&state));
    }
}


double test_dense(cublasHandle_t & handle, int m, int n, int k, int n_run = 100, int n_warmup = 10) {
    float * A_d, * B_d, * result_d;
    cudaMalloc(&A_d, m*k*sizeof(float));
    cudaMalloc(&B_d, k*n*sizeof(float));
    cudaMalloc(&result_d, m*n*sizeof(float));

    dim3 block_size(256);
    dim3 grid_size((m*k+block_size.x-1)/block_size.x);
    fillMatrixWithRandom<<<grid_size, block_size>>>(A_d, m, k);
    grid_size.x = (k * n + block_size.x - 1) / block_size.x;
    fillMatrixWithRandom<<<grid_size, block_size>>>(B_d, k, n);
    cudaDeviceSynchronize();

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status;
    for (int i = 0; i < n_warmup; ++i) {
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_d, m, B_d, k, &beta, result_d, m);
        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << status << std::endl;
            exit(status);
        }
    }

    double total_time = 0;
    float eps;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for (int i = 0; i < n_run; ++i) {
        cudaEventRecord(start, 0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_d, m, B_d, k, &beta, result_d, m);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&eps, start, end);
        total_time += eps;
    }
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(result_d);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    std::cout << "dense " << n << ": " << std::fixed << std::setprecision(4) << total_time / n_run * 1000 << "ms" << std::endl;

    return total_time / n_run * 1000;
}

void gen_rand_perm(int * arr, int size) {
    std::default_random_engine engine{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, size - 1);
    for (int i = 0; i < size; ++i) {
        int r = i + dist(engine) % (size - i);
        if (r < size) {
            std::swap(arr[i], arr[r]);
        }
    }
}


double test_sddmm(cusparseHandle_t & handle, int m, int n, int k, double s, int n_run = 100, int n_warmup = 10) {
    int nz = static_cast<int>(s*m*n);

    float * A_d, * B_d, *result_d;
    cudaMalloc(&A_d, m*k*sizeof(float));
    cudaMalloc(&B_d, k*n*sizeof(float));
    cudaMalloc(&result_d, m*n*sizeof(float));

    int * crow_indices_d, * col_indices_d;
    float * values_d;
    cudaMalloc(&crow_indices_d, (m+1)*sizeof(int));
    cudaMalloc(&col_indices_d, nz*sizeof(int));
    cudaMalloc(&values_d, nz*sizeof(float));

    int * r_d, * c_d;
    float * v_d;
    cudaMalloc(&r_d, (m+1)*sizeof(int));
    cudaMalloc(&c_d, nz*sizeof(int));
    cudaMalloc(&v_d, nz*sizeof(float));

    int * crow_indices_h = new int[m + 1];
    int * col_indices_h = new int[nz];
    float * values_h = new float[nz];
    
    for (int i = 0; i < m + 1; ++i) {
        crow_indices_h[i] = static_cast<int>(i * nz / m);
    }
    for (int i = 0; i < m; ++i) {
        int start = crow_indices_h[i];
        int end = crow_indices_h[i + 1];
        int num_elems = end - start;
        gen_rand_perm(col_indices_h + start, num_elems);
    }
    for (int i = 0; i < nz; ++i) {
        values_h[i] = 1.0F;
    }
    cudaMemcpy(crow_indices_d, crow_indices_h, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_d, col_indices_h, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values_h, nz*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size((m*k+block_size.x-1)/block_size.x);
    fillMatrixWithRandom<<<grid_size, block_size>>>(A_d, m, k);
    grid_size.x = (k * n + block_size.x - 1) / block_size.x;
    fillMatrixWithRandom<<<grid_size, block_size>>>(B_d, k, n);
    cudaDeviceSynchronize();

    cusparseDnMatDescr_t A_cs, B_cs;
    cusparseCreateDnMat(&A_cs, m, k, k, A_d, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&B_cs, k, n, k, B_d, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseSpMatDescr_t S_cs;
    cusparseCreateCsr(&S_cs, m, n, nz, r_d, c_d, v_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    size_t buffer_size;
    cudaError_t status_cuda;
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseSDDMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, A_cs, B_cs, &beta, S_cs, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size);
    float * buffer_d;
    status_cuda = cudaMalloc(&buffer_d, buffer_size);
    if (status_cuda != CUDA_SUCCESS) {
        std::cout << status_cuda << std::endl;
        exit(status_cuda);
    }

    cusparseStatus_t status;
    for (int i = 0; i < n_warmup; ++i) {
        cudaMemcpy(r_d, crow_indices_d, (m+1)*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(c_d, col_indices_d, nz*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_d, values_d, nz*sizeof(float), cudaMemcpyDeviceToDevice);
        cusparseCreateCsr(&S_cs, m, n, nz, r_d, c_d, v_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        status = cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, A_cs, B_cs, &beta, S_cs, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buffer_d);
        cudaDeviceSynchronize();
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << status << std::endl;
            exit(status);
        }
    }

    double total_time = 0;
    float eps;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for (int i = 0; i < n_run; ++i) {
        cudaMemcpy(r_d, crow_indices_d, (m+1)*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(c_d, col_indices_d, nz*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_d, values_d, nz*sizeof(float), cudaMemcpyDeviceToDevice);
        cusparseCreateCsr(&S_cs, m, n, nz, r_d, c_d, v_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cudaEventRecord(start, 0);
        cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha, A_cs, B_cs, &beta, S_cs, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buffer_d);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&eps, start, end);
        total_time += eps;
    }
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(result_d);
    cudaFree(crow_indices_d);
    cudaFree(col_indices_d);
    cudaFree(values_d);
    cudaFree(buffer_d);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    std::cout << std::fixed << std::setprecision(4) << "sddmm " << s << "-" << n << ": " << total_time / n_run * 1000
        << "ms | buffer size: " << buffer_size << " | nnz: " << nz << std::endl;

    return total_time / n_run * 1000;
}


void write_vector_to_stream(const std::vector<double> & vec, std::ofstream & out_file, int precision = 6) {
    for (auto i: vec) {
        out_file << std::fixed << std::setprecision(precision) << i << " ";
    }
    out_file << std::endl;
}


int main() {
    cublasHandle_t handle_cublas;
    cusparseHandle_t handle_cusparse;
    cublasCreate(&handle_cublas);
    cusparseCreate(&handle_cusparse);
    cudaSetDevice(0);
    std::string file_name("/root/projects/MMLU/data/cusparse_time_result.txt");
    std::ofstream out_file(file_name);

    int m = 8;
    int n = 14336;
    int k = 4096;
    std::vector<double> ss{0.15, 0.10, 0.06};

    std::vector<double> time_dense(31), time_sddmm(31);
    for (auto s: ss) {
        for (int i = 0; i < 31; ++i) {
            int n1 = static_cast<int>(n*(i+1)/32);
            int n2 = n - n1;
            double sss = s * (1.0-9.0*i/300.0);
            double total_time = 0;
            for (int j = 0; j < 10; ++j) {
                double time_u = test_dense(handle_cublas, m, n1, k);
                total_time += time_u;
            }
            time_dense[i] = total_time / 10;
            total_time = 0;
            for (int j = 0; j < 10; ++j) {
                double time_u = test_sddmm(handle_cusparse, m, n2, k, sss);
                total_time += time_u;
            }
            time_sddmm[i] = total_time / 10;
        }
        write_vector_to_stream(time_dense, out_file);
        write_vector_to_stream(time_sddmm, out_file);
    }

    out_file.close();
    cublasDestroy(handle_cublas);
    cusparseDestroy(handle_cusparse);
}
