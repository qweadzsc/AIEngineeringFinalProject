#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>  // 包含half数据类型头文件

#define M 8
#define N 14336
#define K 4096
#define s 0.98
#define run_time 21


int main() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    // 初始化矩阵（主机端，使用half类型）
    half *h_X = new half[M * K];
    half *h_U = new half[N * K];
    half *h_G = new half[N * K];
    half *h_D = new half[K * N];

    // 初始化数据（float转换为half）
    for (long long i = 0; i < M * K; ++i) 
        h_X[i] = __float2half((float)rand() / RAND_MAX);
    for (long long i = 0; i < N * K; ++i) 
        h_U[i] = __float2half((float)rand() / RAND_MAX);
    for (long long i = 0; i < N * K; ++i) 
        h_G[i] = __float2half((float)rand() / RAND_MAX);
    for (long long i = 0; i < K * N; ++i) 
        h_D[i] = __float2half((float)rand() / RAND_MAX);

    // 生成稀疏矩阵 Mask（CSR 格式，索引类型为long long）
    long long nnz = static_cast<long long>(M * N * (1 - s));
    long long *h_csrRowPtr = new long long[M + 1];
    long long *h_csrColInd = new long long[nnz];
    half *h_csrVal = new half[nnz];  // val保持half类型，初始化为1

    // 生成rowptr（确保单调递增，索引类型为long long）
    h_csrRowPtr[0] = 0;
    for (long long i = 1; i < M + 1; ++i) {
        long long remaining = nnz - h_csrRowPtr[i-1];
        long long max_add = remaining / (M - i + 1) + 1;
        h_csrRowPtr[i] = h_csrRowPtr[i-1] + rand() % max_add;
    }
    h_csrRowPtr[M] = nnz;  // 确保最后一行指针正确
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "act: " << (float)nnz / M / N << std::endl;

    // 生成colind（列索引在[0, N)范围内）
    for (long long i = 0; i < nnz; ++i) 
        h_csrColInd[i] = rand() % N;

    // 初始化csrVal为1.0（half类型）
    for (long long i = 0; i < nnz; ++i) 
        h_csrVal[i] = __float2half(1.0f);

    // 分配设备内存（half和long long类型）
    half *d_X, *d_U, *d_G, *d_D, *d_I1, *d_I2, *d_I1_Dense, *d_I2_Dense, *d_R, *buf;
    long long *d_csrRowPtr, *d_csrColInd;
    
    cudaMalloc((void**)&d_X, M * K * sizeof(half));
    cudaMalloc((void**)&d_U, N * K * sizeof(half));
    cudaMalloc((void**)&d_G, N * K * sizeof(half));
    cudaMalloc((void**)&d_D, K * N * sizeof(half));
    cudaMalloc((void**)&d_I1, nnz * sizeof(half));
    cudaMalloc((void**)&d_I2, nnz * sizeof(half));
    cudaMalloc((void**)&d_I1_Dense, M * N * sizeof(half));
    cudaMalloc((void**)&d_I2_Dense, M * N * sizeof(half));
    cudaMalloc((void**)&d_R, M * K * sizeof(half));
    cudaMalloc((void**)&d_csrRowPtr, (M + 1) * sizeof(long long));
    cudaMalloc((void**)&d_csrColInd, nnz * sizeof(long long));
    cudaMalloc((void**)&buf, 172188 * sizeof(half));

    // 主机到设备内存复制
    cudaMemcpy(d_X, h_X, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, N * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, h_G, N * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (M + 1) * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I1, h_csrVal, nnz * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I2, h_csrVal, nnz * sizeof(half), cudaMemcpyHostToDevice);

    // 创建cuSPARSE矩阵描述符（指定索引基和数据类型）
    cusparseConstDnMatDescr_t X_d, U_d, G_d, D_d;
    cusparseDnMatDescr_t R_d;
    cusparseSpMatDescr_t I1_d, I2_d;
    cusparseCreateConstDnMat(&X_d, M, K, K, d_X, CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseCreateConstDnMat(&U_d, K, N, K, d_U, CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseCreateConstDnMat(&G_d, K, N, K, d_G, CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseCreateConstDnMat(&D_d, N, K, K, d_D, CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&R_d, M, K, K, d_R, CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseCreateCsr(&I1_d, M, N, nnz, d_csrRowPtr, d_csrColInd, d_I1,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateCsr(&I2_d, M, N, nnz, d_csrRowPtr, d_csrColInd, d_I2,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0.0f;
    float alpha = 1.0f, beta = 0.0f;
    half alpha_half(1.0f), beta_half(0.0f);

    cusparseStatus_t sp_status;

    for (long long i = 0; i < run_time; ++i) {
        if (i != 0) {
            // cudaMemcpy(d_I1, h_csrVal, nnz * sizeof(half), cudaMemcpyHostToDevice);
            // cudaMemcpy(d_I2, h_csrVal, nnz * sizeof(half), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
        }

        // 计算SDDMM
        size_t temp;
        cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, X_d, U_d, &beta, I1_d, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buf);
        cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, X_d, G_d, &beta, I2_d, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buf);

        // 稀疏矩阵向量乘法
        sp_status =
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, I1_d, D_d, &beta, R_d, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, buf);

        if (i != 0) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            totalTime += elapsedTime;
        }
        
        if (sp_status != CUSPARSE_STATUS_SUCCESS) {
            std::cout << "runtime error" << std::endl;
        }
    }

    // 输出平均时间（忽略第一次运行，从第二次开始计时）
    std::cout << "Average time (excluding first run): " 
              << totalTime / (run_time - 1) << " ms" << std::endl;

    totalTime = 0.0f;
    for (long long i = 0; i < run_time; ++i) {
        if (i != 0) {
            cudaDeviceSynchronize();
            cudaEventRecord(start);
        }

        // 计算U、G
        cublasHgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_half,
            d_U, N, d_X, K, &beta_half, d_I1_Dense, N);
        cublasHgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_half,
            d_G, N, d_X, K, &beta_half, d_I2_Dense, N);
        
        // D
        cublasHgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha_half,
            d_D, K, d_I1_Dense, N, &beta_half, d_R, K);


        if (i != 0) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, start, stop);
            totalTime += elapsedTime;
        }
    }

    // 输出平均时间（忽略第一次运行，从第二次开始计时）
    std::cout << "Average time (excluding first run): " 
              << totalTime / (run_time - 1) << " ms" << std::endl;

    // 资源释放
    cudaFree(d_X); cudaFree(d_U); cudaFree(d_G); cudaFree(d_D);
    cudaFree(d_I1); cudaFree(d_I2); cudaFree(d_I1_Dense); cudaFree(d_I2_Dense); cudaFree(d_R);
    cudaFree(d_csrRowPtr); cudaFree(d_csrColInd); cudaFree(buf);
    
    cusparseDestroyDnMat(X_d);
    cusparseDestroyDnMat(U_d);
    cusparseDestroyDnMat(G_d);
    cusparseDestroyDnMat(D_d);
    cusparseDestroyDnMat(R_d);
    cusparseDestroySpMat(I1_d);
    cusparseDestroySpMat(I2_d);
    cusparseDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    delete[] h_X; delete[] h_U; delete[] h_G; delete[] h_D;
    delete[] h_csrRowPtr; delete[] h_csrColInd; delete[] h_csrVal;

    return 0;
}
