#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cusparse.h>

#include "impl.h"


#define DEBUG 6


using Element = cutlass::half_t;
using ElementAccumulator = float;
using LayoutRow = cutlass::layout::RowMajor;
using LayoutCol = cutlass::layout::ColumnMajor;

using GemmRelu = cutlass::gemm::device::Gemm<
    Element, LayoutRow,
    Element, LayoutCol,
    Element, LayoutCol,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationRelu<
        Element,
        128 / cutlass::sizeof_bits<Element>::value,
        ElementAccumulator,
        ElementAccumulator
    >
>;

using Gemm = cutlass::gemm::device::Gemm<
    Element, LayoutCol,
    Element, LayoutCol,
    Element, LayoutCol,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        Element,
        128 / cutlass::sizeof_bits<Element>::value,
        ElementAccumulator,
        ElementAccumulator
    >
>;


__global__ void mul(const Element * A, const Element * B, Element * C, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    const Element * A_ptr = A + row * cols + col;
    const Element * B_ptr = B + row * cols + col;
    Element * C_ptr = C + row * cols + col;
    if (col < cols) {
        *C_ptr = *A_ptr * *B_ptr;
    }
}

__global__ void hmul(const half * A, const half * B, half * C, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const half * A_ptr = A + col;
    const half * B_ptr = B + col;
    half * C_ptr = C + col;
    if (col < cols) {
        *C_ptr = __hmul(__hmax_nan(*A_ptr, __float2half(0.0f)), __hmax_nan(*B_ptr, __float2half(0.0f)));
        // *C_ptr = (*A_ptr>__float2half(0.0f) && *B_ptr>__float2half(0.0f))?__hmul(*A_ptr, *B_ptr):__float2half(0.0f);
    }
}

__global__ void add(const half * A, const half * B, half * C, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    const half * A_ptr = A + row * cols + col;
    const half * B_ptr = B + row * cols + col;
    half * C_ptr = C + row * cols + col;
    if (col < cols) {
        *C_ptr = __hadd(*A_ptr, *B_ptr);
    }
}


void cutlass_mlp(const half * input, const half * up, const half * gate, const half * down,
    long long * mask_c, long long * mask_r, half * mask_v1, half * mask_v2, half * ir1, half * ir2, half * result,
    int bs, int in_d, int mid_d, int t_d, int nnz, void * buf) {
    /*
     * input:     bs x in_d, row-major, or in_d x bs,    col-major
     * up:     mid_d x in_d, row-major, or in_d x mid_d, col-major
     * gate:   mid_d x in_d, row-major, or in_d x mid_d, col-major
     * down:   mid_d x in_d, row-major, or in_d x mid_d, col-major
     * ir1:       bs x t_d,  row-major, or  t_d x bs,    col-major
     * ir2:       bs x t_d,  row-major, or  t_d x bs,    col-major
     * result:   2bs x in_d, row-major, or in_d x 2bs,   col-major
     * 
     * 1. ir11 = relu(up1^T @ input)   // ir12 = (up2^T @ input) · mask2
     * 2. ir21 = relu(gate1^T @ input) // ir22 = (gate2^T @ input) · mask2
     * 3. ir1 = ir1 * ir2
     * 4. result1 = down1^T @ ir11     // result2 = down2^T @ ir12
     * 5. result1 = result1 + result2
     */

    // 1. setup
    // cudaStream_t stream1, stream2;
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t & stream2 = stream1;
    // cudaStreamCreate(&stream2);

    cudaEvent_t st, ed, e11, e12, e13, e14, e21, e22, e23, e24;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventCreate(&e11);
    cudaEventCreate(&e12);
    cudaEventCreate(&e13);
    cudaEventCreate(&e14);
    cudaEventCreate(&e21);
    cudaEventCreate(&e22);
    cudaEventCreate(&e23);
    cudaEventCreate(&e24);

    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    cusparseSetStream(cusparse_handle, stream2);

    cudaEventRecord(st, 0);
    
    const Element * d_i = reinterpret_cast<const Element *>(input);
    const Element * d_u = reinterpret_cast<const Element *>(up);
    const Element * d_g = reinterpret_cast<const Element *>(gate);
    const Element * d_d = reinterpret_cast<const Element *>(down);
    Element * d_1 = reinterpret_cast<Element *>(ir1);
    Element * d_2 = reinterpret_cast<Element *>(ir2);
    Element * d_r = reinterpret_cast<Element *>(result);
    cutlass::Status status_c;
    cusparseStatus_t status_s;

    if (DEBUG == 1) return;

    // 2. calculate ir11 = relu(up1^T @ input) and ir21 = relu(gate1^T @ input)
    //    and ir1 = ir1 * ir2 in stream1
    cudaEventRecord(e11, stream1);
    GemmRelu gemm12;
    GemmRelu::Arguments args1({t_d, bs, in_d}, {d_u, in_d}, {d_i, in_d}, {d_1, t_d}, {d_1, t_d}, {1.0f, 0.0f});
    status_c = gemm12(args1, nullptr, stream1);
    GemmRelu::Arguments args2({t_d, bs, in_d}, {d_g, in_d}, {d_i, in_d}, {d_2, t_d}, {d_2, t_d}, {1.0f, 0.0f});
    status_c = gemm12(args2, nullptr, stream1);
    cudaEventRecord(e12, stream1);
    dim3 grid_m((t_d+255)/256, bs);
    dim3 block_m(256);
    mul<<<grid_m, block_m, 0, stream1>>>(d_1, d_2, d_1, t_d);
    cudaEventRecord(e13, stream1);

    if (DEBUG == 2) return;

    // 3. calculate ir12 = (up2^T @ input) · mask2 and ir22 = (gate2^T @ input) · mask2
    //    and mask_v1 = mask_v1 * mask_v2 in stream2
    cudaEventRecord(e21, stream2);
    int remain = mid_d - t_d;
    cusparseSpMatDescr_t mat1, mat2;
    cusparseDnMatDescr_t matInput, matUp, matGate;
    cusparseCreateCsr(&mat1, bs, remain, nnz, mask_r, mask_c, mask_v1, CUSPARSE_INDEX_64I,
                      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateCsr(&mat2, bs, remain, nnz, mask_r, mask_c, mask_v2, CUSPARSE_INDEX_64I,
                      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnMat(&matInput, bs,   in_d,   in_d, (void*)input,           CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matUp,    in_d, remain, in_d, (void*)(up+in_d*t_d),   CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matGate,  in_d, remain, in_d, (void*)(gate+in_d*t_d), CUDA_R_16F, CUSPARSE_ORDER_COL);
    float alpha = 1.0f, beta = 0.0f;
    status_s = cusparseSDDMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matInput, matUp,   &beta, mat1, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buf);
    status_s = cusparseSDDMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matInput, matGate, &beta, mat2, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buf);
    cusparseDestroySpMat(mat1);
    cusparseDestroySpMat(mat2);
    cusparseDestroyDnMat(matInput);
    cusparseDestroyDnMat(matUp);
    cusparseDestroyDnMat(matGate);
    cudaEventRecord(e22, stream2);
    dim3 grid_h((nnz+255)/256);
    dim3 block_h(256);
    hmul<<<grid_h, block_h, 0, stream2>>>(mask_v1, mask_v2, mask_v1, nnz);
    cudaEventRecord(e23, stream2);

    if (DEBUG == 3) return;

    // 4. calculate result1 = down1^T @ ir11 in stream1
    Gemm::Arguments args3({in_d, bs, t_d}, {d_d, in_d}, {d_1, t_d}, {d_r, in_d}, {d_r, in_d}, {1.0f, 0.0f});
    Gemm gemm3;
    status_c = gemm3(args3, nullptr, stream1);
    cudaEventRecord(e14, stream1);

    if (DEBUG == 4) return;

    // 5. calculate result2 = down2^T @ ir12 in stream2
    cusparseSpMatDescr_t mats;
    cusparseDnMatDescr_t matd, matr;
    cusparseCreateCsr(&mats, bs, remain, nnz, mask_r, mask_c, mask_v1, CUSPARSE_INDEX_64I,
                      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnMat(&matd, remain, in_d, in_d, (void*)(down+in_d*t_d),  CUDA_R_16F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matr, bs,     in_d, in_d, (void*)(result+in_d*bs), CUDA_R_16F, CUSPARSE_ORDER_ROW);
    status_s = cusparseSpMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mats, matd, &beta, matr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, buf);
    cusparseDestroySpMat(mats);
    cusparseDestroyDnMat(matd);
    cusparseDestroyDnMat(matr);
    cudaEventRecord(e24, stream2);

    if (DEBUG == 5) return;

    // 6. calculate result1 = result1 + result2
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    dim3 grid_a((in_d+255)/256, bs);
    dim3 block_a(256);
    add<<<grid_a, block_a>>>(result, result + bs * in_d, result, in_d);
    // add<<<grid_a, block_a>>>(result + bs * in_d, result, result + bs * in_d, in_d);
    cudaEventRecord(ed, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(ed);

    float tt, t11, t12, t13, t21, t22, t23;
    cudaEventElapsedTime(&tt, st, ed);
    cudaEventElapsedTime(&t11, e11, e12);
    cudaEventElapsedTime(&t12, e12, e13);
    cudaEventElapsedTime(&t13, e13, e14);
    cudaEventElapsedTime(&t21, e21, e22);
    cudaEventElapsedTime(&t22, e22, e23);
    cudaEventElapsedTime(&t23, e23, e24);
    std::cout << tt << " | " << t11 << " | " << t12 << " | " << t13 << " | " << t21 << " | " << t22 << " | " << t23 << std::endl;
    
    cudaEventDestroy(st);
    cudaEventDestroy(ed);
    cudaEventDestroy(e11);
    cudaEventDestroy(e12);
    cudaEventDestroy(e13);
    cudaEventDestroy(e14);
    cudaEventDestroy(e21);
    cudaEventDestroy(e22);
    cudaEventDestroy(e23);
    cudaEventDestroy(e24);
}
