#include <iostream>
#include <cublas_v2.h>

#include "config_dense.h"
#include "utils.h"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err \
                      << " (" << cudaGetErrorString(err) << ") \"" << #call << "\" \n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


template <typename Config>
__global__ void
sg(const void * x, const void * up, void * r, int m, int n, int k) {
    using namespace cute;
    using X = Underscore;
    using T = typename Config::T;

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int idx = threadIdx.x;
    int debug = (ix + iy) * 128 + idx;

    // init
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    using SmemLayoutX = typename Config::SmemLayoutX;
    using SmemLayoutUp = typename Config::SmemLayoutUp;
    using SmemLayoutResult = typename Config::SmemLayoutResult;
    using G2SCopyX = typename Config::G2SCopyX;
    using G2SCopyUp = typename Config::G2SCopyUp;
    using S2RCopyAtomX = typename Config::S2RCopyAtomX;
    using S2RCopyAtomUp = typename Config::S2RCopyAtomUp;
    using TiledMMA = typename Config::MMA;
    using R2SCopyAtomR = typename Config::R2SCopyAtomR;
    using S2GCopyR = typename Config::S2GCopyR;

    extern __shared__ T shm_data[];
    T * shmx = shm_data;
    T * shmup = shmx + cute::cosize(SmemLayoutX{});

    Tensor I = make_tensor(make_gmem_ptr((T *)x), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor U = make_tensor(make_gmem_ptr((T *)up), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor R = make_tensor(make_gmem_ptr((T *)r), make_shape(m, n), make_stride(n, Int<1>{}));

    Tensor gX = local_tile(I, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gU = local_tile(U, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gR = local_tile(R, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    auto sX = make_tensor(make_smem_ptr(shmx), SmemLayoutX{});
    auto sU = make_tensor(make_smem_ptr(shmup), SmemLayoutUp{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrU = thr_mma.partition_fragment_A(gU(_, _, 0));
    auto tCrX = thr_mma.partition_fragment_B(gX(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gR);
    clear(tCrD);

    // g2s setting
    G2SCopyX g2s_tiled_copy_x;
    auto g2s_thr_copy_x = g2s_tiled_copy_x.get_slice(idx);
    auto tXgX_copy = g2s_thr_copy_x.partition_S(gX);
    auto tXsX_copy = g2s_thr_copy_x.partition_D(sX);
    G2SCopyUp g2s_tiled_copy_up;
    auto g2s_thr_copy_up = g2s_tiled_copy_up.get_slice(idx);
    auto tUgU_copy = g2s_thr_copy_up.partition_S(gU);
    auto tUsU_copy = g2s_thr_copy_up.partition_D(sU);

    // s2r setting
    auto s2r_tiled_copy_up = make_tiled_copy_A(S2RCopyAtomUp{}, tiled_mma);
    auto s2r_thr_copy_up = s2r_tiled_copy_up.get_slice(idx);
    auto tUsU = s2r_thr_copy_up.partition_S(sU);
    auto tCrU_view = s2r_thr_copy_up.retile_D(tCrU);
    auto s2r_tiled_copy_x = make_tiled_copy_B(S2RCopyAtomX{}, tiled_mma);
    auto s2r_thr_copy_x = s2r_tiled_copy_x.get_slice(idx);
    auto tXsX = s2r_thr_copy_x.partition_S(sX);
    auto tCrX_view = s2r_thr_copy_x.retile_D(tCrX);

    // stage
    int itile_to_read = kStage - 1;  // first tile read that has not been submitted 
    int ismem_read = 0;              // smem that calculation is working on
    int ismem_write = kStage - 1;    // smem write that has not been submitted

    // submit kStage - 1 g2s read
#pragma unroll
    for (int istage = 0; istage < kStage-1; ++istage) {
        cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, istage), tXsX_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_up, tUgU_copy(_, _, _, istage), tUsU_copy(_, _, _, istage));
        cp_async_fence();
    }

    // first s2r read (x & up)
    cp_async_wait<kStage - 2>();
    __syncthreads();
    cute::copy(s2r_tiled_copy_x, tXsX(_, _, 0, ismem_read), tCrX_view(_, _, 0));
    cute::copy(s2r_tiled_copy_up, tUsU(_, _, 0, ismem_read), tCrU_view(_, _, 0));

    // loop over k
    int ntile = k / kTileK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        // loop in kTileK
        int nk = size<2>(tCrU);
#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_n = (ik + 1) % nk;
            if (ik == nk - 1) {
                // wait next g2s read done
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }
            // copy next up & x
            cute::copy(s2r_tiled_copy_x, tXsX(_, _, ik_n, ismem_read), tCrX_view(_, _, ik_n));
            cute::copy(s2r_tiled_copy_up, tUsU(_, _, ik_n, ismem_read), tCrU_view(_, _, ik_n));
            if (ik == 0) {
                // submit next g2s read
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, itile_to_read),
                               tXsX_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_up, tUgU_copy(_, _, _, itile_to_read),
                               tUsU_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            cute::gemm(tiled_mma, tCrD, tCrU(_, _, ik), tCrX(_, _, ik), tCrD);
        }
    }

    // epi
    auto sD = make_tensor(make_smem_ptr(shm_data), SmemLayoutResult{});

    auto r2s_tiled_copy_r = make_tiled_copy_C(R2SCopyAtomR{}, tiled_mma);
    auto r2s_thr_copy_r = r2s_tiled_copy_r.get_slice(idx);
    auto tCrD_r2s = r2s_thr_copy_r.retile_S(tCrD);
    auto tCsD_r2s = r2s_thr_copy_r.partition_D(sD);
    auto tCrD_r2sx = group_modes<1, 3>(tCrD_r2s);

    S2GCopyR s2g_tiled_copy_r;
    auto s2g_thr_copy_r = s2g_tiled_copy_r.get_thread_slice(idx);
    auto tRsR_s2g = s2g_thr_copy_r.partition_S(sD);
    auto tRgR_s2g = s2g_thr_copy_r.partition_D(gR);
    auto tRgR_s2gx = group_modes<1, 3>(tRgR_s2g);

    int step = size<1>(tCrD_r2sx);
    int cpbs = size<3>(tCsD_r2s);

#pragma unroll
    for (int i = 0; i < step; i+=cpbs) {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < cpbs; ++j) {
            cute::copy(r2s_tiled_copy_r, tCrD_r2sx(_, i+j), tCsD_r2s(_, 0, 0, j));
            // auto t = make_tensor_like<T>(tCrD_r2sx(_, i+j));
            // cute::copy(tCrD_r2sx(_, i+j), t);
            // cute::copy(r2s_tiled_copy_c, t, tCsD_r2s(_, 0, 0, j));
        }
        __syncthreads();
    
        // shm -> global
#pragma unroll
        for (int j = 0; j < cpbs; ++j) {
            cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, 0, 0, j), tRgR_s2gx(_, i+j));
        }
        __syncthreads();
    }

}


int main() {
    using T = cute::half_t;
    using namespace cute;
    using X = Underscore;

    srand(1);

    const int m = 32768;
    const int n = 512;
    const int k = 4096;

    cublasHandle_t handle;
    cublasCreate(&handle);

    T * h_x    = new T[n * k];
    T * h_up   = new T[m * k];
    T * h_r1   = new T[m * n];
    T * h_r2   = new T[m * n];

    T * d_x, * d_up, * d_r1, * d_r2;

    CUDA_CHECK(cudaMalloc(&d_x,  sizeof(T) * n * k));
    CUDA_CHECK(cudaMalloc(&d_up, sizeof(T) * m * k));
    CUDA_CHECK(cudaMalloc(&d_r1, sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(&d_r2, sizeof(T) * m * n));

    auto tx  = make_tensor(h_x,  make_shape(n, k), make_stride(k, 1));
    auto tu  = make_tensor(h_up, make_shape(m, k), make_stride(k, 1));
    auto tr1 = make_tensor(h_r1, make_shape(m, n), make_stride(n, 1));
    auto tr2 = make_tensor(h_r2, make_shape(m, n), make_stride(n, 1));

    cpu_rand_data(&tx);
    cpu_rand_data(&tu);
    clear(tr1);
    clear(tr2);

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(T)*n*k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up, sizeof(T)*m*k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r1, 0, sizeof(T)*m*n));
    CUDA_CHECK(cudaMemset(d_r2, 0, sizeof(T)*m*n));

    // calculation

    constexpr int kTileM = 256;
    constexpr int kTileN = 256;
    constexpr int kTileK = 32;
    config::Config<T, kTileM, kTileN, kTileK> cfg;
    dim3 block(128);
    dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
    int shm_size = cfg.kShmSize;

    int run_num = 21;
    cublasStatus_t ret;
    cudaEvent_t cublas_start, cublas_end;
    cudaEvent_t ours_start, ours_end;
    CUDA_CHECK(cudaEventCreate(&cublas_start));
    CUDA_CHECK(cudaEventCreate(&cublas_end));
    CUDA_CHECK(cudaEventCreate(&ours_start));
    CUDA_CHECK(cudaEventCreate(&ours_end));
    float cublas_avg_time = 0.0f;
    float ours_avg_time = 0.0f;
    for (int i = 0; i < run_num; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());

        // cuBLAS
        CUDA_CHECK(cudaEventRecord(cublas_start));
        half alpha = 1.f;
        half beta = 0.f;
        CUDA_CHECK(cudaMemset(d_r2, 0, sizeof(T)*m*n));
        ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                          &alpha, (half *)d_x, k, (half *)d_up, k,
                          &beta, (half *)d_r2, n);
        if (ret != CUBLAS_STATUS_SUCCESS) {
            printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
        }
        CUDA_CHECK(cudaEventRecord(cublas_end));
        CUDA_CHECK(cudaEventSynchronize(cublas_end));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, cublas_start, cublas_end));
        if (i != 0) cublas_avg_time += elapsed_time;

        // ours
        CUDA_CHECK(cudaEventRecord(ours_start));
        CUDA_CHECK(cudaMemset(d_r1, 0, sizeof(T)*m*n));
        cudaFuncSetAttribute(sg<decltype(cfg)>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        sg<decltype(cfg)><<<grid, block, shm_size>>>(d_x, d_up, d_r1, m, n, k);
        CUDA_CHECK(cudaEventRecord(ours_end));
        CUDA_CHECK(cudaEventSynchronize(ours_end));

        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, ours_start, ours_end));
        if (i != 0) ours_avg_time += elapsed_time;
    }

    cublas_avg_time /= run_num-1;
    ours_avg_time /= run_num-1;
    printf("Average cuBLAS time: %.4f ms\n", cublas_avg_time);
    printf("Average ours time:   %.4f ms\n", ours_avg_time);
    CUDA_CHECK(cudaEventDestroy(cublas_start));
    CUDA_CHECK(cudaEventDestroy(cublas_end));
    CUDA_CHECK(cudaEventDestroy(ours_start));
    CUDA_CHECK(cudaEventDestroy(ours_end));

    auto err = cudaGetLastError();
    printf("block = (%d,), gird = (%d, %d), shm = %d\n", block.x, grid.x, grid.y, shm_size);
    if (err == cudaSuccess) {
      printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    } else {
      printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    }

    // end

    CUDA_CHECK(cudaMemcpy(h_r1, d_r1, sizeof(T)*m*n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_r2, d_r2, sizeof(T)*m*n, cudaMemcpyDeviceToHost));

    gpu_compare(d_r1, d_r2, m * n, 0.1f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_r1));
    CUDA_CHECK(cudaFree(d_r2));

    delete[] h_x;
    delete[] h_up;
    delete[] h_r1;
    delete[] h_r2;
    
    cublasDestroy(handle);
}
