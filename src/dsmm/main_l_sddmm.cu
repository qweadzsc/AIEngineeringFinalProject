#include <iostream>
#include <cublas_v2.h>

#include "config_l_sddmm.h"
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
sg(const void * x, const void * up, const void * gate, const void * mask, const void * cal,
   void * r, void * val, void * col, int m, int n, int p, int k) {
    /*
     * x - m*k
     * up - n*k
     * gate - n*k
     * mask - m*(n-p)
     * cal - (n-p)/32, 表示这一块每列要算的nz
     * r - m*p
     * val - nnz
     * col - nnz
     */

    using namespace cute;
    using X = Underscore;
    using T = typename Config::T;

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int idx = threadIdx.x;
    int il = idx % 32;
    int nl = idx / 32;
    int ig = il % 8;
    int ng = il / 8;
    int debug = (ix * gridDim.y + iy) * 128 + idx;

    // init
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kTileS = Config::kTileS;
    constexpr int kStage = Config::kStage;

    extern __shared__ T shm_data[];

    // sp init
    constexpr int maxnnz_pt = Config::maxnnz_pt;  // max nnz per thread, 4 assumed
    constexpr int cal_time = Config::cal_time;  // kTileK / 16, 2 assumed
    constexpr int elmt_per_thread = kTileM/4;  // element of mask per thread
    int num_sp_block = (n-p)/32;
    bool pure_dense = (ix >= num_sp_block);
    int nnz_per_thread = ((int *)cal)[ix+iy*num_sp_block];
    int nz_st = (ix+iy*num_sp_block)*128*maxnnz_pt;

    int col_off[4*maxnnz_pt];
    T mask_th[elmt_per_thread];  // mask thread
    T mask_v[maxnnz_pt];  // mask value
    T rsx[maxnnz_pt*16*cal_time];
    T rsu[16*cal_time];
    T rsg[16*cal_time];
    T rsr[maxnnz_pt*4];

    using SmemLayoutX = typename Config::SmemLayoutX;
    using SmemLayoutUp = typename Config::SmemLayoutUp;
    using SmemLayoutGate = typename Config::SmemLayoutGate;
    using SmemLayoutSpUp = typename Config::SmemLayoutSpUp;
    using SmemLayoutSpGate = typename Config::SmemLayoutSpGate;
    using SmemLayoutResult = typename Config::SmemLayoutResult;
    using G2SCopyX = typename Config::G2SCopyX;
    using G2SCopyUp = typename Config::G2SCopyUp;
    using G2SCopyGate = typename Config::G2SCopyGate;
    using G2SCopySpUp = typename Config::G2SCopySpUp;
    using G2SCopySpGate = typename Config::G2SCopySpGate;
    using S2RCopyAtomX = typename Config::S2RCopyAtomX;
    using S2RCopyAtomUp = typename Config::S2RCopyAtomUp;
    using S2RCopyAtomGate = typename Config::S2RCopyAtomGate;
    using S2RCopySpX = typename Config::S2RCopySpX;
    using S2RCopySpUp = typename Config::S2RCopySpUp;
    using S2RCopySpGate = typename Config::S2RCopySpGate;
    using TiledMMA = typename Config::MMA;
    using R2SCopyAtomR = typename Config::R2SCopyAtomR;
    using S2GCopyR = typename Config::S2GCopyR;

    T * shmx = shm_data;
    T * shmup = shmx + cute::cosize(SmemLayoutX{});
    T * shmgate = shmup + cute::cosize(SmemLayoutUp{});
    T * shmspup = shmgate + cute::cosize(SmemLayoutGate{});
    T * shmspgate = shmspup + cute::cosize(SmemLayoutSpUp{});

    Tensor I = make_tensor(make_gmem_ptr((T *)x), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor U = make_tensor(make_gmem_ptr((T *)up), make_shape(p, k), make_stride(k, Int<1>{}));
    Tensor G = make_tensor(make_gmem_ptr((T *)gate), make_shape(p, k), make_stride(k, Int<1>{}));
    Tensor R = make_tensor(make_gmem_ptr((T *)r), make_shape(m, p), make_stride(p, Int<1>{}));
    Tensor M = make_tensor(make_gmem_ptr((T *)mask), make_shape(m, n-p),
                           make_stride(n-p, Int<1>{}));
    Tensor SU = make_tensor(make_gmem_ptr((T *)up+p*k),
                            make_shape(n-p, k), make_stride(k, Int<1>{}));
    Tensor SG = make_tensor(make_gmem_ptr((T *)gate+p*k),
                            make_shape(n-p, k), make_stride(k, Int<1>{}));
    Tensor V = make_tensor(make_gmem_ptr((T *)val+nz_st), make_shape(Int<128*maxnnz_pt>{}));
    Tensor C = make_tensor(make_gmem_ptr((int *)col+nz_st), make_shape(Int<128*maxnnz_pt>{}));

    Tensor gX = local_tile(I, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gU = local_tile(U, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gG = local_tile(G, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gR = local_tile(R, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    Tensor gM = local_tile(M, make_tile(Int<elmt_per_thread>{}, Int<kTileS>{}),
                           make_coord(iy*4+ng, ix));
    Tensor gSU = local_tile(SU, make_tile(Int<kTileS>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gSG = local_tile(SG, make_tile(Int<kTileS>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gV = local_tile(V, make_tile(Int<maxnnz_pt>{}), make_coord(ng+ig*4+nl*32));
    Tensor gC = local_tile(C, make_tile(Int<maxnnz_pt>{}), make_coord(ng+ig*4+nl*32));

    auto sX = make_tensor(make_smem_ptr(shmx), SmemLayoutX{});
    auto sU = make_tensor(make_smem_ptr(shmup), SmemLayoutUp{});
    auto sG = make_tensor(make_smem_ptr(shmgate), SmemLayoutGate{});
    auto sSU = make_tensor(make_smem_ptr(shmspup), SmemLayoutSpUp{});
    auto sSG = make_tensor(make_smem_ptr(shmspgate), SmemLayoutSpGate{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrX = thr_mma.partition_fragment_A(gX(_, _, 0));
    auto tCrU = thr_mma.partition_fragment_B(gU(_, _, 0));
    auto tCrG = thr_mma.partition_fragment_B(gG(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gR);
    auto tCrE = thr_mma.partition_fragment_C(gR);
    auto tMrM = make_tensor(make_rmem_ptr(mask_th), make_shape(Int<elmt_per_thread>{}));
    auto tMrM_v = make_tensor(make_rmem_ptr(mask_v), make_shape(Int<maxnnz_pt>{}));
    auto tCrSX = make_tensor(
        make_rmem_ptr(rsx),
        make_shape(_4{}, _1{}, Int<maxnnz_pt>{}, _4{}, Int<cal_time>{}),
        make_stride(_1{}, _0{}, _4{}, Int<4*maxnnz_pt>{}, Int<16*maxnnz_pt>{})
    );
    auto tCrSU = make_tensor(
        make_rmem_ptr(rsu),
        make_shape(_4{}, _1{}, Int<cal_time>{}),
        make_stride(_1{}, _0{}, _4{})
    );
    auto tCrSG = make_tensor(
        make_rmem_ptr(rsg),
        make_shape(_4{}, _1{}, Int<cal_time>{}),
        make_stride(_1{}, _0{}, _4{})
    );
    auto tCrSR = make_tensor(
        make_rmem_ptr(rsr),
        make_shape(_4{}, Int<maxnnz_pt>{}, _2{}),
        make_stride(Int<maxnnz_pt>{}, _1{}, Int<maxnnz_pt*4>{})
    );
    auto tCrC = make_tensor(
        make_rmem_ptr(col_off),
        make_shape(Int<maxnnz_pt>{}, _4{}),
        make_stride(_1{}, Int<maxnnz_pt>{})
    );
    clear(tCrD);
    clear(tCrE);
    clear(tCrSR);
    clear(tMrM_v);

    // cal mask in sparse format
    // topk in one forth col where k is nz_per_thread
    auto tMgM = gM(_, ig);
    float mv;
    if (pure_dense) {
        goto mask_end;
    }
    cute::copy(tMgM, tMrM);
#pragma unroll
    for (int i = 0; i < elmt_per_thread; ++i) {
        mv = tMrM(i);
        if (tMrM_v(0) > mv) continue;
        for (int j = 1; j < nnz_per_thread+1; ++j) {
            if (j == nnz_per_thread || tMrM_v(j) > mv) {
                tCrC(j-1, 0) = i+elmt_per_thread*ng;
                tMrM_v(j-1) = mv;
                break;
            } else {
                tCrC(j-1, 0) = tCrC(j, 0);
                tMrM_v(j-1) = tMrM_v(j);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < nnz_per_thread; ++i) {
        tCrC(i, 1) = __shfl_xor_sync(0xffffffff, tCrC(i, 0), 8U, 32);
    }
#pragma unroll
    for (int i = 0; i < nnz_per_thread; ++i) {
        tCrC(i, 2) = __shfl_xor_sync(0xffffffff, tCrC(i, 0), 16U, 32);
        tCrC(i, 3) = __shfl_xor_sync(0xffffffff, tCrC(i, 1), 16U, 32);
    }
mask_end:

    // g2s setting
    G2SCopyX g2s_tiled_copy_x;
    auto g2s_thr_copy_x = g2s_tiled_copy_x.get_slice(idx);
    auto tXgX_copy = g2s_thr_copy_x.partition_S(gX);
    auto tXsX_copy = g2s_thr_copy_x.partition_D(sX);
    G2SCopyUp g2s_tiled_copy_up;
    auto g2s_thr_copy_up = g2s_tiled_copy_up.get_slice(idx);
    auto tUgU_copy = g2s_thr_copy_up.partition_S(gU);
    auto tUsU_copy = g2s_thr_copy_up.partition_D(sU);
    G2SCopyGate g2s_tiled_copy_gate;
    auto g2s_thr_copy_gate = g2s_tiled_copy_gate.get_slice(idx);
    auto tGgG_copy = g2s_thr_copy_gate.partition_S(gG);
    auto tGsG_copy = g2s_thr_copy_gate.partition_D(sG);
    G2SCopySpUp g2s_tiled_copy_spup;
    auto g2s_thr_copy_spup = g2s_tiled_copy_spup.get_slice(idx);
    auto tSUgSU_copy = g2s_thr_copy_spup.partition_S(gSU);
    auto tSUsSU_copy = g2s_thr_copy_spup.partition_D(sSU);
    G2SCopySpGate g2s_tiled_copy_spgate;
    auto g2s_thr_copy_spgate = g2s_tiled_copy_spgate.get_slice(idx);
    auto tSGgSG_copy = g2s_thr_copy_spgate.partition_S(gSG);
    auto tSGsSG_copy = g2s_thr_copy_spgate.partition_D(sSG);

    // s2r setting
    auto s2r_tiled_copy_x = make_tiled_copy_A(S2RCopyAtomX{}, tiled_mma);
    auto s2r_thr_copy_x = s2r_tiled_copy_x.get_slice(idx);
    auto tXsX = s2r_thr_copy_x.partition_S(sX);
    auto tCrX_view = s2r_thr_copy_x.retile_D(tCrX);
    auto s2r_tiled_copy_up = make_tiled_copy_B(S2RCopyAtomUp{}, tiled_mma);
    auto s2r_thr_copy_up = s2r_tiled_copy_up.get_slice(idx);
    auto tUsU = s2r_thr_copy_up.partition_S(sU);
    auto tCrU_view = s2r_thr_copy_up.retile_D(tCrU);
    auto s2r_tiled_copy_gate = make_tiled_copy_B(S2RCopyAtomGate{}, tiled_mma);
    auto s2r_thr_copy_gate = s2r_tiled_copy_gate.get_slice(idx);
    auto tGsG = s2r_thr_copy_gate.partition_S(sG);
    auto tCrG_view = s2r_thr_copy_gate.retile_D(tCrG);
    // sp s2r
    S2RCopySpX s2r_tiled_copy_spx;
    auto s2r_thr_copy_spx = s2r_tiled_copy_spx.get_slice(il);
    auto tSXsSX = s2r_thr_copy_spx.partition_S(local_tile(
        sX, make_tile(_1{}, _32{}), make_coord(_, 0)
    ));
    auto tCrSX_view = s2r_thr_copy_spx.retile_D(tCrSX);
    S2RCopySpUp s2r_tiled_copy_spup;
    auto s2r_thr_copy_spup = s2r_tiled_copy_spup.get_slice(il);
    auto tSUsSU = s2r_thr_copy_spup.partition_S(local_tile(
        sSU, make_tile(_1{}, _32{}), make_coord(_, 0)
    ));
    auto tCrSU_view = s2r_thr_copy_spup.retile_D(tCrSU);
    S2RCopySpGate s2r_tiled_copy_spgate;
    auto s2r_thr_copy_spgate = s2r_tiled_copy_spgate.get_slice(il);
    auto tSGsSG = s2r_thr_copy_spgate.partition_S(local_tile(
        sSG, make_tile(_1{}, _32{}), make_coord(_, 0)
    ));
    auto tCrSG_view = s2r_thr_copy_spgate.retile_D(tCrSG);

    // stage
    int itile_to_read = kStage - 1;  // first tile read that has not been submitted 
    int ismem_read = 0;              // smem that calculation is working on
    int ismem_write = kStage - 1;    // smem write that has not been submitted

    if (!pure_dense) {
        // submit kStage - 1 g2s read
#pragma unroll
        for (int istage = 0; istage < kStage-1; ++istage) {
            cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, istage), tXsX_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_up, tUgU_copy(_, _, _, istage), tUsU_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_gate, tGgG_copy(_, _, _, istage), tGsG_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_spup, tSUgSU_copy(_, _, _, istage),
                        tSUsSU_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_spgate, tSGgSG_copy(_, _, _, istage),
                        tSGsSG_copy(_, _, _, istage));
            cp_async_fence();
        }

        // first s2r read (x & up)
        cp_async_wait<kStage - 2>();
        __syncthreads();
        cute::copy(s2r_tiled_copy_x, tXsX(_, _, 0, ismem_read), tCrX_view(_, _, 0));
        cute::copy(s2r_tiled_copy_up, tUsU(_, _, 0, ismem_read), tCrU_view(_, _, 0));
#pragma unroll
        for (int cpx = 0; cpx < 4; ++cpx) {
#pragma unroll
            for (int cpn = 0; cpn < 2; ++cpn) {
                cute::copy(s2r_tiled_copy_spx, tSXsSX(_, _, 0, tCrC(cpn, cpx^ng), ismem_read),
                           tCrSX_view(_, _, cpn, cpx, 0));
            }
        }
        cute::copy(s2r_tiled_copy_spup, tSUsSU(_, _, 0, 8*nl+ig, ismem_read), tCrSU_view(_, _, 0));

        // loop over k
        int ntile = k / kTileK;
#pragma unroll 1
        for (int itile = 0; itile < ntile; ++itile) {
            // loop in kTileK
            int nk = size<2>(tCrU);
#pragma unroll
            for (int ik = 0; ik < nk; ++ik) {
                // read gate -> cal up -> read next x & up -> cal gate
                // read gate
                cute::copy(s2r_tiled_copy_gate, tGsG(_, _, ik, ismem_read), tCrG_view(_, _, ik));
                cute::copy(s2r_tiled_copy_spgate, tSGsSG(_, _, ik, 8*nl+ig, ismem_read),
                           tCrSU_view(_, _, ik));
                // cal up
                cute::gemm(tiled_mma, tCrD, tCrX(_, _, ik), tCrU(_, _, ik), tCrD);
                // cal sp up
#pragma unroll
                for (int spx = 0; spx < 4; ++spx) {
#pragma unroll
                    for (int spn = 0; spn < 2; ++spn) {
#pragma unroll
                        for (int spy = 0; spy < 4; ++spy) {
                            // tCrSR(spx, spn, 0) += tCrSX(spy, 0, spn, spx, ik) * tCrSU(spy, 0, ik);
                        }
                    }
                }

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
#pragma unroll
                for (int cpn = 0; cpn < 2; ++cpn) {
#pragma unroll
                    for (int cpx = 0; cpx < 4; ++cpx) {
                        cute::copy(s2r_tiled_copy_spx,
                                   tSXsSX(_, _, ik_n, tCrC(cpn, cpx^ng), ismem_read),
                                   tCrSX_view(_, _, cpn, cpx, ik_n));
                    }
                }
                cute::copy(s2r_tiled_copy_spup, tSUsSU(_, _, ik_n, 8*nl+ig, ismem_read),
                        tCrSU_view(_, _, ik_n));
                if (ik == 0) {
                    // submit next g2s read
                    if (itile_to_read < ntile) {
                        cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, itile_to_read),
                                tXsX_copy(_, _, _, ismem_write));
                        cute::copy(g2s_tiled_copy_up, tUgU_copy(_, _, _, itile_to_read),
                                tUsU_copy(_, _, _, ismem_write));
                        cute::copy(g2s_tiled_copy_gate, tGgG_copy(_, _, _, itile_to_read),
                                tGsG_copy(_, _, _, ismem_write));
                        cute::copy(g2s_tiled_copy_spup, tSUgSU_copy(_, _, _, itile_to_read),
                                tSUsSU_copy(_, _, _, ismem_write));
                        cute::copy(g2s_tiled_copy_spgate, tSGgSG_copy(_, _, _, itile_to_read),
                                tSGsSG_copy(_, _, _, ismem_write));
                        ++itile_to_read;
                        ismem_write = (ismem_write + 1) % kStage;
                    }
                    cp_async_fence();
                }
                // cal gate
                cute::gemm(tiled_mma, tCrE, tCrX(_, _, ik), tCrG(_, _, ik), tCrE);
                // cal sp gate
#pragma unroll
                for (int spx = 0; spx < 4; ++spx) {
#pragma unroll
                    for (int spn = 0; spn < 2; ++spn) {
#pragma unroll
                        for (int spy = 0; spy < 4; ++spy) {
                            // tCrSR(spx, spn, 1) += tCrSX(spy, 0, spn, spx, ik) * tCrSG(spy, 0, ik);
                        }
                    }
                }
            }
        }

        // sp add
#pragma unroll
        for (int spn = 0; spn < nnz_per_thread; ++spn) {
#pragma unroll
            for (int spx = 0; spx < 4; ++spx) {
                tCrSR(spx, spn, 0) +=
                    static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(spx, spn, 0), 8U, 32));
                tCrSR(spx, spn, 0) +=
                    static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(spx, spn, 0), 16U, 32));
                tCrSR(spx, spn, 1) +=
                    static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(spx, spn, 1), 8U, 32));
                tCrSR(spx, spn, 1) +=
                    static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(spx, spn, 1), 16U, 32));
            }
        }
    } else {
        // submit kStage - 1 g2s read
#pragma unroll
        for (int istage = 0; istage < kStage-1; ++istage) {
            cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, istage), tXsX_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_up, tUgU_copy(_, _, _, istage), tUsU_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_gate, tGgG_copy(_, _, _, istage),
                       tGsG_copy(_, _, _, istage));
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
                // read gate -> cal up -> read next x & up -> cal gate
                // read gate
                cute::copy(s2r_tiled_copy_gate, tGsG(_, _, ik, ismem_read), tCrG_view(_, _, ik));
                // cal up
                cute::gemm(tiled_mma, tCrD, tCrX(_, _, ik), tCrU(_, _, ik), tCrD);

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
                        cute::copy(g2s_tiled_copy_gate, tGgG_copy(_, _, _, itile_to_read),
                                tGsG_copy(_, _, _, ismem_write));
                        ++itile_to_read;
                        ismem_write = (ismem_write + 1) % kStage;
                    }
                    cp_async_fence();
                }
                // cal gate
                cute::gemm(tiled_mma, tCrE, tCrX(_, _, ik), tCrG(_, _, ik), tCrE);
            }
        }
    }

    // relu and mul
    T * d_data = tCrD.data();
    T * e_data = tCrE.data();
    float lhs, rhs;
#pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
        lhs = d_data[i];
        rhs = e_data[i];
        lhs = (lhs>0&&rhs>0)?(lhs*rhs):0;
        d_data[i] = static_cast<T>(lhs);
    }
    if (!pure_dense) {
#pragma unroll
        for (int spn = 0; spn < nnz_per_thread; ++spn) {
            lhs = tCrSR(ng, spn, 0);
            rhs = tCrSR(ng, spn, 1);
            lhs = (lhs>0&&rhs>0)?(lhs*rhs):0;
            tCrSR(ng, spn, 0) = static_cast<T>(lhs);
        }
        cute::copy(tCrSR(ng, _, 0), gV);
        cute::copy(tCrC(_, 0), gC);
    }

    // epi
    auto sD = make_tensor(sX(_, _, ismem_read).data(), SmemLayoutResult{});

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

    int step = size<3>(tCsD_r2s);
    int numcp = size<1>(tCrD_r2sx);
    for (int i = 0; i < numcp; i += step) {
        // reg -> shm
    #pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(r2s_tiled_copy_r, tCrD_r2sx(_, i + j), tCsD_r2s(_, 0, 0, j));
        }
        __syncthreads();
    
        // shm -> global
    #pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, 0, 0, j), tRgR_s2gx(_, i + j));
        }
        __syncthreads();
    }
}


int main() {
    using T = cute::half_t;
    using namespace cute;
    using X = Underscore;

    srand(1);

    const int m = 1024;
    const int n = 14336;
    const int k = 4096;
    const int p = 9600;
    const int nnz_thread = 2;
    const int num_block = (n-p) * m / 256 / 32;
    const int nnz = 4 * 128 * num_block;

    cublasHandle_t handle;
    cublasCreate(&handle);

    T * h_x    = new T[m * k];
    T * h_up   = new T[n * k];
    T * h_gate = new T[n * k];
    T * h_mask = new T[(n-p) * k];
    T * h_r1   = new T[m * n];
    T * h_r2   = new T[m * n];
    T * h_r3   = new T[m * n];
    T * h_val  = new T[nnz];
    int * h_col_off = new int[nnz];
    int * h_cal = new int[num_block];

    T * d_x, * d_up, * d_gate, * d_mask, * d_r1, * d_r2, * d_r3, * d_val;
    int * d_col_off, * d_cal;

    CUDA_CHECK(cudaMalloc(&d_x,    sizeof(T) * m * k));
    CUDA_CHECK(cudaMalloc(&d_up,   sizeof(T) * n * k));
    CUDA_CHECK(cudaMalloc(&d_gate, sizeof(T) * n * k));
    CUDA_CHECK(cudaMalloc(&d_mask, sizeof(T) * (n-p) * k));
    CUDA_CHECK(cudaMalloc(&d_r1,   sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(&d_r2,   sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(&d_r3,   sizeof(T) * m * n));
    CUDA_CHECK(cudaMalloc(&d_val,  sizeof(T) * nnz));
    CUDA_CHECK(cudaMalloc(&d_col_off, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_cal, sizeof(int) * num_block));

    auto tx  = make_tensor(h_x,    make_shape(m, k),   make_stride(k, 1));
    auto tu  = make_tensor(h_up,   make_shape(n, k),   make_stride(k, 1));
    auto tg  = make_tensor(h_gate, make_shape(n, k),   make_stride(k, 1));
    auto tm  = make_tensor(h_mask, make_shape(n-p, k), make_stride(k, 1));
    auto tr1 = make_tensor(h_r1,   make_shape(m, n),   make_stride(n, 1));
    auto tr2 = make_tensor(h_r2,   make_shape(m, n),   make_stride(n, 1));
    auto tr3 = make_tensor(h_r3,   make_shape(m, n),   make_stride(n, 1));
    auto tv  = make_tensor(h_val,  make_shape(nnz),    make_stride(1));

    cpu_rand_data(&tx);
    cpu_rand_data(&tu);
    cpu_rand_data(&tg);
    cpu_rand_data(&tm, 0, 100);
    clear(tv);
    clear(tr1);
    clear(tr2);
    clear(tr3);

    for (int i = 0; i < num_block; ++i) {
        h_cal[i] = nnz_thread;
    }

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(T)*m*k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up, sizeof(T)*n*k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate, sizeof(T)*n*k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask, sizeof(T)*(n-p)*k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cal, h_cal, sizeof(int)*num_block, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_r1, 0, sizeof(T)*m*n));
    CUDA_CHECK(cudaMemset(d_r2, 0, sizeof(T)*m*n));
    CUDA_CHECK(cudaMemset(d_r3, 0, sizeof(T)*m*n));
    CUDA_CHECK(cudaMemset(d_val, 0, sizeof(T)*nnz));
    CUDA_CHECK(cudaMemset(d_col_off, 0, sizeof(int)*nnz));

    // calculation

    constexpr int kTileM = 256;
    constexpr int kTileN = 64;
    constexpr int kTileK = 32;
    constexpr int kTileS = 32;
    config::Config<T, kTileM, kTileN, kTileK, kTileS> cfg;
    dim3 block(128);
    dim3 grid((p + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
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
        half alpha = 1.f;
        half beta = 0.f;
        CUDA_CHECK(cudaMemset(d_r2, 0, sizeof(T)*m*n));
        CUDA_CHECK(cudaMemset(d_r3, 0, sizeof(T)*m*n));
        CUDA_CHECK(cudaEventRecord(cublas_start));
        // ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        //                   &alpha, (half *)d_up, k, (half *)d_x, k,
        //                   &beta, (half *)d_r2, n);
        // ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        //                   &alpha, (half *)d_gate, k, (half *)d_x, k,
        //                   &beta, (half *)d_r3, n);
        // dim3 gmr((m*n+255)/256);
        // dim3 bmr(256);
        // gpu_hmul_relu<<<gmr, bmr>>>((half *)d_r2, (half *)d_r3, m*n);
        CUDA_CHECK(cudaEventRecord(cublas_end));
        CUDA_CHECK(cudaEventSynchronize(cublas_end));
        if (ret != CUBLAS_STATUS_SUCCESS) {
            printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
        }

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, cublas_start, cublas_end));
        if (i != 0) cublas_avg_time += elapsed_time;

        // ours
        CUDA_CHECK(cudaMemset(d_r1, 0, sizeof(T)*m*n));
        CUDA_CHECK(cudaEventRecord(ours_start));
        cudaFuncSetAttribute(sg<decltype(cfg)>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        sg<decltype(cfg)><<<grid, block, shm_size>>>(d_x, d_up, d_gate, d_mask, d_cal,
                                                     d_r1, d_val, d_col_off, m, n, p, k);
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
    CUDA_CHECK(cudaMemcpy(h_r3, d_r3, sizeof(T)*m*n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val, d_val, sizeof(T)*nnz, cudaMemcpyDeviceToHost));

    // gpu_compare(d_r1, d_r2, p * m, 0.1f);
    // gpu_compare_sp(d_val, d_row_ind, d_col_off, d_r2+p*n, nnz, n, 0.1f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_r1));
    CUDA_CHECK(cudaFree(d_r2));
    CUDA_CHECK(cudaFree(d_r3));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_col_off));
    CUDA_CHECK(cudaFree(d_cal));

    delete[] h_x;
    delete[] h_up;
    delete[] h_gate;
    delete[] h_mask;
    delete[] h_r1;
    delete[] h_r2;
    delete[] h_r3;
    delete[] h_val;
    delete[] h_col_off;
    delete[] h_cal;
    
    cublasDestroy(handle);
}
