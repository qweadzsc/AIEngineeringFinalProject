#include "dsmm.h"
#include "sddmm.h"


using T = cute::half_t;


template<typename Config>
__global__ static void
sddmm(const T * x, const T * up, const T * gate, T * r,
      const long long * row, const long long * col, T * val,
      int m, int n, int k, int e, int p)
{
    /*
     * e: inter_size of one expert
     * p: num of ds experts
     */

    using namespace cute;
    using X = Underscore;

    int ix = blockIdx.x;  // pos in expert, 0-(block_per_expert-1)
    int iy = blockIdx.y;
    int iz = blockIdx.z;  // expert id
    int idx = threadIdx.x;
    int debug = (ix + 48*iz) * 128 + idx;

    // init
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kTileS = Config::kTileS;
    constexpr int maxnnz = Config::maxnnz;
    constexpr int kStage = Config::kStage;

    constexpr int sp_per_ds = Config::sp_per_ds;
    constexpr int repeat_x = Config::repeat_x;
    constexpr int repeat_y = Config::repeat_y;
    int block_per_expert = e / kTileN;
    int six = (iz % sp_per_ds) * block_per_expert + ix;
    int siz = iz / sp_per_ds;

    int dsn = p * e;
    int spw = dsn / sp_per_ds;
    int num_experts = n / e;
    int nt = gridDim.y;

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
    using S2RCopySpAtom = typename Config::S2RCopySpAtom;
    using TiledMMA = typename Config::MMA;
    using R2SCopyAtomR = typename Config::R2SCopyAtomR;
    using S2GCopyR = typename Config::S2GCopyR;

    extern __shared__ T shm_data[];
    T * shmx = shm_data;
    T * shmup = shmx + cute::cosize(SmemLayoutX{});
    T * shmgate = shmup + cute::cosize(SmemLayoutUp{});
    T * shmspup = shmgate + cute::cosize(SmemLayoutGate{});
    T * shmspgate = shmspup + cute::cosize(SmemLayoutSpUp{});
    long long * shmnzp = (long long *)(shmspgate + cute::cosize(SmemLayoutSpGate{}));

    Tensor I = make_tensor(make_gmem_ptr(x), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor U = make_tensor(make_gmem_ptr(up), make_shape(e, k, num_experts),
                           make_stride(k, Int<1>{}, e*k));
    Tensor G = make_tensor(make_gmem_ptr(gate), make_shape(e, k, num_experts),
                           make_stride(k, Int<1>{}, e*k));
    Tensor R = make_tensor(make_gmem_ptr(r), make_shape(m, dsn, _2{}),
                           make_stride(dsn, Int<1>{}, m*dsn));
    Tensor V = make_tensor(make_gmem_ptr(val), make_shape(Int<maxnnz>{}, spw, nt, _2{}),
                           make_stride(spw, Int<1>{}, maxnnz*spw, maxnnz*spw*nt));
    Tensor Row = make_tensor(make_gmem_ptr(row), make_shape(Int<maxnnz>{}, p/sp_per_ds, nt),
                             make_stride(p/sp_per_ds, _1{}, p/sp_per_ds*maxnnz));
    Tensor Col = make_tensor(make_gmem_ptr(col), make_shape(nt, p+p/sp_per_ds),
                             make_stride(p+p/sp_per_ds, _1{}));

    // nnz pos
    long long real_iz = Col(iy, iz);
    long long real_siz = Col(iy, p+siz);
    array<long long, repeat_y> nz_rows_data;
    auto nz_rows = make_tensor(make_rmem_ptr(nz_rows_data.data()), make_shape(Int<repeat_y>{}));
    if (idx < maxnnz) {  // maxnnz <= 64
        shmnzp[idx] = Row(idx, siz, iy);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < repeat_y; ++i) {
        nz_rows(i) = shmnzp[i];
    }
    __syncthreads();

    Tensor gX = local_tile(I, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gU = local_tile(U, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _, real_iz));
    Tensor gG = local_tile(G, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _, real_iz));
    Tensor gR = local_tile(R, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                           make_coord(iy, ix+block_per_expert*iz, _));
    Tensor gSU = local_tile(U, make_tile(Int<kTileS>{}, Int<kTileK>{}),
                            make_coord(six, _, real_siz));
    Tensor gSG = local_tile(G, make_tile(Int<kTileS>{}, Int<kTileK>{}),
                            make_coord(six, _, real_siz));
    Tensor gV = local_tile(V, make_tile(Int<maxnnz>{}, Int<kTileS>{}),
                           make_coord(0, ix+block_per_expert*iz, iy, _));

    auto sX = make_tensor(make_smem_ptr(shmx), SmemLayoutX{});
    auto sU = make_tensor(make_smem_ptr(shmup), SmemLayoutUp{});
    auto sG = make_tensor(make_smem_ptr(shmgate), SmemLayoutGate{});
    auto sSU = make_tensor(make_smem_ptr(shmspup), SmemLayoutSpUp{});
    auto sSG = make_tensor(make_smem_ptr(shmspgate), SmemLayoutSpGate{});

    array<T, 16*repeat_y*2> spx_data;
    array<T, 16*repeat_x*2> spup_data;
    array<T, 16*repeat_x*2> spgate_data;
    array<T, 2*repeat_x*repeat_y> spr_data;
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrX = thr_mma.partition_fragment_A(gX(_, _, 0));
    auto tCrU = thr_mma.partition_fragment_B(gU(_, _, 0));
    auto tCrG = thr_mma.partition_fragment_B(gG(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gR(_, _, 0));
    auto tCrE = thr_mma.partition_fragment_C(gR(_, _, 0));
    auto tCrSX = make_tensor(make_rmem_ptr(spx_data.data()),
        make_shape(Int<repeat_y>{}, _16{}, _2{}), make_stride(_16{}, _1{}, Int<16*repeat_y>{}));
    auto tCrSU = make_tensor(make_rmem_ptr(spup_data.data()),
        make_shape(Int<repeat_x>{}, _16{}, _2{}), make_stride(_16{}, _1{}, Int<16*repeat_x>{}));
    auto tCrSG = make_tensor(make_rmem_ptr(spgate_data.data()),
        make_shape(Int<repeat_x>{}, _16{}, _2{}), make_stride(_16{}, _1{}, Int<16*repeat_x>{}));
    auto tCrSR = make_tensor(make_rmem_ptr(spr_data.data()),
        make_shape(Int<repeat_y>{}, Int<repeat_x>{}, _2{}),
        make_stride(Int<repeat_x>{}, _1{}, Int<repeat_y*repeat_x>{}));
    clear(tCrD);
    clear(tCrE);
    clear(tCrSR);

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
    S2RCopySpAtom s2r_tiled_copy_sp_atom;
    auto tSXsX = local_tile(sX, make_tile(_1{}, _8{}), make_coord(_, _))(0, _, _, _, _);
    auto tSUsSU = local_tile(sSU, make_tile(_128{}, _8{}), make_coord(0, _))(idx, _, _, _);
    auto tSGsSG = local_tile(sSG, make_tile(_128{}, _8{}), make_coord(0, _))(idx, _, _, _);
    auto tSXrSX_s2r = local_tile(tCrSX, make_tile(Int<repeat_y>{}, _8{}), make_coord(_, _));
    auto tSUrSU_s2r = local_tile(tCrSU, make_tile(Int<repeat_x>{}, _8{}), make_coord(_, _));
    auto tSGrSG_s2r = local_tile(tCrSG, make_tile(Int<repeat_x>{}, _8{}), make_coord(_, _));

    // stage
    int itile_to_read = kStage - 1;  // first tile read that has not been submitted 
    int ismem_read = 0;              // smem that calculation is working on
    int ismem_write = kStage - 1;    // smem write that has not been submitted

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
    for (int spy = 0; spy < repeat_y; ++spy) {
        cute::copy(s2r_tiled_copy_sp_atom, tSXsX(_, nz_rows(spy), 0, ismem_read),
                   tSXrSX_s2r(spy, _, 0, 0, 0));
        cute::copy(s2r_tiled_copy_sp_atom, tSXsX(_, nz_rows(spy), 1, ismem_read),
                   tSXrSX_s2r(spy, _, 0, 1, 0));
    }
#pragma unroll
    for (int spx = 0; spx < repeat_x; ++spx) {
        cute::copy(s2r_tiled_copy_sp_atom, tSUsSU(_, 0, ismem_read),
                   tSUrSU_s2r(spx, _, 0, 0, 0));
        cute::copy(s2r_tiled_copy_sp_atom, tSUsSU(_, 1, ismem_read),
                   tSUrSU_s2r(spx, _, 0, 1, 0));
    }

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
#pragma unroll
            for (int spx = 0; spx < repeat_x; ++spx) {
                cute::copy(s2r_tiled_copy_sp_atom, tSGsSG(_, 2*ik+0, ismem_read),
                           tSGrSG_s2r(spx, _, 0, 0, ik));
                cute::copy(s2r_tiled_copy_sp_atom, tSGsSG(_, 2*ik+1, ismem_read),
                           tSGrSG_s2r(spx, _, 0, 1, ik));
            }
            // cal up
            cute::gemm(tiled_mma, tCrD, tCrX(_, _, ik), tCrU(_, _, ik), tCrD);
            // cal sp up
#pragma unroll
            for (int spx = 0; spx < repeat_x; ++spx) {
#pragma unroll
                for (int spy = 0; spy < repeat_y; ++spy) {
#pragma unroll
                    for (int spz = 0; spz < 16; ++spz) {
                        tCrSR(spy, spx, 0) += tCrSX(spy, spz, ik) * tCrSU(spx, spz, ik);
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
            for (int spy = 0; spy < repeat_y; ++spy) {
                cute::copy(s2r_tiled_copy_sp_atom, tSXsX(_, nz_rows(spy), 2*ik_n+0, ismem_read),
                           tSXrSX_s2r(spy, _, 0, 0, ik_n));
                cute::copy(s2r_tiled_copy_sp_atom, tSXsX(_, nz_rows(spy), 2*ik_n+1, ismem_read),
                           tSXrSX_s2r(spy, _, 0, 1, ik_n));
            }
#pragma unroll
            for (int spx = 0; spx < repeat_x; ++spx) {
                cute::copy(s2r_tiled_copy_sp_atom, tSUsSU(_, 2*ik_n+0, ismem_read),
                           tSUrSU_s2r(spx, _, 0, 0, ik_n));
                cute::copy(s2r_tiled_copy_sp_atom, tSUsSU(_, 2*ik_n+1, ismem_read),
                           tSUrSU_s2r(spx, _, 0, 1, ik_n));
            }
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
            for (int spx = 0; spx < repeat_x; ++spx) {
#pragma unroll
                for (int spy = 0; spy < repeat_y; ++spy) {
#pragma unroll
                    for (int spz = 0; spz < 16; ++spz) {
                        tCrSR(spy, spx, 1) += tCrSX(spy, spz, ik) * tCrSG(spx, spz, ik);
                    }
                }
            }
        }
    }

    // silu
    T * d_data = tCrD.data();
    T * e_data = tCrE.data();
    float lhs, rhs;
#pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
        lhs = static_cast<float>(d_data[i]);
        rhs = static_cast<float>(e_data[i]);
        lhs = lhs * rhs / (1.0f + expf(-rhs));
        d_data[i] = static_cast<T>(lhs);
    }
#pragma unroll
    for (int spx = 0; spx < repeat_x; ++spx) {
#pragma unroll
        for (int spy = 0; spy < repeat_y; ++spy) {
            lhs = static_cast<float>(tCrSR(spy, spx, 0));
            rhs = static_cast<float>(tCrSR(spy, spx, 1));
            lhs = lhs * rhs / (1.0f + expf(-rhs));
            tCrSR(spy, spx, 0) = static_cast<T>(lhs);
        }
    }

    // epi
    auto sD = make_tensor(sU(_, _, ismem_read).data(), SmemLayoutResult{});

    auto r2s_tiled_copy_r = make_tiled_copy_C(R2SCopyAtomR{}, tiled_mma);
    auto r2s_thr_copy_r = r2s_tiled_copy_r.get_slice(idx);
    auto tCsD_r2s = r2s_thr_copy_r.partition_D(sD);
    auto tCrD_r2s = r2s_thr_copy_r.retile_S(tCrD);
    auto tCrD_r2sx = group_modes<1, 3>(tCrD_r2s);
    auto tCrE_r2s = r2s_thr_copy_r.retile_S(tCrE);
    auto tCrE_r2sx = group_modes<1, 3>(tCrE_r2s);

    S2GCopyR s2g_tiled_copy_r;
    auto s2g_thr_copy_r = s2g_tiled_copy_r.get_thread_slice(idx);
    auto tRsR_s2g = s2g_thr_copy_r.partition_S(sD);
    auto tRgR_s2g = s2g_thr_copy_r.partition_D(gR);
    auto tRgR_s2gx = group_modes<1, 3>(tRgR_s2g);

    int step = size<3>(tCsD_r2s);  // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrD_r2sx); i += step) {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j) {
            auto t = make_tensor_like<T>(tCrD_r2sx(_, i + j));
            cute::copy(tCrD_r2sx(_, i + j), t);
            cute::copy(r2s_tiled_copy_r, t, tCsD_r2s(_, 0, 0, j));
        }
        __syncthreads();

        // shm -> global
#pragma unroll
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, 0, 0, j), tRgR_s2gx(_, i + j, 0));
        }
        __syncthreads();
    }

// #pragma unroll
//     for (int i = 0; i < size<1>(tCrE_r2sx); i += step) {
//         // reg -> shm
// #pragma unroll
//         for (int j = 0; j < step; ++j) {
//             auto t = make_tensor_like<T>(tCrE_r2sx(_, i + j));
//             cute::copy(tCrE_r2sx(_, i + j), t);
//             cute::copy(r2s_tiled_copy_r, t, tCsD_r2s(_, 0, 0, j));
//         }
//         __syncthreads();

//         // shm -> global
// #pragma unroll
//         for (int j = 0; j < step; ++j) {
//             cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, 0, 0, j), tRgR_s2gx(_, i + j, 1));
//         }
//         __syncthreads();
//     }

    // sp epi
#pragma unroll
    for (int spx = 0; spx < repeat_x; ++spx) {
#pragma unroll
        for (int spy = 0; spy < repeat_y; ++spy) {
            gV(spy, idx, 0) = tCrSR(spy, spx, 0);
        }
    }
    __syncthreads();

// #pragma unroll
//     for (int spx = 0; spx < repeat_x; ++spx) {
// #pragma unroll
//         for (int spy = 0; spy < repeat_y; ++spy) {
//             gV(spy, idx, 1) = tCrSR(spy, spx, 1);
//         }
//     }
//     __syncthreads();
}


template <int kTileM_, int maxnnz_>
static void sddmm_var_m(const half * x, const half * up, const half * gate, half * r,
               const long long * row, const long long * col, half * val,
               int m, int n, int k, int e, int p)
{
    constexpr int kTileM = kTileM_;
    constexpr int kTileN = 128;
    constexpr int kTileK = 32;
    constexpr int kTileS = 128;
    constexpr int maxnnz = maxnnz_;
    config::Config<T, kTileM, kTileN, kTileK, kTileS, maxnnz> cfg;
    dim3 block(128);
    dim3 grid((e + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM, p);
    int shm_size = cfg.kShmSize;
    cudaFuncSetAttribute(sddmm<decltype(cfg)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    sddmm<decltype(cfg)><<<grid, block, shm_size>>>(
        (const T *)x, (const T *)up, (const T *)gate, (T *)r, row, col, (T *)val, m, n, k, e, p);
    CUDA_CHECK(cudaGetLastError());
}


#define SDDMM_DISPATCH(M_VAL, MN_VAL) \
    sddmm_var_m<M_VAL, MN_VAL>(x, up, gate, r, row, col, val, m, n, k, e, p)

template<int M>
void dispatch_mn(const half * x, const half * up, const half * gate, half * r,
               const long long * row, const long long * col, half * val,
               int m, int n, int k, int e, int p, int mn)
{
    switch (mn) {
        case 1: SDDMM_DISPATCH(M, 1); break;
        case 2: SDDMM_DISPATCH(M, 2); break;
        case 3: SDDMM_DISPATCH(M, 3); break;
        case 4: SDDMM_DISPATCH(M, 4); break;
        case 5: SDDMM_DISPATCH(M, 5); break;
        case 6: SDDMM_DISPATCH(M, 6); break;
        case 7: SDDMM_DISPATCH(M, 7); break;
        case 8: SDDMM_DISPATCH(M, 8); break;
        default: throw std::invalid_argument("MN must be between 1 and 8");
    }
}

#define SDDMM_DISPATCH_MN(M_VAL) \
    dispatch_mn<M_VAL>(x, up, gate, r, row, col, val, m, n, k, e, p, mn)

void sddmm_api(const half * x, const half * up, const half * gate, half * r,
               const long long * row, const long long * col, half * val,
               int m, int n, int k, int e, int p, int mn)
{
    switch (m) {
        case 32:  SDDMM_DISPATCH_MN(32); break;
        case 64:  SDDMM_DISPATCH_MN(64); break;
        case 128: SDDMM_DISPATCH_MN(128); break;
        default: throw std::invalid_argument("M must be 32, 64, or 128");
    }
}

#undef SDDMM_DISPATCH
#undef SDDMM_DISPATCH_MN
