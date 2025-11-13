#include "dsmm.h"
#include "config_sddmm.h"


using T = cute::half_t;


template<typename Config>
__global__ static void
ds(const T * x, const T * up, const T * gate, T * r,
   const long long * row, const long long * col, T * val,
   int m, int n, int k, int p)
{
    /*
     * x: n x k in row-major
     * up: m x k in row-major
     * gate: m x k in row-major
     * r: n x m in row-major
     * 
     * For dense part, r^T = relu(up @ x^T) * relu(gate @ x^T)
     * For sparse part, val = mask(up @ x^T * gate @ x^T)
     */

    using namespace cute;
    using X = Underscore;

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int idx = threadIdx.x;
    int debug = (ix + iy) * 128 + idx;

    // init
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;
    constexpr int maxnnz = Config::maxnnz;
    constexpr int cal_time_block = Config::cal_time_block;
    constexpr int sp_cal_num = Config::sp_cal_num;
    constexpr int KForTile = Config::KForTile;
    constexpr int NForTile = Config::NForTile;
    constexpr int sp_cal_k = KForTile / 8;
    constexpr int sp_cal_n = NForTile / 32;

    int sp_ix = ix * kTileN + iy % kTileN;
    int sp_iy = iy / kTileN;

    constexpr int g2s_copy_num_thread = maxnnz / 16;
    long long col_off[g2s_copy_num_thread];
    int nnz_col = row[sp_ix+1] - row[sp_ix];
    int nz_st = sp_ix*nnz_col+sp_iy*maxnnz;
#pragma unroll
    for (int i = 0; i < g2s_copy_num_thread; ++i) {
        col_off[i] = col[nz_st+i*16+idx/8];
    }  // TODO: use cute::copy

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

    extern __shared__ T shm_data[];
    T * shmx = shm_data;
    T * shmup = shmx + cute::cosize(SmemLayoutX{});
    T * shmgate = shmup + cute::cosize(SmemLayoutUp{});
    T * shmspup = shmgate + cute::cosize(SmemLayoutGate{});
    T * shmspgate = shmspup + cute::cosize(SmemLayoutSpUp{});

    Tensor I = make_tensor(make_gmem_ptr((T *)x), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor U = make_tensor(make_gmem_ptr((T *)up), make_shape(p, k), make_stride(k, Int<1>{}));
    Tensor G = make_tensor(make_gmem_ptr((T *)gate), make_shape(p, k), make_stride(k, Int<1>{}));
    Tensor R = make_tensor(make_gmem_ptr((T *)r), make_shape(p, n), make_stride(n, Int<1>{}));
    Tensor SU = make_tensor(make_gmem_ptr((T *)up+p*k),
                            make_shape(m-p, k), make_stride(k, Int<1>{}));
    Tensor SG = make_tensor(make_gmem_ptr((T *)gate+p*k),
                            make_shape(m-p, k), make_stride(k, Int<1>{}));

    Tensor gX = local_tile(I, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gU = local_tile(U, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gG = local_tile(G, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gR = local_tile(R, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    Tensor gSU = local_tile(SU, make_tile(Int<1>{}, Int<kTileK>{}), make_coord(_, _));
    Tensor gSG = local_tile(SG, make_tile(Int<1>{}, Int<kTileK>{}), make_coord(_, _));

    auto sX = make_tensor(make_smem_ptr(shmx), SmemLayoutX{});
    auto sU = make_tensor(make_smem_ptr(shmup), SmemLayoutUp{});
    auto sG = make_tensor(make_smem_ptr(shmgate), SmemLayoutGate{});
    auto sSU = make_tensor(make_smem_ptr(shmspup), SmemLayoutSpUp{});
    auto sSG = make_tensor(make_smem_ptr(shmspgate), SmemLayoutSpGate{});

    T rsx[sp_cal_k*2*cal_time_block];
    T rsu[sp_cal_num*cal_time_block];
    T rsg[sp_cal_num*cal_time_block];
    T rsr[sp_cal_n*2];
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrU = thr_mma.partition_fragment_A(gU(_, _, 0));
    auto tCrG = thr_mma.partition_fragment_A(gG(_, _, 0));
    auto tCrX = thr_mma.partition_fragment_B(gX(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gR);
    auto tCrE = thr_mma.partition_fragment_C(gR);
    auto tCrSX = make_tensor(
        make_rmem_ptr(rsx),
        make_shape((make_shape(_2{}, Int<sp_cal_k>{})), _1{}, Int<cal_time_block>{}),
        make_stride((make_stride(_1{}, _2{})), _0{}, Int<2*sp_cal_k>{})
    );
    auto tCrSU = make_tensor(
        make_rmem_ptr(rsu),
        make_shape((make_shape(_2{}, Int<sp_cal_n>{}, Int<sp_cal_k>{})),
                    _1{}, Int<cal_time_block>{}),
        make_stride((make_stride(_1{}, _2{}, Int<2*sp_cal_n>{})), _0{}, Int<sp_cal_num>{})
    );
    auto tCrSG = make_tensor(
        make_rmem_ptr(rsg),
        make_shape((make_shape(_2{}, Int<sp_cal_n>{}, Int<sp_cal_k>{})),
                    _1{}, Int<cal_time_block>{}),
        make_stride((make_stride(_1{}, _2{}, Int<2*sp_cal_n>{})), _0{}, Int<sp_cal_num>{})
    );
    auto tCrSR = make_tensor(
        make_rmem_ptr(rsr),
        make_shape(Int<sp_cal_n>{}, _2{}),
        make_stride(_2{}, _1{})
    );
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
    auto s2r_tiled_copy_x = make_tiled_copy_B(S2RCopyAtomX{}, tiled_mma);
    auto s2r_thr_copy_x = s2r_tiled_copy_x.get_slice(idx);
    auto tXsX = s2r_thr_copy_x.partition_S(sX);
    auto tCrX_view = s2r_thr_copy_x.retile_D(tCrX);
    auto s2r_tiled_copy_up = make_tiled_copy_A(S2RCopyAtomUp{}, tiled_mma);
    auto s2r_thr_copy_up = s2r_tiled_copy_up.get_slice(idx);
    auto tUsU = s2r_thr_copy_up.partition_S(sU);
    auto tCrU_view = s2r_thr_copy_up.retile_D(tCrU);
    auto s2r_tiled_copy_gate = make_tiled_copy_A(S2RCopyAtomGate{}, tiled_mma);
    auto s2r_thr_copy_gate = s2r_tiled_copy_gate.get_slice(idx);
    auto tGsG = s2r_thr_copy_gate.partition_S(sG);
    auto tCrG_view = s2r_thr_copy_gate.retile_D(tCrG);
    // sp s2r
    S2RCopySpX s2r_tiled_copy_spx;
    auto s2r_thr_copy_spx = s2r_tiled_copy_spx.get_slice(idx);
    auto tSXsSX = group_modes<3, 5>(s2r_thr_copy_spx.partition_S(
        local_tile(sX, make_shape(_1{}, Int<kTileK>{}), make_coord(sp_ix%kTileN, _, _))
    ));
    auto tCrSX_view = s2r_thr_copy_spx.retile_D(tCrSX);
    S2RCopySpUp s2r_tiled_copy_spup;
    auto s2r_thr_copy_spup = s2r_tiled_copy_spup.get_slice(idx);
    auto tSUsSU = s2r_thr_copy_spup.partition_S(sSU);
    auto tCrSU_view = s2r_thr_copy_spup.retile_D(tCrSU);
    S2RCopySpGate s2r_tiled_copy_spgate;
    auto s2r_thr_copy_spgate = s2r_tiled_copy_spgate.get_slice(idx);
    auto tSGsSG = s2r_thr_copy_spgate.partition_S(sSG);
    auto tCrSG_view = s2r_thr_copy_spgate.retile_D(tCrSG);

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
#pragma unroll
        for (int cp = 0; cp < g2s_copy_num_thread; ++cp) {
            cute::copy(g2s_tiled_copy_spup, tSUgSU_copy(_, 0, _, col_off[cp], istage),
                       tSUsSU_copy(_, cp, _, istage));
            cute::copy(g2s_tiled_copy_spgate, tSGgSG_copy(_, 0, _, col_off[cp], istage),
                       tSGsSG_copy(_, cp, _, istage));
        }
        cp_async_fence();
    }

    // first s2r read (x & up)
    cp_async_wait<kStage - 2>();
    __syncthreads();
    cute::copy(s2r_tiled_copy_x, tXsX(_, _, 0, ismem_read), tCrX_view(_, _, 0));
    cute::copy(s2r_tiled_copy_up, tUsU(_, _, 0, ismem_read), tCrU_view(_, _, 0));
    cute::copy(s2r_tiled_copy_spx, tSXsSX(_, _, 0, ismem_read), tCrSX_view(_, _, 0));
    cute::copy(s2r_tiled_copy_spup, tSUsSU(_, _, 0, ismem_read), tCrSU_view(_, _, 0));

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
            cute::copy(s2r_tiled_copy_spgate, tSGsSG(_, _, ik, ismem_read), tCrSG_view(_, _, ik));
            // cal up
            cute::gemm(tiled_mma, tCrD, tCrU(_, _, ik), tCrX(_, _, ik), tCrD);
            // cal sp up
#pragma unroll
            for (int spx = 0; spx < sp_cal_n; ++spx) {
#pragma unroll
                for (int spy = 0; spy < sp_cal_k; ++spy) {
                    tCrSR(spx, 0) += tCrSU(2*sp_cal_n*spy+2*spx, 0, ik) * tCrSX(2*spy, 0, ik) +
                                     tCrSU(2*sp_cal_n*spy+2*spx+1, 0, ik) * tCrSX(2*spy+1, 0, ik);
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
            cute::copy(s2r_tiled_copy_spx, tSXsSX(_, _, ik_n, ismem_read), tCrSX_view(_, _, ik_n));
            cute::copy(s2r_tiled_copy_spup, tSUsSU(_, _, ik_n, ismem_read), tCrSU_view(_, _, ik_n));
            if (ik == 0) {
                // submit next g2s read
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, itile_to_read),
                               tXsX_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_up, tUgU_copy(_, _, _, itile_to_read),
                               tUsU_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_gate, tGgG_copy(_, _, _, itile_to_read),
                               tGsG_copy(_, _, _, ismem_write));
#pragma unroll
                    for (int cp = 0; cp < g2s_copy_num_thread; ++cp) {
                        cute::copy(g2s_tiled_copy_spup, 
                                   tSUgSU_copy(_, 0, _, col_off[cp], itile_to_read),
                                   tSUsSU_copy(_, cp, _, ismem_write));
                        cute::copy(g2s_tiled_copy_spgate,
                                   tSGgSG_copy(_, 0, _, col_off[cp], itile_to_read),
                                   tSGsSG_copy(_, cp, _, ismem_write));
                    }
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            // cal gate
            cute::gemm(tiled_mma, tCrE, tCrG(_, _, ik), tCrX(_, _, ik), tCrE);
            // cal sp gate
#pragma unroll
            for (int spx = 0; spx < sp_cal_n; ++spx) {
#pragma unroll
                for (int spy = 0; spy < sp_cal_k; ++spy) {
                    tCrSR(spx, 1) += tCrSG(2*sp_cal_n*spy+2*spx, 0, ik) * tCrSX(2*spy, 0, ik) +
                                     tCrSG(2*sp_cal_n*spy+2*spx+1, 0, ik) * tCrSX(2*spy+1, 0, ik);
                }
            }
        }
    }

    // sp add
#pragma unroll
    for (int sp = 0; sp < sp_cal_n; ++sp) {
        tCrSR(sp, 0) += static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(sp, 0), 1U, 32));
        tCrSR(sp, 0) += static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(sp, 0), 2U, 32));
        tCrSR(sp, 1) += static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(sp, 1), 1U, 32));
        tCrSR(sp, 1) += static_cast<T>(__shfl_xor_sync(0xffffffff, tCrSR(sp, 1), 2U, 32));
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
    if (idx%4 == 0) {
#pragma unroll
        for (int sp = 0; sp < sp_cal_n; ++sp) {
            lhs = tCrSR(sp, 0);
            rhs = tCrSR(sp, 1);
            lhs = (lhs>0&&rhs>0)?(lhs*rhs):0;
            ((T *)val)[nz_st+sp*32+idx/4] = static_cast<T>(lhs);
        }
    }

    // epi
    auto sD = make_tensor(sU(_, _, ismem_read).data(), SmemLayoutResult{});

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
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
        cute::copy(r2s_tiled_copy_r, tCrD_r2sx(_, j), tCsD_r2s(_, 0, 0, j));
    }
    __syncthreads();

    // shm -> global
#pragma unroll
    for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, 0, 0, j), tRgR_s2gx(_, j));
    }
    __syncthreads();
}

void dense_sddmm(const half * x, const half * up, const half * gate, half * result,
                 const long long * row, const long long * col, half * val,
                 int bs, int hid_d, int mid_d, int t_d)
{
    // TODO: choose kernel and block size
    constexpr int kTileM = 64;
    constexpr int kTileN = 16;
    constexpr int kTileK = 64;
    constexpr int maxnnz = 32;
    config::Config<T, kTileM, kTileN, kTileK, maxnnz> cfg;
    dim3 block(128);
    dim3 grid((bs + kTileN - 1) / kTileN, (t_d + kTileM - 1) / kTileM);
    int shm_size = cfg.kShmSize;
    cudaFuncSetAttribute(ds<decltype(cfg)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    ds<decltype(cfg)><<<grid, block, shm_size>>>((const T *)x, (const T *)up, (const T *)gate,
                                                 (T *)result, row, col, (T *)val,
                                                 mid_d, bs, hid_d, t_d);
}
