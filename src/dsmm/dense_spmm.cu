#include "dsmm.h"
#include "config_spmm.h"


using T = cute::half_t;


template<typename Config>
__global__ static void
ds(const T * x, const T * down1, const T * down2, T * r,
   const long long * row, const long long * col, const T * val,
   int m, int n, int k, int p)
{
    /*
     * x: n x p in row-major
     * down1: m x p in row-major
     * down2: (k-p) x m in row-major
     * r: n x m in row-major
     * 
     * For dense part, r1^T = down1 @ x^T
     * For sparse part, r2 = spmm(val, down2)
     * r = r1 + r2
     */

    using namespace cute;
    using X = Underscore;

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;
    int idx = threadIdx.x;
    int warp_id = idx / 32;
    int spix = ix * 16 + iz * 4 + iy % 4;
    int spiy = iy / 4;
    int nz_st = row[spix];

    // init
    using SmemLayoutX = typename Config::SmemLayoutX;
    using SmemLayoutDown = typename Config::SmemLayoutDown;
    using SmemLayoutResult = typename Config::SmemLayoutResult;
    using SmemLayoutSpDown = typename Config::SmemLayoutSpDown;
    using SmemLayoutSpDownCp = typename Config::SmemLayoutSpDownCp;
    using SmemLayoutVal = typename Config::SmemLayoutVal;
    using SmemLayoutCol = typename Config::SmemLayoutCol;
    using SmemLayoutSpResult = typename Config::SmemLayoutSpResult;
    using G2SCopyX = typename Config::G2SCopyX;
    using G2SCopyDown = typename Config::G2SCopyDown;
    using G2SCopySpDown = typename Config::G2SCopySpDown;
    using G2SCopyCol = typename Config::G2SCopyCol;
    using G2SCopyVal = typename Config::G2SCopyVal;
    using TiledMMA = typename Config::MMA;
    using S2RCopyAtomX = typename Config::S2RCopyAtomX;
    using S2RCopyAtomDown = typename Config::S2RCopyAtomDown;
    using S2RCopySpDown = typename Config::S2RCopySpDown;
    using S2RCopyVal = typename Config::S2RCopyVal;
    using R2SCopyAtomR = typename Config::R2SCopyAtomR;
    using R2SCopySpResult = typename Config::R2SCopySpResult;
    using S2GCopyR = typename Config::S2GCopyR;
    using S2GCopySpR = typename Config::S2GCopySpR;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;
    constexpr int maxnnz = Config::maxnnz;
    constexpr int kBlock = Config::kBlock;

    extern __shared__ T shm_data[];
    T * shmx = shm_data;
    T * shmdown = shmx + cute::cosize(SmemLayoutX{});
    T * shmspdown = shmdown + cute::cosize(SmemLayoutDown{});
    T * shmval = shmspdown + cute::cosize(SmemLayoutSpDown{});
    long long * shmcol = (long long *)(shmval + cute::cosize(SmemLayoutVal{}));

    Tensor I = make_tensor(make_gmem_ptr((T *)x), make_shape(n, p), make_stride(p, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr((T *)down1), make_shape(m, p), make_stride(p, Int<1>{}));
    Tensor R = make_tensor(make_gmem_ptr((T *)r), make_shape(m, n, kBlock),
                           make_stride(n, Int<1>{}, m*n));
    Tensor SD = make_tensor(make_gmem_ptr((T *)down2), make_shape(k-p, m),
                            make_stride(m, _1{}));
    Tensor SR = make_tensor(make_gmem_ptr((T *)r+m*n*kBlock), make_shape(n, m),
                            make_stride(m, _1{}));

    Tensor gX = local_tile(I, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gR = local_tile(R, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix, iz));
    Tensor gSD = local_tile(SD, make_tile(_1{}, Int<kTileM*4>{}), make_coord(_, spiy));
    Tensor gSR = local_tile(SR, make_tile(_1{}, Int<kTileM*4>{}), make_coord(spix, spiy));
    Tensor gC = make_tensor(make_gmem_ptr((long long *)col+nz_st), make_shape(Int<maxnnz>{}));
    Tensor gV = make_tensor(make_gmem_ptr((T *)val+nz_st), make_shape(Int<maxnnz>{}));

    auto sX = make_tensor(make_smem_ptr(shmx), SmemLayoutX{});
    auto sD = make_tensor(make_smem_ptr(shmdown), SmemLayoutDown{});
    auto sSD_copy = make_tensor(make_smem_ptr(shmspdown), SmemLayoutSpDownCp{});
    auto sSD = make_tensor(make_smem_ptr(shmspdown), SmemLayoutSpDown{});
    auto sV = make_tensor(make_smem_ptr(shmval), SmemLayoutVal{});
    auto sC = make_tensor(make_smem_ptr(shmcol), SmemLayoutCol{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrD = thr_mma.partition_fragment_A(gD(_, _, 0));
    auto tCrX = thr_mma.partition_fragment_B(gX(_, _, 0));
    auto tCrR = thr_mma.partition_fragment_C(gR);
    clear(tCrR);
    T rSD[16];
    T rSR[2];
    T rV[8];
    auto tCrSD = make_tensor(make_rmem_ptr(rSD), make_layout(make_shape(_4{}, _1{}, _4{}),
                                                             make_stride(_1{}, _0{}, _4{})));
    auto tCrSR = make_tensor(make_rmem_ptr(rSR), make_layout(make_shape(_2{})));
    auto tCrV = make_tensor(make_rmem_ptr(rV), make_layout(make_shape(_2{}, _1{}, _4{}),
                                                           make_stride(_1{}, _0{}, _2{})));
    clear(tCrSR);

    // g2s setting
    G2SCopyX g2s_tiled_copy_x;
    auto g2s_thr_copy_x = g2s_tiled_copy_x.get_slice(idx);
    auto tXgX_copy = g2s_thr_copy_x.partition_S(gX);
    auto tXsX_copy = g2s_thr_copy_x.partition_D(sX);
    G2SCopyDown g2s_tiled_copy_down;
    auto g2s_thr_copy_down = g2s_tiled_copy_down.get_slice(idx);
    auto tDgD_copy = g2s_thr_copy_down.partition_S(gD);
    auto tDsD_copy = g2s_thr_copy_down.partition_D(sD);
    G2SCopySpDown g2s_tiled_copy_spdown;
    auto g2s_thr_copy_spdown = g2s_tiled_copy_spdown.get_slice(idx);
    auto tSDgSD_copy = g2s_thr_copy_spdown.partition_S(gSD);
    auto tSDsSD_copy = g2s_thr_copy_spdown.partition_D(sSD_copy);
    G2SCopyCol g2s_tiled_copy_c;
    auto g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(idx);
    auto tCsC_copy = g2s_thr_copy_c.partition_D(sC);
    auto tCgC_copy = local_tile(gC, make_shape(_2{}), make_coord(_));
    auto tCgC_copy1 = tCgC_copy(_, idx);
    auto tCgC_copy2 = tCgC_copy(_, 0);
    if (idx*2 < maxnnz-256) {
        tCgC_copy2 = tCgC_copy(_, idx+128);
    }
    G2SCopyVal g2s_tiled_copy_v;
    auto g2s_thr_copy_v = g2s_tiled_copy_v.get_slice(idx);
    auto tVgV_copy = g2s_thr_copy_v.partition_S(gV);
    auto tVsV_copy = g2s_thr_copy_v.partition_D(sV);
    if (idx*8 >= maxnnz) {
        tVgV_copy = make_tensor(
            gV.data(), make_shape(make_shape(_8{}, _1{}), make_shape(_1{})),
            make_stride(make_stride(_1{}, _0{}), make_stride(_0{}))
        );
    }

    // s2r setting
    auto s2r_tiled_copy_down = make_tiled_copy_A(S2RCopyAtomDown{}, tiled_mma);
    auto s2r_thr_copy_down = s2r_tiled_copy_down.get_slice(idx);
    auto tDsD = s2r_thr_copy_down.partition_S(sD);
    auto tCrD_view = s2r_thr_copy_down.retile_D(tCrD);
    auto s2r_tiled_copy_x = make_tiled_copy_B(S2RCopyAtomX{}, tiled_mma);
    auto s2r_thr_copy_x = s2r_tiled_copy_x.get_slice(idx);
    auto tXsX = s2r_thr_copy_x.partition_S(sX);
    auto tCrX_view = s2r_thr_copy_x.retile_D(tCrX);
    S2RCopySpDown s2r_tiled_copy_spdown;
    auto s2r_thr_copy_spdown = s2r_tiled_copy_spdown.get_slice(idx);
    auto tSDsSD = s2r_thr_copy_spdown.partition_S(sSD);
    auto tCrSD_view = s2r_thr_copy_spdown.retile_D(tCrSD);
    S2RCopyVal s2r_tiled_copy_val;

    // read all val and col to smem
    cute::copy(g2s_tiled_copy_c, tCgC_copy1, tCsC_copy(_, 0));
    cute::copy(g2s_tiled_copy_c, tCgC_copy2, tCsC_copy(_, 1));
    cute::copy(g2s_tiled_copy_v, tVgV_copy, tVsV_copy);
    cp_async_fence();

    auto tVsV = local_tile(local_tile(sV, make_shape(_8{}), make_coord(_)),
                           make_shape(_2{}, _1{}), make_coord(_, _));
    int bfpt = 1;
    long long scbf[4];
    cp_async_wait<0>();
    __syncthreads();
    scbf[0] = sC(warp_id);
    scbf[2] = sC(warp_id+4);
    __syncthreads();

    // stage
    int itile_to_read = kStage - 1;  // first tile read that has not been submitted 
    int ismem_read = 0;              // smem that calculation is working on
    int ismem_write = kStage - 1;    // smem write that has not been submitted
    int val_read = kStage - 1;       // for val read

    // submit kStage - 1 g2s read
#pragma unroll
    for (int istage = 0; istage < kStage-1; ++istage) {
        scbf[bfpt] = sC(warp_id+istage*8+8);
        scbf[bfpt+2] = sC(warp_id+istage*8+12);
        bfpt = 1-bfpt;
        cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, kBlock*istage+iz),
                   tXsX_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_down, tDgD_copy(_, _, _, kBlock*istage+iz),
                   tDsD_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_spdown, tSDgSD_copy(_, 0, 0, scbf[bfpt]),
                   tSDsSD_copy(_, 0, 0, istage));
        cute::copy(g2s_tiled_copy_spdown, tSDgSD_copy(_, 0, 0, scbf[bfpt+2]),
                   tSDsSD_copy(_, 1, 0, istage));
        cp_async_fence();
    }

    // first s2r read
    cp_async_wait<kStage - 2>();
    __syncthreads();
    cute::copy(s2r_tiled_copy_x, tXsX(_, _, 0, ismem_read), tCrX_view(_, _, 0));
    cute::copy(s2r_tiled_copy_down, tDsD(_, _, 0, ismem_read), tCrD_view(_, _, 0));
    cute::copy(s2r_tiled_copy_spdown, tSDsSD(_, 0, _, ismem_read), tCrSD_view(_, _, 0));
    cute::copy(s2r_tiled_copy_val, tVsV(_, _, 0, ismem_read), tCrV(_, _, 0));

    // loop over k
    int ntile = p / kTileK / kBlock;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        // loop in kTileK
        int nk = size<2>(tCrD);
#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_n = (ik + 1) % nk;
            if (ik == nk - 1) {
                // wait next g2s read done
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
                val_read = (val_read + 1) % ntile;
            }
            cute::copy(s2r_tiled_copy_x, tXsX(_, _, ik_n, ismem_read), tCrX_view(_, _, ik_n));
            cute::copy(s2r_tiled_copy_down, tDsD(_, _, ik_n, ismem_read), tCrD_view(_, _, ik_n));
            cute::copy(s2r_tiled_copy_spdown, tSDsSD(_, ik_n, _, ismem_read),
                       tCrSD_view(_, _, ik_n));
            cute::copy(s2r_tiled_copy_val, tVsV(_, _, ik_n, val_read), tCrV(_, _, ik_n));
            if (ik == 0) {
                if (itile_to_read < ntile-1) {
                    scbf[bfpt] = sC(warp_id+itile_to_read*8+8);
                    scbf[bfpt+2] = sC(warp_id+itile_to_read*8+12);
                }
                bfpt = 1-bfpt;
                // submit next g2s read
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, kBlock*itile_to_read+iz),
                               tXsX_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_down, tDgD_copy(_, _, _, kBlock*itile_to_read+iz),
                               tDsD_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_spdown, tSDgSD_copy(_, 0, 0, scbf[bfpt]),
                               tSDsSD_copy(_, 0, 0, ismem_write));
                    cute::copy(g2s_tiled_copy_spdown, tSDgSD_copy(_, 0, 0, scbf[bfpt+2]),
                               tSDsSD_copy(_, 1, 0, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            cute::gemm(tiled_mma, tCrR, tCrD(_, _, ik), tCrX(_, _, ik), tCrR);
            tCrSR(0) += tCrSD(0, 0, ik) * tCrV(0, 0, ik) + tCrSD(2, 0, ik) * tCrV(1, 0, ik);
            tCrSR(1) += tCrSD(1, 0, ik) * tCrV(0, 0, ik) + tCrSD(3, 0, ik) * tCrV(1, 0, ik);
        }
    }

    // epi
    auto sR = make_tensor(sD(_, _, ismem_read).data(), SmemLayoutResult{});
    auto sSR = make_tensor(sSD(_, _, ismem_read).data(), SmemLayoutSpResult{});

    auto r2s_tiled_copy_r = make_tiled_copy_C(R2SCopyAtomR{}, tiled_mma);
    auto r2s_thr_copy_r = r2s_tiled_copy_r.get_slice(idx);
    auto tCrR_r2s = r2s_thr_copy_r.retile_S(tCrR);
    auto tCsR_r2s = r2s_thr_copy_r.partition_D(sR);
    auto tCrR_r2sx = group_modes<1, 3>(tCrR_r2s);
    R2SCopySpResult r2s_tiled_copy_sr;
    auto tSRsSR_r2s = local_tile(sSR, make_shape(_2{}), make_coord(_));

    S2GCopyR s2g_tiled_copy_r;
    auto s2g_thr_copy_r = s2g_tiled_copy_r.get_thread_slice(idx);
    auto tRsR_s2g = s2g_thr_copy_r.partition_S(sR);
    auto tRgR_s2g = s2g_thr_copy_r.partition_D(gR);
    auto tRgR_s2gx = group_modes<1, 3>(tRgR_s2g);
    S2GCopySpR s2g_tiled_copy_sr;
    auto tSRsSR_s2g = local_tile(sSR, make_shape(_8{}), make_coord(_));
    auto tSRgSR_s2g = local_tile(gSR(0, _), make_shape(_8{}), make_coord(_));

    cute::copy(r2s_tiled_copy_r, tCrR_r2sx(_, 0), tCsR_r2s(_, 0, 0, 0));
    cute::copy(r2s_tiled_copy_sr, tCrSR, flatten(tSRsSR_r2s(_, idx%4+warp_id*4+((idx/4)%8)*16)));
    __syncthreads();
    cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, 0, 0, 0), tRgR_s2gx(_, 0));
    if (warp_id == 0) {
        cute::copy(s2g_tiled_copy_sr, tSRsSR_s2g(_, idx), tRgR_s2gx(_, idx));
    }
    __syncthreads();
}

void dense_spmm(const half * x, const half * down1, const half * down2, half * result,
                const long long * row, const long long * col, const half * val,
                int bs, int hid_d, int mid_d, int t_d)
{
    // TODO: choose kernel and block size
    constexpr int kTileM = 64;
    constexpr int kTileN = 16;
    constexpr int kTileK = 64;
    constexpr int kBlock = 4;
    config::Config<T, kTileM, kTileN, kTileK, kBlock> cfg;
    dim3 block(128);
    dim3 grid((bs + kTileN - 1) / kTileN, (hid_d + kTileM - 1) / kTileM, kBlock);
    int shm_size = cfg.kShmSize;
    cudaFuncSetAttribute(ds<decltype(cfg)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    ds<decltype(cfg)><<<grid, block, shm_size>>>((const T *)x, (const T *)down1, (const T *)down2,
                                                 (T *)result, row, col, (const T *)val,
                                                 hid_d, bs, mid_d, t_d);
}
