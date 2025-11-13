#include "dsmm.h"
#include "spmm.h"


using T = cute::half_t;


template<typename Config>
__global__ static void
spmm(const T * x, const T * down, T * r,
     const long long * row, const long long * col, T * val, const T * w,
     int m, int n, int k, int e, int p)
{
    /*
     * e: inter_size of one expert
     * p: num of ds experts
     */

    using namespace cute;
    using X = Underscore;

    int ix = blockIdx.x;  // pos in expert, 0-(block_per_expert-1)
    int iy = blockIdx.y;  // 0
    int iz = blockIdx.z;  // expert id
    int idx = threadIdx.x;
    int debug = (ix + 16*iz) * 128 + idx;

    // init
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kTileS = Config::kTileS;
    constexpr int maxnnz = Config::maxnnz;
    constexpr int kStage = Config::kStage;

    // constexpr int block_per_expert = Config::block_per_expert;
    constexpr int sp_per_ds = Config::sp_per_ds;
    constexpr int repeat_x = Config::repeat_x;
    constexpr int repeat_y = Config::repeat_y;
    int six = iz % sp_per_ds + ix * sp_per_ds;
    int siz = iz / sp_per_ds;

    int dsk = p * e;
    int spw = dsk / sp_per_ds;
    int num_experts = k / e;
    int nt = gridDim.y;

    using SmemLayoutX = typename Config::SmemLayoutX;
    using SmemLayoutDown = typename Config::SmemLayoutDown;
    using SmemLayoutSpX = typename Config::SmemLayoutSpX;
    using SmemLayoutSpDown = typename Config::SmemLayoutSpDown;
    using SmemLayoutResult = typename Config::SmemLayoutResult;
    using G2SCopyX = typename Config::G2SCopyX;
    using G2SCopyDown = typename Config::G2SCopyDown;
    using G2SCopyV = typename Config::G2SCopyV;
    using G2SCopySpDown = typename Config::G2SCopySpDown;
    using S2RCopyAtomX = typename Config::S2RCopyAtomX;
    using S2RCopyAtomDown = typename Config::S2RCopyAtomDown;
    using S2RCopySpAtom = typename Config::S2RCopySpAtom;
    using TiledMMA = typename Config::MMA;
    using R2SCopyAtomR = typename Config::R2SCopyAtomR;
    using S2GCopyR = typename Config::S2GCopyR;

    extern __shared__ T shm_data[];
    T * shmx = shm_data;
    T * shmdown = shmx + cute::cosize(SmemLayoutX{});
    T * shmspx = shmdown + cute::cosize(SmemLayoutDown{});
    T * shmspdown = shmspx + cute::cosize(SmemLayoutSpX{});
    T * shmw = shmspdown + cute::cosize(SmemLayoutSpDown{});
    long long * shmnzp = (long long *)(shmw + 2*kTileM);

    Tensor I = make_tensor(make_gmem_ptr(x), make_shape(m, e, p), make_stride(dsk, Int<1>{}, e));
    Tensor D = make_tensor(make_gmem_ptr(down), make_shape(n, e, num_experts),
                           make_stride(k, Int<1>{}, e));
    Tensor R = make_tensor(make_gmem_ptr(r+iz*m*n), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor V = make_tensor(make_gmem_ptr(val), make_shape(Int<maxnnz>{}, e, p/sp_per_ds, nt),
                           make_stride(spw, Int<1>{}, e, maxnnz*spw));
    Tensor W = make_tensor(make_gmem_ptr(w), make_shape(Int<kTileM>{}, num_experts, nt),
                           make_stride(num_experts, _1{}, kTileM*num_experts));
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

    Tensor gX = local_tile(I, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _, iz));
    Tensor gD = local_tile(D, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _, real_iz));
    Tensor gR = local_tile(R, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    Tensor gV = local_tile(V, make_tile(Int<maxnnz>{}, Int<kTileK>{}), make_coord(iy, _, siz, iy));
    Tensor gSD = local_tile(D, make_tile(Int<kTileS>{}, Int<kTileK>{}),
                            make_coord(six, _, real_siz));

    auto sX = make_tensor(make_smem_ptr(shmx), SmemLayoutX{});
    auto sD = make_tensor(make_smem_ptr(shmdown), SmemLayoutDown{});
    auto sV = make_tensor(make_smem_ptr(shmspx), SmemLayoutSpX{});
    auto sSD = make_tensor(make_smem_ptr(shmspdown), SmemLayoutSpDown{});
    auto sW = make_tensor(make_smem_ptr(shmw), make_shape(_2{}, Int<kTileM>{}),
                          make_stride(Int<kTileM>{}, _1{}));

    // weight load
    if (idx < kTileM) {
        sW(0, idx) = W(idx, real_iz, iy);
        sW(1, idx) = W(idx, real_siz, iy);
    }

    array<T, 16*repeat_y*4> v_data;
    array<T, 16*repeat_x*4> spdown_data;
    array<T, repeat_x*repeat_y> spr_data;
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrX = thr_mma.partition_fragment_A(gX(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_B(gD(_, _, 0));
    auto tCrR = thr_mma.partition_fragment_C(gR);
    auto tCrV = make_tensor(make_rmem_ptr(v_data.data()),
        make_shape(Int<repeat_y>{}, _16{}, _4{}), make_stride(_16{}, _1{}, Int<16*repeat_y>{}));
    auto tCrSD = make_tensor(make_rmem_ptr(spdown_data.data()),
        make_shape(Int<repeat_x>{}, _16{}, _4{}), make_stride(_16{}, _1{}, Int<16*repeat_x>{}));
    auto tCrSR = make_tensor(make_rmem_ptr(spr_data.data()),
        make_shape(Int<repeat_y>{}, Int<repeat_x>{}), make_stride(Int<repeat_x>{}, _1{}));
    clear(tCrR);
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
    // G2SCopyV g2s_tiled_copy_v;
    // auto g2s_thr_copy_v = g2s_tiled_copy_v.get_slice(idx);
    // auto tVgV_copy = g2s_thr_copy_v.partition_S(gV);
    // auto tVsV_copy = g2s_thr_copy_v.partition_D(sV);
    G2SCopySpDown g2s_tiled_copy_spdown;
    auto g2s_thr_copy_spdown = g2s_tiled_copy_spdown.get_slice(idx);
    auto tSDgSD_copy = g2s_thr_copy_spdown.partition_S(gSD);
    auto tSDsSD_copy = g2s_thr_copy_spdown.partition_D(sSD);

    // s2r setting
    auto s2r_tiled_copy_x = make_tiled_copy_A(S2RCopyAtomX{}, tiled_mma);
    auto s2r_thr_copy_x = s2r_tiled_copy_x.get_slice(idx);
    auto tXsX = s2r_thr_copy_x.partition_S(sX);
    auto tCrX_view = s2r_thr_copy_x.retile_D(tCrX);
    auto s2r_tiled_copy_down = make_tiled_copy_B(S2RCopyAtomDown{}, tiled_mma);
    auto s2r_thr_copy_down = s2r_tiled_copy_down.get_slice(idx);
    auto tDsD = s2r_thr_copy_down.partition_S(sD);
    auto tCrD_view = s2r_thr_copy_down.retile_D(tCrD);
    // sp s2r
    S2RCopySpAtom s2r_tiled_copy_sp_atom;
    auto tVsV = local_tile(sV, make_tile(_1{}, _8{}), make_coord(_, _))(0, _, _, _, _);
    auto tSDsSD = local_tile(sSD, make_tile(_128{}, _8{}), make_coord(0, _))(idx, _, _, _);
    auto tVrV_s2r = local_tile(
        tCrV, make_tile(Int<repeat_y>{}, _8{}), make_coord(0, _))(_, _, _, _);
    auto tSDrSD_s2r = local_tile(
        tCrSD, make_tile(Int<repeat_x>{}, _8{}), make_coord(0, _))(_, _, _, _);

    // stage
    int itile_to_read = kStage - 1;  // first tile read that has not been submitted 
    int ismem_read = 0;              // smem that calculation is working on
    int ismem_write = kStage - 1;    // smem write that has not been submitted

    // submit kStage - 1 g2s read
#pragma unroll
    for (int istage = 0; istage < kStage-1; ++istage) {
        cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, istage), tXsX_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_down, tDgD_copy(_, _, _, istage), tDsD_copy(_, _, _, istage));
        // cute::copy(g2s_tiled_copy_v, tVgV_copy(_, _, _, istage), tVsV_copy(_, _, _, istage));
        if (idx < kTileK) {
#pragma unroll
            for (int i = 0; i < maxnnz; ++i) {
                sV(i, idx, istage) = gV(i, idx, istage);
            }
        }
        cute::copy(g2s_tiled_copy_spdown, tSDgSD_copy(_, _, _, istage),
                   tSDsSD_copy(_, _, _, istage));
        cp_async_fence();
    }

    // first s2r read
    cp_async_wait<kStage - 2>();
    __syncthreads();
    cute::copy(s2r_tiled_copy_x, tXsX(_, _, 0, ismem_read), tCrX_view(_, _, 0));
    cute::copy(s2r_tiled_copy_down, tDsD(_, _, 0, ismem_read), tCrD_view(_, _, 0));
#pragma unroll
    for (int spy = 0; spy < repeat_y; ++spy) {
        cute::copy(s2r_tiled_copy_sp_atom, tVsV(_, spy, 0, ismem_read), tVrV_s2r(spy, _, 0, 0));
        cute::copy(s2r_tiled_copy_sp_atom, tVsV(_, spy, 1, ismem_read), tVrV_s2r(spy, _, 1, 0));
    }
#pragma unroll
    for (int spx = 0; spx < repeat_x; ++spx) {
        cute::copy(s2r_tiled_copy_sp_atom, tSDsSD(_, 0, ismem_read),
                   tSDrSD_s2r(spx, _, 0, 0));
        cute::copy(s2r_tiled_copy_sp_atom, tSDsSD(_, 1, ismem_read),
                   tSDrSD_s2r(spx, _, 1, 0));
    }

    // loop over k
    int ntile = e / kTileK;
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
            }
            cute::copy(s2r_tiled_copy_x, tXsX(_, _, ik_n, ismem_read), tCrX_view(_, _, ik_n));
            cute::copy(s2r_tiled_copy_down, tDsD(_, _, ik_n, ismem_read), tCrD_view(_, _, ik_n));
#pragma unroll
            for (int spy = 0; spy < repeat_y; ++spy) {
                cute::copy(s2r_tiled_copy_sp_atom, tVsV(_, spy, 2*ik_n+0, ismem_read),
                           tVrV_s2r(spy, _, 0, ik_n));
                cute::copy(s2r_tiled_copy_sp_atom, tVsV(_, spy, 2*ik_n+1, ismem_read),
                           tVrV_s2r(spy, _, 1, ik_n));
            }
#pragma unroll
            for (int spx = 0; spx < repeat_x; ++spx) {
                cute::copy(s2r_tiled_copy_sp_atom, tSDsSD(_, 2*ik_n+0, ismem_read),
                           tSDrSD_s2r(spx, _, 0, ik_n));
                cute::copy(s2r_tiled_copy_sp_atom, tSDsSD(_, 2*ik_n+1, ismem_read),
                           tSDrSD_s2r(spx, _, 1, ik_n));
            }
            if (ik == 0) {
                // submit next g2s read
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_x, tXgX_copy(_, _, _, itile_to_read),
                               tXsX_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_down, tDgD_copy(_, _, _, itile_to_read),
                               tDsD_copy(_, _, _, ismem_write));
                    // cute::copy(g2s_tiled_copy_v, tVgV_copy(_, _, _, itile_to_read),
                    //            tVsV_copy(_, _, _, ismem_write));
                    if (idx < kTileK) {
#pragma unroll
                        for (int i = 0; i < maxnnz; ++i) {
                            sV(i, idx, ismem_write) = gV(i, idx, itile_to_read);
                        }
                    }
                    cute::copy(g2s_tiled_copy_spdown, tSDgSD_copy(_, _, _, itile_to_read),
                               tSDsSD_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            cute::gemm(tiled_mma, tCrR, tCrX(_, _, ik), tCrD(_, _, ik), tCrR);
#pragma unroll
            for (int spx = 0; spx < repeat_x; ++spx) {
#pragma unroll
                for (int spy = 0; spy < repeat_y; ++spy) {
#pragma unroll
                    for (int spz = 0; spz < 16; ++spz) {
                        tCrSR(spy, spx) += tCrV(spy, spz, ik) * tCrSD(spx, spz, ik);
                    }
                }
            }
        }
    }

    // weight  TODO: FIXME
#pragma unroll
    for (int dx = 0; dx < 4; ++dx) {
#pragma unroll
        for (int dy = 0; dy < size<1>(tCrR); ++dy) {
            T weight_xy = sW(0, (idx%32)/4+(dx/2)*8+((idx/32)%2)*16+dy*32);
#pragma unroll
            for (int dz = 0; dz < size<2>(tCrR); ++dz) {
                tCrR(dx, dy, dz) *= weight_xy;
            }
        }
    }
#pragma unroll
    for (int spy = 0; spy < repeat_y; ++spy) {
        T weight_y = sW(1, nz_rows(spy));
#pragma unroll
        for (int spx = 0; spx < repeat_x; ++spx) {
            tCrSR(spy, spx) *= weight_y;
        }
    }

    // epi
    auto sR = make_tensor(sD(_, _, ismem_read).data(), SmemLayoutResult{});

    auto r2s_tiled_copy_r = make_tiled_copy_C(R2SCopyAtomR{}, tiled_mma);
    auto r2s_thr_copy_r = r2s_tiled_copy_r.get_slice(idx);
    auto tCrR_r2s = r2s_thr_copy_r.retile_S(tCrR);
    auto tCsR_r2s = r2s_thr_copy_r.partition_D(sR);

    S2GCopyR s2g_tiled_copy_r;
    auto s2g_thr_copy_r = s2g_tiled_copy_r.get_thread_slice(idx);
    auto tRsR_s2g = s2g_thr_copy_r.partition_S(sR);
    auto tRgR_s2g = s2g_thr_copy_r.partition_D(gR);

    int step = size<3>(tCsR_r2s);  // pipe
    int m_ti = size<1>(tCrR_r2s);
#pragma unroll
    for (int i = 0; i < size<2>(tCrR_r2s); i += step) {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j) {
#pragma unroll
            for (int m_i = 0; m_i < m_ti; ++m_i) {
                auto t = make_tensor_like<T>(tCrR_r2s(_, m_i, i + j));
                cute::copy(tCrR_r2s(_, m_i, i + j), t);
                cute::copy(r2s_tiled_copy_r, t, tCsR_r2s(_, m_i, 0, j));
            }
        }
        __syncthreads();

        // add sp
        // assume that repeat_x=1
        int spx = idx / 32;
        int j = (iz%sp_per_ds)*repeat_x+spx - i;
        if (j == 0 || j == 1) {
#pragma unroll
            for (int spy = 0; spy < repeat_y; ++spy) {
                sR(nz_rows(spy), idx%32, j) += tCrSR(spy, 0);
            }
        }
        __syncthreads();

        // shm -> global
#pragma unroll
        for (int j = 0; j < step; ++j) {
#pragma unroll
            for (int m_i = 0; m_i < m_ti; ++m_i) {
                cute::copy(s2g_tiled_copy_r, tRsR_s2g(_, m_i, 0, j), tRgR_s2g(_, m_i, i + j));
            }
        }
        __syncthreads();
    }
}

template <int kTileM_, int maxnnz_>
static void spmm_var_m(const half * x, const half * down, half * r,
              const long long * row, const long long * col, const half * val, const half * w,
              int m, int n, int k, int e, int p)
{
    constexpr int kTileM = kTileM_;
    constexpr int kTileN = 128;
    constexpr int kTileK = 64;
    constexpr int kTileS = 128;
    constexpr int maxnnz = maxnnz_;
    config::Config<T, kTileM, kTileN, kTileK, kTileS, maxnnz> cfg;
    dim3 block(128);
    dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM, p);
    int shm_size = cfg.kShmSize;
    cudaFuncSetAttribute(spmm<decltype(cfg)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    spmm<decltype(cfg)><<<grid, block, shm_size>>>(
        (const T *)x, (const T *)down, (T *)r, row, col, (T *)val, (const T *)w, m, n, k, e, p);
    CUDA_CHECK(cudaGetLastError());
}


#define SPMM_DISPATCH(M_VAL, MN_VAL) \
    spmm_var_m<M_VAL, MN_VAL>(x, down, r, row, col, val, w, m, n, k, e, p)

template<int M>
void dispatch_mn(const half * x, const half * down, half * r,
              const long long * row, const long long * col, const half * val, const half * w,
              int m, int n, int k, int e, int p, int mn)
{
    switch (mn) {
        case 1: SPMM_DISPATCH(M, 1); break;
        case 2: SPMM_DISPATCH(M, 2); break;
        case 3: SPMM_DISPATCH(M, 3); break;
        case 4: SPMM_DISPATCH(M, 4); break;
        case 5: SPMM_DISPATCH(M, 5); break;
        case 6: SPMM_DISPATCH(M, 6); break;
        case 7: SPMM_DISPATCH(M, 7); break;
        case 8: SPMM_DISPATCH(M, 8); break;
        default: throw std::invalid_argument("MN must be between 1 and 8");
    }
}

#define SPMM_DISPATCH_MN(M_VAL) \
    dispatch_mn<M_VAL>(x, down, r, row, col, val, w, m, n, k, e, p, mn)

void spmm_api(const half * x, const half * down, half * r,
              const long long * row, const long long * col, const half * val, const half * w,
              int m, int n, int k, int e, int p, int mn)
{
    switch (m) {
        case 32:  SPMM_DISPATCH_MN(32); break;
        case 64:  SPMM_DISPATCH_MN(64); break;
        case 128: SPMM_DISPATCH_MN(128); break;
        default: throw std::invalid_argument("M must be 32, 64, or 128");
    }
}

#undef SPMM_DISPATCH
#undef SPMM_DISPATCH_MN
