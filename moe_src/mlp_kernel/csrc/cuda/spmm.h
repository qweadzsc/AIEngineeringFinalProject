#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace config {

using namespace cute;

template <typename T_, int kTileM_=64, int kTileN_=128, int kTileK_=32, int kTileS_=8,
          int maxnnz_=16, int kStage_=3>
struct Config {
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kTileS = kTileS_;
    static constexpr int maxnnz = maxnnz_;
    static constexpr int kStage = kStage_;

    static constexpr int sp_per_ds = kTileN / kTileS;
    static constexpr int repeat_x = kTileS / 128;
    static constexpr int repeat_y = maxnnz;

    // static_assert(kTileK == 32, "only support tileK=32");
    static_assert(kTileS >= 128, "too small tileS");

    // shm config
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<512/kTileK>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, _1{}))
    ));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutDown = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    // using SmemLayoutSpX = decltype(tile_to_shape(
    //     composition(
    //         Swizzle<2, 3, 3>{},
    //         make_layout(make_shape(Int<256/kTileK>{}, Int<kTileK>{}),
    //                     make_stride(Int<kTileK>{}, _1{}))
    //     ), make_shape(Int<maxnnz>{}, Int<kTileK>{}, Int<kStage>{})
    // ));
    using SmemLayoutSpX = decltype(make_layout(
        make_shape(Int<maxnnz>{}, Int<kTileK>{}, Int<kStage>{}),
        make_stride(Int<kTileK>{}, _1{}, Int<maxnnz*kTileK>{})));
    using SmemLayoutSpDown = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileS>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    // g2s config
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyX = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<1024/kTileK>{}, Int<kTileK/8>{}),
                    make_stride(Int<kTileK/8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyDown = G2SCopyX;
    using G2SCopyV = G2SCopyX;
    using G2SCopySpDown = G2SCopyX;

    // s2r config
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_atom = Copy_Atom<Copy_Traits<s2r_copy_op>, T>;

    using S2RCopyAtomX = s2r_copy_atom;
    using S2RCopyAtomDown = s2r_copy_atom;
    using S2RCopySpAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;

    // cal config
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(
        mma_atom{}, make_layout(make_shape(_2{}, _2{}, _1{})),
        Tile<_32, _32, _16>{}
    ));

    // r2s config
    static constexpr int kSmemLayoutCBatch = 2;
    using SmemLayoutAtomResult = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))
    ));
    using SmemLayoutResult = decltype(tile_to_shape(
        SmemLayoutAtomResult{},
        make_shape(Int<kTileM>{}, Int<32>{}, Int<kSmemLayoutCBatch>{})
    ));
    using R2SCopyAtomR = Copy_Atom<UniversalCopy<int>, T>;

    // s2g config
    using S2GCopyAtomR = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyR = decltype(make_tiled_copy(
        S2GCopyAtomR{},
        make_layout(make_shape(Int<32>{}, Int<4>{}),
                    make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // other config
    static constexpr int shm_size = cute::cosize(SmemLayoutX{}) +
                                    cute::cosize(SmemLayoutDown{}) +
                                    cute::cosize(SmemLayoutSpX{}) +
                                    cute::cosize(SmemLayoutSpDown{}) +
                                    kTileM * 2 +  // for weight
                                    maxnnz * 4;  // for nnz
    static constexpr int kShmSize = shm_size * sizeof(T);
};

}
