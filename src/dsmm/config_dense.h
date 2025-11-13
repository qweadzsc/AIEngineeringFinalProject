#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace config {

using namespace cute;

template <typename T_, int kTileM_=64, int kTileN_=16, int kTileK_=64, int kStage_=3>
struct Config {
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kStage = kStage_;

    // shm config
    using SmemLayoutAtom = decltype(composition(
        Swizzle<_3{}, _3{}, _3{}>{},
        make_layout(make_shape(Int<512/kTileK>{}, Int<kTileK>{}),
        make_stride(Int<kTileK>{}, _1{}))
    ));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutUp = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
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
    using G2SCopyUp = G2SCopyX;

    // s2r config
    using s2r_copy_op_4 = SM75_U32x4_LDSM_N;
    using s2r_copy_atom_4 = Copy_Atom<Copy_Traits<s2r_copy_op_4>, T>;

    using S2RCopyAtomX = s2r_copy_atom_4;
    using S2RCopyAtomUp = s2r_copy_atom_4;

    // cal config
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // if constexpr (kTileN <= 8) {
    //     using MMA = decltype(make_tiled_mma(
    //         mma_atom{}, make_layout(make_shape(_4{}, _1{}, _1{})),
    //         Tile<_64{}, _8{}, _32{}>{}
    //     ));
    // } else if constexpr (kTileN == 16) {
        using MMA = decltype(make_tiled_mma(
            mma_atom{}, make_layout(make_shape(_2{}, _2{}, _1{})),
            Tile<_32, _32, _16>{}
        ));
    // } else {
    //     using MMA = decltype(make_tiled_mma(
    //         mma_atom{}, make_layout(make_shape(_2{}, _2{}, _1{})),
    //         Tile<_32{}, _32{}, _16{}>{}
    //     ));
    // }

    // r2s config
    static constexpr int kSmemLayoutCBatch = 2;
    using SmemLayoutAtomResult = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<512/kTileN>{}, Int<kTileN>{}),
                    make_stride(Int<kTileN>{}, Int<1>{}))
    ));
    using SmemLayoutResult = decltype(tile_to_shape(
        SmemLayoutAtomResult{},
        make_shape(Int<1024/kTileN>{}, Int<kTileN>{}, Int<kSmemLayoutCBatch>{})
    ));
    using R2SCopyAtomR = Copy_Atom<UniversalCopy<int>, T>;

    // s2g config
    using S2GCopyAtomR = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyR = decltype(make_tiled_copy(
        S2GCopyAtomR{},
        make_layout(make_shape(Int<1024/kTileN>{}, Int<kTileN/8>{}),
                    make_stride(Int<kTileN/8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // other config
    static constexpr int shm_size = cute::cosize(SmemLayoutX{}) +
                                    cute::cosize(SmemLayoutUp{});
    static constexpr int kShmSize = shm_size * sizeof(T);
};

}