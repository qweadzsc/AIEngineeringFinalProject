#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace config {

using namespace cute;

template <typename T_, int kTileM_=32, int kTileN_=32, int kTileK_=64,
          int kTileS_=80, int kStage_=3>
struct Config {
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kTileS = kTileS_;
    static constexpr int kStage = kStage_;

    static constexpr int maxnnz_pt = 4;
    static constexpr int cal_time = kTileK / 16;

    // shm config
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<512/kTileK>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, _1{}))
    ));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutUp = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutGate = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutSpUp = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileS>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutSpGate = decltype(tile_to_shape(
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
    using G2SCopyUp = G2SCopyX;
    using G2SCopyGate = G2SCopyX;
    using G2SCopySpUp = G2SCopyX;
    using G2SCopySpGate = G2SCopyX;

    // s2r config
    // using s2r_copy_op_1 = SM75_U32x1_LDSM_N;
    // using s2r_copy_atom_1 = Copy_Atom<Copy_Traits<s2r_copy_op_1>, T>;
    using s2r_copy_op_2 = SM75_U32x2_LDSM_N;
    using s2r_copy_atom_2 = Copy_Atom<Copy_Traits<s2r_copy_op_2>, T>;
    using s2r_copy_op_4 = SM75_U32x4_LDSM_N;
    using s2r_copy_atom_4 = Copy_Atom<Copy_Traits<s2r_copy_op_4>, T>;

    using S2RCopyAtomX = s2r_copy_atom_4;
    using S2RCopyAtomUp = s2r_copy_atom_4;
    using S2RCopyAtomGate = s2r_copy_atom_4;

    // s2r for sp
    using tile_mma_atom = MMA_Atom<MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>>;
    using tile_mma_perm = decltype(make_layout(make_shape(Int<1>{}, Int<4>{}, Int<1>{})));
    using tile_atom_x = s2r_copy_atom_2;
    using tile_atom_a = s2r_copy_atom_2;
    using S2RCopySpX = decltype(make_tiled_copy_B(
        tile_atom_x{},
        make_tiled_mma(tile_mma_atom{}, tile_mma_perm{},
                       Tile<Int<16>, Int<32>, Int<16>>{})
    ));
    using S2RCopySpUp = decltype(make_tiled_copy_B(
        tile_atom_a{},
        make_tiled_mma(tile_mma_atom{}, tile_mma_perm{},
                       Tile<Int<16>, Int<32>, Int<16>>{})
    ));
    using S2RCopySpGate = S2RCopySpUp;

    // cal config
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaPM = 32;
    static constexpr int kMmaPN = 32;
    static constexpr int kMmaPK = 16;
    using MMA = decltype(make_tiled_mma(
        mma_atom{}, make_layout(make_shape(_2{}, _2{}, _1{})),
        Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>{}
    ));

    // r2s config
    static constexpr int kSmemLayoutCBatch = 2;
    using SmemLayoutAtomResult = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                    make_stride(Int<kMmaPN>{}, Int<1>{}))
    ));
    using SmemLayoutResult = decltype(tile_to_shape(
        SmemLayoutAtomResult{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})
    ));
    using R2SCopyAtomR = Copy_Atom<UniversalCopy<int>, T>;

    // s2g config
    using S2GCopyAtomR = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyR = decltype(make_tiled_copy(
        S2GCopyAtomR{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN/8>{}),
                    make_stride(Int<kMmaPN/8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // other config
    static constexpr int shm_size = cute::cosize(SmemLayoutX{}) +
                                    cute::cosize(SmemLayoutUp{}) * 2 +
                                    cute::cosize(SmemLayoutSpUp{}) * 2;
    static constexpr int kShmSize = shm_size * sizeof(T);
};

}
