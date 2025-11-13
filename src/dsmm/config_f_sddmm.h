#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace config {

using namespace cute;

template <typename T_, int kTileM_=32, int kTileN_=32, int kTileK_=64, int kTileS_=80,
          int maxnnz_=64, int kStage_=3>
struct Config {
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kTileS = kTileS_;
    static constexpr int maxnnz = maxnnz_;
    static constexpr int kStage = kStage_;

    static constexpr int cal_time_block = kTileM * kTileN / 256;
    static constexpr int sp_cal_num = maxnnz / 2 / cal_time_block;

    // shm config
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<512/kTileK>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, _1{}))
    ));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutUp = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutGate = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
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
    using s2r_copy_op_1 = SM75_U32x1_LDSM_N;
    using s2r_copy_atom_1 = Copy_Atom<Copy_Traits<s2r_copy_op_1>, T>;
    using s2r_copy_op_2 = SM75_U32x2_LDSM_N;
    using s2r_copy_atom_2 = Copy_Atom<Copy_Traits<s2r_copy_op_2>, T>;
    using s2r_copy_op_4 = SM75_U32x4_LDSM_N;
    using s2r_copy_atom_4 = Copy_Atom<Copy_Traits<s2r_copy_op_4>, T>;

    using S2RCopyAtomX = s2r_copy_atom_4;
    using S2RCopyAtomUp = s2r_copy_atom_4;
    using S2RCopyAtomGate = s2r_copy_atom_4;

    // s2r for sp
    using tile_mma_atom = MMA_Atom<MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>>;
    using tile_mma_perm = decltype(make_layout(make_shape(Int<1>{}, Int<4>{}, Int<1>{})));
    static constexpr int KForTile = 64 / cal_time_block;
    static constexpr int NForTile = sp_cal_num * 128 / KForTile;
    // if constexpr (KForTile == 8) {
    //     using tile_atom_x = s2r_copy_atom_1;
    // } else if constexpr (kForTile == 16) {
    //     using tile_atom_x = s2r_copy_atom_2;
    // } else {
    //     using tile_atom_x = s2r_copy_atom_4;
    // }
    // using S2RCopySpX = decltype(make_tiled_copy_B(
    //     tile_atom_x{},
    //     make_tiled_mma(tile_mma_atom{}, tile_mma_perm{},
    //                    Tile<Int<16>, Int<32>, Int<KForTile>>{})
    // ));
    // if constexpr (sp_cal_num == 2) {
    //     using tile_atom_a = s2r_copy_atom_1;
    // } else if constexpr (sp_cal_num == 4) {
    //     using tile_atom_a = s2r_copy_atom_2;
    // } else {
    //     using tile_atom_a = s2r_copy_atom_4;
    // }
    // using S2RCopySpUp = decltype(make_tiled_copy_B(
    //     tile_atom_a{},
    //     make_tiled_mma(tile_mma_atom{}, tile_mma_perm{},
    //                    Tile<Int<16>, Int<NForTile>, Int<KForTile>>{})
    // ));
    // using S2RCopySpGate = S2RCopySpUp;

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
    //     using MMA = decltype(make_tiled_mma(
    //         mma_atom{}, make_layout(make_shape(_4{}, _1{}, _1{})),
    //         Tile<_64{}, _16{}, _16{}>{}
    //     ));
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
        make_layout(make_shape(Int<64>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // other config
    static constexpr int shm_size = cute::cosize(SmemLayoutX{}) +
                                    cute::cosize(SmemLayoutUp{}) * 2 +
                                    cute::cosize(SmemLayoutSpUp{}) * 2;
    static constexpr int kShmSize = shm_size * sizeof(T);
};

// 32x8x64, nnz 32
template <typename T_, int kTileS_>
struct Config<T_, 32, 8, 64, kTileS_, 32, 3> {
    using T = T_;
    static constexpr int kTileM = 32;
    static constexpr int kTileN = 8;
    static constexpr int kTileK = 64;
    static constexpr int kTileS = kTileS_;
    static constexpr int maxnnz = 32;
    static constexpr int kStage = 3;
    static_assert(kTileS >= 32);

    // shm config
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(_8{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, _1{}))
    ));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutUp = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutGate = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutSpUp = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<maxnnz>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutSpGate = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<maxnnz>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    // mask config
    static constexpr int thread_per_col = 128 / kTileN;
    static constexpr int elmt_per_thread = kTileS / thread_per_col;

    // g2s config
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopyX = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyUp = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyGate = G2SCopyUp;
    using G2SCopySpUp = G2SCopyUp;
    using G2SCopySpGate = G2SCopyUp;

    // s2r config
    // using s2r_copy_op_1 = SM75_U32x1_LDSM_N;
    // using s2r_copy_atom_1 = Copy_Atom<Copy_Traits<s2r_copy_op_1>, T>;
    using s2r_copy_op_2 = SM75_U32x2_LDSM_N;
    using s2r_copy_atom_2 = Copy_Atom<Copy_Traits<s2r_copy_op_2>, T>;
    using s2r_copy_op_4 = SM75_U32x4_LDSM_N;
    using s2r_copy_atom_4 = Copy_Atom<Copy_Traits<s2r_copy_op_4>, T>;

    using S2RCopyAtomX = s2r_copy_atom_2;
    using S2RCopyAtomUp = s2r_copy_atom_4;
    using S2RCopyAtomGate = s2r_copy_atom_4;

    // s2r for sp
    using tile_mma_atom = MMA_Atom<MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>>;
    using tile_mma_perm = decltype(make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})));
    using MMA_For_Tile = decltype(make_tiled_mma(
        tile_mma_atom{}, tile_mma_perm{}, Tile<_16, _8, _32>{}
    ));
    using S2RCopySpX = decltype(make_tiled_copy_B(s2r_copy_atom_4{}, MMA_For_Tile{}));
    using S2RCopySpUp = S2RCopySpX;
    using S2RCopySpGate = S2RCopySpUp;

    // cal config
    static constexpr int cal_time = 2;
    using mma_atom = MMA_Atom<MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>>;
    using MMA = decltype(make_tiled_mma(
        mma_atom{}, make_layout(make_shape(_2{}, _1{}, _2{})),
        Tile<_32, _8, _32>{}
    ));
    static constexpr int sp_cal_n = 1;
    static constexpr int sp_cal_k = 4;

    // r2s config
    static constexpr int kSmemLayoutCBatch = 4;
    using SmemLayoutAtomResult = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<8>{}),
                    make_stride(Int<8>{}, Int<1>{}))
    ));
    using SmemLayoutResult = decltype(tile_to_shape(
        SmemLayoutAtomResult{},
        make_shape(Int<32>{}, Int<8>{}, Int<kSmemLayoutCBatch>{})
    ));
    using R2SCopyAtomR = Copy_Atom<UniversalCopy<int>, T>;

    // s2g config
    using S2GCopyAtomR = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyR = decltype(make_tiled_copy(
        S2GCopyAtomR{},
        make_layout(make_shape(Int<32>{}, Int<1>{}), make_stride(Int<1>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // share mem
    static constexpr int shm_size = cute::cosize(SmemLayoutX{}) +
                                    cute::cosize(SmemLayoutUp{}) * 2 +
                                    cute::cosize(SmemLayoutSpUp{}) * 2;
    static constexpr int kShmSize = shm_size * sizeof(T);
};

}
