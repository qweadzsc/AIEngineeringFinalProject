#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>


namespace config {

using namespace cute;

template <typename T_, int kTileM_=64, int kTileN_=16, int kTileK_=64, int kBlock_=4,
          int maxnnz_=320, int kStage_=3>
struct Config {
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kBlock = kBlock_;
    static constexpr int maxnnz = maxnnz_;
    static constexpr int kStage = kStage_;

    // g2s config
    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;
    using SwizzleSize = Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>;
    using SmemLayoutAtom = decltype(composition(
        SwizzleSize{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}))
    ));
    using SmemLayoutX = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutDown = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutSpDown = decltype(tile_to_shape(
        composition(SwizzleSize{}, make_layout(make_shape(_16{}, _32{}),
                                               make_stride(_32{}, _1{}))),
        make_shape(Int<64>{}, Int<32>{}, Int<kStage>{})
    ));
    using SmemLayoutSpDownCp = decltype(tile_to_shape(
        composition(SwizzleSize{}, make_layout(make_shape(_2{}, _256{}),
                                               make_stride(_256{}, _1{}))),
        make_shape(Int<8>{}, Int<256>{}, Int<kStage>{})
    ));
    using SmemLayoutVal = decltype(make_layout(make_shape(_1024{})));
    using SmemLayoutCol = decltype(make_layout(make_shape(_512{})));

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using g2s_copy_atom_int64 = Copy_Atom<g2s_copy_traits, long long>;

    using G2SCopyX = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyDown = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopySpDown = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyCol = decltype(make_tiled_copy(
        g2s_copy_atom_int64{},
        make_layout(make_shape(Int<128>{})),
        make_layout(make_shape(Int<2>{}))
    ));
    using G2SCopyVal = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<128>{})),
        make_layout(make_shape(Int<8>{}))
    ));

    // s2r config
    using s2r_copy_op_a = SM75_U32x4_LDSM_N;
    using s2r_copy_traits_a = Copy_Traits<s2r_copy_op_a>;
    using s2r_copy_atom_a = Copy_Atom<s2r_copy_traits_a, T>;
    using s2r_copy_op_b = SM75_U32x2_LDSM_N;
    using s2r_copy_traits_b = Copy_Traits<s2r_copy_op_b>;
    using s2r_copy_atom_b = Copy_Atom<s2r_copy_traits_b, T>;

    using S2RCopyAtomX = s2r_copy_atom_a;
    using S2RCopyAtomDown = s2r_copy_atom_a;
    using S2RCopyVal = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<int>, T>{},
        make_layout(make_shape(_1{})), make_layout(make_shape(_2{}))
    ));

    // cal config
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 4;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;  // (16, 8, 16)
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(
        make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{}))
    );
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    // sp smem <-> reg config
    using MMAForCopy = decltype(make_tiled_mma(
        MMA_Atom<MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<4>{})),
        Tile<Int<16>, Int<8>, Int<32>>{}
    ));
    using S2RCopySpDown = decltype(make_tiled_copy_A(s2r_copy_atom_b{}, MMAForCopy{}));

    // r2s config
    static constexpr int kSmemLayoutCBatch = 2;
    using SmemLayoutAtomResult = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), make_stride(Int<kMmaPN>{}, Int<1>{}))
    ));
    using SmemLayoutResult = decltype(tile_to_shape(
        SmemLayoutAtomResult{}, make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})
    ));
    using SmemLayoutSpResult = decltype(make_layout(make_shape(Int<4*kTileM>{})));
    using R2SCopyAtomR = Copy_Atom<UniversalCopy<int>, T>;
    using R2SCopySpResult = S2RCopyVal;

    // s2g config
    using S2GCopyAtomR = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyR = decltype(make_tiled_copy(
        S2GCopyAtomR{},
        make_layout(make_shape(Int<64>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using S2GCopySpR = decltype(make_tiled_copy(
        S2GCopyAtomR{},
        make_layout(make_shape(Int<1>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // other config
    static constexpr int shm_size = cute::cosize(SmemLayoutX{}) +
                                    cute::cosize(SmemLayoutDown{}) +
                                    cute::cosize(SmemLayoutSpDownCp{}) +
                                    cute::cosize(SmemLayoutCol{})*4 +
                                    cute::cosize(SmemLayoutVal{});
    static constexpr int kShmSize = shm_size * sizeof(T);
};

}
