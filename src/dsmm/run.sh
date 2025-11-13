set -e

cd /share/public/zhouyongkang/projects/sc/src/dsmm
nvcc main_ref.cu -o main -O3 -arch=sm_80 -std=c++17 \
    --expt-relaxed-constexpr -cudart shared --cudadevrt none \
    -lcublas -lcublasLt -lcusparse \
    -I /share/public/zhouyongkang/projects/sc/deps/cutlass/include \
    -I /share/public/zhouyongkang/projects/sc/src/dsmm \
    -Xptxas -v --use_fast_math \
    2>compile.err

CUDA_VISIBLE_DEVICES=7 ./main
