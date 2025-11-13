set -e

cd /share/public/zhouyongkang/projects/sc/src
export TORCH_CUDA_ARCH_LIST="8.0"
CUDA_VISIBLE_DEVICES=5 python main.py 2>out.err
