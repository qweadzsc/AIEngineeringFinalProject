set -e

cd /share/xujiaming/train_machine/yongkang/cusparse_test/source
export TORCH_CUDA_ARCH_LIST="8.0"
CUDA_VISIBLE_DEVICES=5 python main.py 2>out.err
