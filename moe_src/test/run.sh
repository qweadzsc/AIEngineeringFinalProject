set -e

cd /share/public/zhouyongkang/projects/sc/moe_src
export TORCH_CUDA_ARCH_LIST="8.0"
# CUDA_LAUNCH_BLOCKING=1 \
# CUDA_VISIBLE_DEVICES=4 \
# compute-sanitizer --tool memcheck --print-level info --log-file sanitizer_output.txt \
# python test/test.py

CUDA_VISIBLE_DEVICES=2 \
python test/test.py
