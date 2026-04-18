set -e

# CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=6 \
python main.py --dataset 0 --method bmeagle
