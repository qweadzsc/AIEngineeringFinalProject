set -e

CUDA_VISIBLE_DEVICES=6 CUDA_LAUNCH_BLOCKING=1 \
python main_new.py --dataset 0 --method mtp
