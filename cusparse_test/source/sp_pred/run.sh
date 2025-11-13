set -e

CUDA_VISIBLE_DEVICES=6 python train.py >/share/xujiaming/train_machine/yongkang/cusparse_test/data/preds/out.txt
