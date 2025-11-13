#!/bin/bash

export TORCH_CUDA_ARCH_LIST="8.0"
CUDA_VISIBLE_DEVICES=7 python evaluate_p4.py \
-d /share/public/zhouyongkang/projects/sc/data/benchmark/mmlu \
-s /share/public/zhouyongkang/projects/sc/cusparse_test/result/mmlu_results \
-m test
