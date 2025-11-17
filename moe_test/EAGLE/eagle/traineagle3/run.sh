cd eagle/traineagle3
CUDA_VISIBLE_DEVICES=4,5,6,7 \
deepspeed main.py \
    --basepath /share/public/zhouyongkang/models/Phi-tiny-MoE-instruct/ \
    --trainpath \
    --savedir /share/public/zhouyongkang/projects/sc/moe_test/EAGLE/eagle/traineagle3/em/ \
    --deepspeed_config ds_config.json