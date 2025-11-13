#pragma once
#include <cuda_fp16.h>


void mlp(const half * input, const half * up, const half * gate, const half * down,
         const half * spdown, const long long * mask_c, const long long * mask_r,
         half * mask_v, half * ir, half * result,
         int bs, int in_d, int mid_d, int t_d);
