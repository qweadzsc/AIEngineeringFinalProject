#pragma once
#include <cuda_fp16.h>


void cutlass_mlp(const half * input, const half * up, const half * gate, const half * down,
                 long long * mask_c, long long * mask_r, half * mask_v1, half * mask_v2, half * ir1, half * ir2, half * result,
                 int bs, int in_d, int mid_d, int t_d, int nnz, void * buf);
