#include "impl.h"
#include "dsmm/dsmm.h"

void mlp(const half * input, const half * up, const half * gate, const half * down,
         const half * spdown, const long long * mask_c, const long long * mask_r,
         half * mask_v, half * ir, half * result,
         int bs, int in_d, int mid_d, int t_d)
{
    dense_sddmm(input, up, gate, ir, mask_r, mask_c, mask_v, bs, in_d, mid_d, t_d);
    dense_spmm(ir, down, spdown, result, mask_r, mask_c, mask_v, bs, in_d, mid_d, t_d);
}