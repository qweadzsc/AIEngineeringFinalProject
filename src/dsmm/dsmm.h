#pragma once

#include <cuda_fp16.h>


void dense_sddmm(const half * x, const half * up, const half * gate, half * result,
                 const long long * row, const long long * col, half * val,
                 int bs, int hid_d, int mid_d, int t_d);

void dense_spmm(const half * x, const half * down1, const half * down2, half * result,
                const long long * row, const long long * col, const half * val,
                int bs, int hid_d, int mid_d, int t_d);
