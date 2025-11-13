#include <torch/extension.h>
#include "dsmm.h"


void torch_launch_sddmm_kernel(
          torch::Tensor & x,
    const torch::Tensor & up,
    const torch::Tensor & gate,
          torch::Tensor & ir,
    const torch::Tensor & mask_r,
    const torch::Tensor & mask_c,
          torch::Tensor & mask_v,
    int bs, int in_d, int mid_d, int exp_d, int t_dense, int maxnnz
) {
    sddmm_api((half *)x.data_ptr(),
              (const half *)up.data_ptr(),
              (const half *)gate.data_ptr(),
              (half *)ir.data_ptr(),
              (const long long *)mask_r.data_ptr(),
              (const long long *)mask_c.data_ptr(),
              (half *)mask_v.data_ptr(),
              bs, mid_d, in_d, exp_d, t_dense, maxnnz);
}


void torch_launch_spmm_kernel(
          torch::Tensor & x,
    const torch::Tensor & down,
          torch::Tensor & result,
    const torch::Tensor & mask_r,
    const torch::Tensor & mask_c,
    const torch::Tensor & mask_v,
    const torch::Tensor & exp_w,
    int bs, int in_d, int mid_d, int exp_d, int t_dense, int maxnnz
) {
    spmm_api((half *)x.data_ptr(),
             (const half *)down.data_ptr(),
             (half *)result.data_ptr(),
             (const long long *)mask_r.data_ptr(),
             (const long long *)mask_c.data_ptr(),
             (const half *)mask_v.data_ptr(),
             (const half *)exp_w.data_ptr(),
             bs, in_d, mid_d, exp_d, t_dense, maxnnz);
}


PYBIND11_MODULE(mlp_kernel, m) {
    m.def("torch_launch_sddmm_kernel",
          &torch_launch_sddmm_kernel,
          "sddmm kernel wrapper");
    
    m.def("torch_launch_spmm_kernel",
          &torch_launch_spmm_kernel,
          "spmm kernel wrapper");
}
