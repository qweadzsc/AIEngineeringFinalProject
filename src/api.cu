#include <torch/extension.h>
#include "impl.h"


void torch_launch_mlp_kernel(
    const torch::Tensor & input,
    const torch::Tensor & up,
    const torch::Tensor & gate,
    const torch::Tensor & down1,
    const torch::Tensor & down2,
          torch::Tensor & mask_c,
          torch::Tensor & mask_r,
          torch::Tensor & mask_v,
          torch::Tensor & ir,
          torch::Tensor & result,
    int bs, int in_d, int mid_d, int t_dense
) {
    mlp((const half *)input.data_ptr(),
        (const half *)up.data_ptr(),
        (const half *)gate.data_ptr(),
        (const half *)down1.data_ptr(),
        (const half *)down2.data_ptr(),
        (long long *)mask_c.data_ptr(),
        (long long *)mask_r.data_ptr(),
        (half *)mask_v.data_ptr(),
        (half *)ir.data_ptr(),
        (half *)result.data_ptr(),
        bs, in_d, mid_d, t_dense);
}


PYBIND11_MODULE(mlp_kernel, m) {
    m.def("torch_launch_mlp_kernel",
          &torch_launch_mlp_kernel,
          "mlp kernel wrapper");
}
