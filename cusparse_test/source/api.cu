#include <torch/extension.h>
#include "impl.h"


void torch_launch_cutlass_mlp(
    const torch::Tensor & input,
    const torch::Tensor & up,
    const torch::Tensor & gate,
    const torch::Tensor & down,
          torch::Tensor & mask_c,
          torch::Tensor & mask_r,
          torch::Tensor & mask_v1,
          torch::Tensor & mask_v2,
          torch::Tensor & ir1,
          torch::Tensor & ir2,
          torch::Tensor & result,
          torch::Tensor & buf,
    int bs, int in_d, int mid_d, int t_dense, int nnz
) {
    cutlass_mlp((const half *)input.data_ptr(),
                (const half *)up.data_ptr(),
                (const half *)gate.data_ptr(),
                (const half *)down.data_ptr(),
                (long long *)mask_c.data_ptr(),
                (long long *)mask_r.data_ptr(),
                (half *)mask_v1.data_ptr(),
                (half *)mask_v2.data_ptr(),
                (half *)ir1.data_ptr(),
                (half *)ir2.data_ptr(),
                (half *)result.data_ptr(),
                bs, in_d, mid_d, t_dense, nnz,
                (void *)buf.data_ptr());
}


PYBIND11_MODULE(cutlass_mlp, m) {
    m.def("torch_launch_cutlass_mlp",
          &torch_launch_cutlass_mlp,
          "cublas mlp kernel wrapper");
}
