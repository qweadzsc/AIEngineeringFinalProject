#include <ATen/ATen.h> // 包含 ATen 头文件，提供 Tensor 等
#include <torch/all.h>
#include <torch/library.h> // TORCH_LIBRARY 相关

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h> // 获取 CUDA 流

#include "dsmm.h"


namespace mlp_kernel {

void launch_sddmm_kernel(
    const at::Tensor& x,
    const at::Tensor& up,
    const at::Tensor& gate,
          at::Tensor& ir,
    const at::Tensor& mask_r,
    const at::Tensor& mask_c,
          at::Tensor& mask_v,
    int64_t bs,
    int64_t in_d,
    int64_t mid_d,
    int64_t exp_d,
    int64_t t_dense,
    int64_t maxnnz
) {
    TORCH_CHECK(x.is_cuda() && up.is_cuda() && gate.is_cuda() && ir.is_cuda() &&
                mask_r.is_cuda() && mask_c.is_cuda() && mask_v.is_cuda(),
                "All tensors must be on CUDA device");
    TORCH_CHECK(x.dtype() == at::kHalf && up.dtype() == at::kHalf && gate.dtype() == at::kHalf &&
                ir.dtype() == at::kHalf && mask_v.dtype() == at::kHalf,
                "Tensor x, up, gate, ir, mask_v must be of type Half");
    TORCH_CHECK(mask_r.dtype() == at::kLong && mask_c.dtype() == at::kLong,
                "Tensor mask_r, mask_c must be of type Long");

    sddmm_api(
        reinterpret_cast<half*>(x.data_ptr()),
        reinterpret_cast<const half*>(up.data_ptr()),
        reinterpret_cast<const half*>(gate.data_ptr()),
        reinterpret_cast<half*>(ir.data_ptr()),
        reinterpret_cast<const long long*>(mask_r.data_ptr()),
        reinterpret_cast<const long long*>(mask_c.data_ptr()),
        reinterpret_cast<half*>(mask_v.data_ptr()),
        static_cast<int>(bs),
        static_cast<int>(mid_d),
        static_cast<int>(in_d),
        static_cast<int>(exp_d),
        static_cast<int>(t_dense),
        static_cast<int>(maxnnz)
    );
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // cudaStreamSynchronize(stream);
}


void launch_spmm_kernel(
    const at::Tensor& x,
    const at::Tensor& down,
          at::Tensor& result,
    const at::Tensor& mask_r,
    const at::Tensor& mask_c,
    const at::Tensor& mask_v,
    const at::Tensor& exp_w,
    int64_t bs,
    int64_t in_d,
    int64_t mid_d,
    int64_t exp_d,
    int64_t t_dense,
    int64_t maxnnz
) {
    TORCH_CHECK(x.is_cuda() && down.is_cuda() && result.is_cuda() &&
                mask_r.is_cuda() && mask_c.is_cuda() && mask_v.is_cuda() && exp_w.is_cuda(),
                "All tensors must be on CUDA device");
    TORCH_CHECK(x.dtype() == at::kHalf && down.dtype() == at::kHalf &&
                result.dtype() == at::kHalf && mask_v.dtype() == at::kHalf && exp_w.dtype() == at::kHalf,
                "Tensors x, down, result, mask_v, exp_w must be of type Half");
    TORCH_CHECK(mask_r.dtype() == at::kLong && mask_c.dtype() == at::kLong,
                "Tensor mask_r, mask_c must be of type Long");

    spmm_api(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(down.data_ptr()),
        reinterpret_cast<half*>(result.data_ptr()),
        reinterpret_cast<const long long*>(mask_r.data_ptr()),
        reinterpret_cast<const long long*>(mask_c.data_ptr()),
        reinterpret_cast<const half*>(mask_v.data_ptr()),
        reinterpret_cast<const half*>(exp_w.data_ptr()),
        static_cast<int>(bs),
        static_cast<int>(in_d),
        static_cast<int>(mid_d),
        static_cast<int>(exp_d),
        static_cast<int>(t_dense),
        static_cast<int>(maxnnz)
    );
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // cudaStreamSynchronize(stream);
}

TORCH_LIBRARY_IMPL(mlp_kernel, CUDA, m) {
    // 将 C++ 函数实现与算子名称关联
    m.impl("launch_sddmm_kernel", &launch_sddmm_kernel);
    m.impl("launch_spmm_kernel", &launch_spmm_kernel);
}

}