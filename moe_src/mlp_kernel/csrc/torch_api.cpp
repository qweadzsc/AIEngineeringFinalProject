#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace mlp_kernel { // 使用命名空间来组织代码

void launch_sddmm_kernel_cpu(
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
    // CPU 版本的 SDDMM 实现 (占位符)
    // 目前仅抛出错误，表明 CPU 实现未提供
    TORCH_CHECK_NOT_IMPLEMENTED(false, "launch_sddmm_kernel CPU implementation is not available.");
    // 如果要提供空操作而不是错误，可以留空或添加注释：
    // TORCH_WARN_ONCE("launch_sddmm_kernel called on CPU, no implementation available.");
}

void launch_spmm_kernel_cpu(
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
    // CPU 版本的 SPMM 实现 (占位符)
    // 目前仅抛出错误，表明 CPU 实现未提供
    TORCH_CHECK_NOT_IMPLEMENTED(false, "launch_spmm_kernel CPU implementation is not available.");
    // 如果要提供空操作而不是错误，可以留空或添加注释：
    // TORCH_WARN_ONCE("launch_spmm_kernel called on CPU, no implementation available.");
}

// --- 注册 PyTorch 算子 ---
TORCH_LIBRARY(mlp_kernel, m) {
    // 注册 sddmm 算子
    // 签名字符串描述了算子的输入输出和属性
    // (Tensor x, Tensor up, Tensor gate, Tensor ir, Tensor mask_r, Tensor mask_c, Tensor mask_v,
    //  int bs, int in_d, int mid_d, int exp_d, int t_dense, int maxnnz) -> ()
    m.def("launch_sddmm_kernel(Tensor x, Tensor up, Tensor gate, Tensor(a!) ir, Tensor mask_r, Tensor mask_c, Tensor(b!) mask_v, int bs, int in_d, int mid_d, int exp_d, int t_dense, int maxnnz) -> ()");

    // 注册 spmm 算子
    // (Tensor x, Tensor down, Tensor result, Tensor mask_r, Tensor mask_c, Tensor mask_v, Tensor exp_w,
    //  int bs, int in_d, int mid_d, int exp_d, int t_dense, int maxnnz) -> ()
    m.def("launch_spmm_kernel(Tensor x, Tensor down, Tensor(a!) result, Tensor mask_r, Tensor mask_c, Tensor mask_v, Tensor exp_w, int bs, int in_d, int mid_d, int exp_d, int t_dense, int maxnnz) -> ()");
}

// --- 注册 CPU 后端实现 ---
TORCH_LIBRARY_IMPL(mlp_kernel, CPU, m) { // 注册到 CPU 后端
    m.impl("launch_sddmm_kernel", &launch_sddmm_kernel_cpu);
    m.impl("launch_spmm_kernel", &launch_spmm_kernel_cpu);
}

}
