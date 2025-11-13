import os
import torch
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- 配置部分 ---
this_dir = os.path.dirname(os.path.curdir)
extensions_dir = os.path.join(this_dir, "mlp_kernel", "csrc")
# sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
# sources += cuda_sources

sources = [
    "/share/public/zhouyongkang/projects/sc/moe_src/mlp_kernel/csrc/torch_api.cpp",
    "/share/public/zhouyongkang/projects/sc/moe_src/mlp_kernel/csrc/cuda/cuda_api.cu",
    "/share/public/zhouyongkang/projects/sc/moe_src/mlp_kernel/csrc/cuda/sddmm.cu",
    "/share/public/zhouyongkang/projects/sc/moe_src/mlp_kernel/csrc/cuda/spmm.cu",
]

cutlass_path = "/share/public/zhouyongkang/projects/sc/deps/cutlass"
include_dirs = [
    '.',
    extensions_cuda_dir,
    f'{cutlass_path}/include',
]

extension_name = 'mlp_kernel'

# --- 编译选项 ---
extra_compile_args = {
    'cxx': [
        '-O3',
        "-fdiagnostics-color=always",
        # "-DPy_LIMITED_API=0x030A0000",
    ],
    'nvcc': [
        '-O3',
        '-arch=sm_80',
        '--expt-relaxed-constexpr',
        '--extended-lambda',
        '--use_fast_math',
        # '-lineinfo', '-g',
        # '-Xptxas', '-v',
    ]
}

# libraries = ['cusparse', 'cublas']
libraries = []
library_dirs = []
# --- 配置结束 ---

setup(
    name=extension_name,
    packages=find_packages(),
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name=f"{extension_name}._C",  # Python 导入名
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
            library_dirs=library_dirs,
            py_limited_api=True,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False, # 通常对于 C++ 扩展建议设为 False
    # options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
