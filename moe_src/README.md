# moe_src

`moe_src` contains two parallel implementations for the MoE MLP path:

- A custom CUDA extension exposed as `mlp_kernel.ops.sddmm` and `mlp_kernel.ops.spmm`.
- A newer Triton validation path for mixed dense/sparse SDDMM.

## Directory Layout

- `setup.py`: builds the `mlp_kernel._C` extension from `torch_api.cpp`, `cuda_api.cu`, `sddmm.cu`, and `spmm.cu`.
- `mlp_kernel/`: Python wrapper plus C++/CUDA extension sources.
- `sddmm_validation/`: logical metadata and PyTorch reference implementation for mixed dense/sparse SDDMM.
- `triton_kernels/`: Triton prototype kernels plus launch metadata/program mapping helpers.
- `test/test_mixed_sddmm.py`: main structured validation suite for metadata, reference slicing, Triton correctness, and Triton benchmarks.
- `test/test.py`: legacy benchmark and replay script covering custom CUDA, Triton, and dense `torch.bmm`.
- `test/test_oss.py`: older OSS-style replay script with extra activation/bias assumptions; it is not part of the default test set.
- `test/test_torch_cuda.py`: focused benchmark for `torch.bmm` vs the custom CUDA kernel across different `t_d` values.
- `test/test_torch_cuda_hybrid.py`: focused benchmark for `torch.bmm` vs a hybrid path that first probes the sparse routing pattern, then still runs the existing custom CUDA kernel with the configured `t_d`.
- `run_test.sh`: default test entrypoint used by `run.sh`.

## Key Implementation Notes

### Custom CUDA path

- `mlp_kernel/csrc/torch_api.cpp` registers:
  - `mlp_kernel::launch_sddmm_kernel`
  - `mlp_kernel::launch_spmm_kernel`
- `mlp_kernel/csrc/cuda/cuda_api.cu` binds those operators to the CUDA implementations.
- `mlp_kernel/ops.py` exposes user-facing Python wrappers.

The CUDA path is the older production-style implementation. It computes:

- Dense expert intermediates for `t_d` experts over all tokens.
- Sparse expert intermediates for `t_d // sp_pd` experts over only `maxnnz` selected rows.

### Triton path

- `sddmm_validation/mixed_sddmm.py` freezes the logical layout:
  - `mask_c_dense[dense_width]`
  - `mask_c_sparse[sparse_width]`
  - `mask_r_sparse[maxnnz, sparse_width]`
  - outputs `ir_dense[batch, dense_width, expert_block_size]`
  - outputs `mask_v_sparse[maxnnz, sparse_width, expert_block_size]`
- `triton_kernels/mixed_sddmm.py` provides:
  - mixed kernel
  - dense-only specialized kernel
  - sparse-only specialized kernel
  - launch metadata and program-id mapping helpers

## What The Existing Tests Check

### `test/test_mixed_sddmm.py`

This is the main test file.

- `MixedSDDMMMetadataTest`
  - verifies dense/sparse expert splitting
  - verifies `mask_r_sparse` row selection
  - checks legal edge cases such as `dense_width == 0` or `sparse_width == 0`
- `MixedSDDMMReferenceTest`
  - verifies the reference path matches "full SDDMM then slice by metadata"
- `MixedSDDMMTritonTest`
  - checks Triton vs reference across multiple mixed-width cases
  - checks ragged shapes
  - checks dense-only and sparse-only specialized kernels vs the mixed kernel
  - checks launch metadata and program-id coverage
  - contains optional Triton performance benchmarks

Optional benchmark sections in that file:

- `RUN_MIXED_SDDMM_BENCH=1`
  - dense-only step-4 benchmark
  - larger-total-N dense-only benchmark
  - mixed-vs-dense torch `bmm` benchmark
- `RUN_MIXED_SDDMM_TUNE=1`
  - dense-only Triton parameter sweep

### `test/test.py`

This is a benchmark-oriented legacy script.

- `forward_cuda(...)`
  - uses the custom CUDA `sddmm + spmm` path
- `forward_triton(...)`
  - uses the Triton mixed SDDMM plus the custom CUDA `spmm`
- `dense_forward(...)`
  - computes the full dense MoE MLP using batched `torch.bmm`

It also contains replay helpers that compare:

- dense SDDMM intermediates vs PyTorch oracle
- sparse SDDMM intermediates vs PyTorch oracle
- selected-path end-to-end replay vs kernel output
- full-MoE oracle vs the selected-path approximation

`main_v2()` is the current default entrypoint and prints only speed benchmarks.

### `test/test_oss.py`

This is an older replay script with additional assumptions:

- explicit bias additions
- clipping
- custom activation reshaping
- zeroing the tail after `original_expert_dim`

Because of those extra assumptions and layout differences, it is treated as a legacy optional test instead of a default one.

### `test/test_torch_cuda.py`

This benchmark compares:

- `TestCUDAMoe.forward_cuda(...)`
- `TestCUDAMoe.dense_forward(...)`

It sweeps `t_d` from `1` to `64`, measures median CUDA-event latency, and writes:

- `test/torch_cuda_td_sweep.csv`
- `test/torch_cuda_td_sweep_metadata.txt`
- `test/torch_cuda_td_sweep.png`

The figure uses a white background and has one subplot per `(batch_size, maxnnz)` pair.

The current version fixes the benchmark seed within each subplot:

- `benchmark_seed = batch_size * 1000 + maxnnz * 100`
- the seed does not depend on `t_d`
- within one subplot, the `torch.bmm` baseline therefore sees the same random model weights and the same random inputs

### `test/test_torch_cuda_hybrid.py`

This benchmark compares:

- a hybrid path: first run `build_sparse_routing_weights(...)`, then count the nonzero token count per expert, then compute `actual_t_d = count(nonzero_count > maxnnz)`, and finally still run `forward_cuda(...)` with the configured `t_d`
- `TestCUDAMoe.dense_forward(...)`

Important semantic note:

- `actual_t_d` is only used as an extra timed preprocessing step and as a recorded metric.
- The CUDA kernel still executes with the configured `t_d`, not the probed `actual_t_d`.

This benchmark writes:

- `test/torch_cuda_hybrid_td_sweep.csv`
- `test/torch_cuda_hybrid_td_sweep_metadata.txt`
- `test/torch_cuda_hybrid_td_sweep.png`

The figure also uses a white background and has one subplot per `(batch_size, maxnnz)` pair.

## Input-Scale Constraints For `test.py`

The CUDA and dense `torch.bmm` paths in `test.py` impose practical constraints:

- `maxnnz <= batch_size * sequence_length`
  - sparse rows are selected with `topk(maxnnz, dim=0)` over tokens
- `t_d + (t_d // sp_pd) <= num_experts`
  - the CUDA path selects `dense_width + sparse_width` experts with `topk(...)`
- `sp_pd > 0`
- `top_k <= num_experts`
- `expert_w % 128 == 0` is strongly preferred
  - `test.py` warns that the CUDA SDDMM layout assumes 128-column tiles

For the two focused benchmark files, the fixed hyperparameters are:

- `batch_size in {64, 128}`
- `sequence_length = 1`
- `tokens = batch_size * sequence_length`
- `maxnnz in {4, 8}`
- `t_d in [1, 64]`
- `hid_dim = 2048`
- `num_experts = 128`
- `expert_w = 512`
- `top_k = 8`
- `sp_pd = 1`
- `dtype = float16`
- `device = cuda`

With `sp_pd = 1`, the CUDA path uses:

- `sparse_width = t_d`
- `selected_experts = t_d + sparse_width = 2 * t_d`

So the sweep is legal because the largest case is `2 * 64 = 128 = num_experts`.

## Environment Notes

The provided environment for the kernel benchmarks is:

- Python: `/share/zhouyongkang/conda_envs/mtpmoe_env/bin/python`

In practice this environment should be launched with `-s`:

```bash
/share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s ...
```

Without `-s`, user-site packages under `~/.local/lib/python3.12` can shadow the environment's `numpy`, which breaks `torch` import.

## How To Run

Main structured validation:

```bash
cd /share/zhouyongkang/projects/sc/moe_src
/share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v test.test_mixed_sddmm
```

Enable Triton benchmarks:

```bash
cd /share/zhouyongkang/projects/sc/moe_src
RUN_MIXED_SDDMM_BENCH=1 /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v test.test_mixed_sddmm
RUN_MIXED_SDDMM_TUNE=1 /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_parameter_sweep
```

Legacy benchmark script:

```bash
cd /share/zhouyongkang/projects/sc
/share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s moe_src/test/test.py
```

Focused CUDA-vs-`torch.bmm` benchmark:

```bash
cd /share/zhouyongkang/projects/sc
RUN_TORCH_CUDA_BENCH=1 /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v moe_src.test.test_torch_cuda
```

Focused hybrid-vs-`torch.bmm` benchmark:

```bash
cd /share/zhouyongkang/projects/sc
RUN_TORCH_CUDA_HYBRID_BENCH=1 /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v moe_src.test.test_torch_cuda_hybrid
```

## Measured Results On July 8, 2026

Environment used during inspection:

- GPU: NVIDIA A800-SXM4-80GB
- Torch: `2.6.0+cu124`
- Triton: `3.2.0`

### `test_mixed_sddmm.py`

- Default run:
  - `18` tests ran
  - `14` executed and passed
  - `4` Triton benchmark tests skipped by design
- With `RUN_MIXED_SDDMM_BENCH=1`:
  - small `total_n=2048` dense-only Triton was slower than the dense PyTorch baseline
  - larger `total_n=8192` tuned dense-only Triton became faster than the dense PyTorch baseline
- With `RUN_MIXED_SDDMM_TUNE=1`:
  - best observed config on this machine was:
    - `{'block_m': 64, 'block_n': 128, 'block_k': 32, 'dense_group_m': 4, 'num_warps': 8, 'num_stages': 3}`

### `test.py`

- The default `main_v2()` benchmark showed:
  - the legacy custom CUDA path was usually faster than the Triton path
  - the Triton path only approached or slightly beat the legacy CUDA path in the largest tested case
- Manual replay of `forward_triton(...)` showed:
  - dense SDDMM mismatches: `0/524288`
  - sparse SDDMM mismatches: `0/32768`
  - selected-path end-to-end replay mismatches: `1/131072`
- Manual replay of `forward_cuda(...)` showed:
  - dense SDDMM mismatches: `7/524288`
  - sparse SDDMM mismatches: `157/32768`
  - selected-path end-to-end replay mismatches: `2/131072`
- The "full MoE oracle" mismatch remains large for both paths because the kernel path computes only the selected dense/sparse subset rather than the full all-expert dense computation.

### `test_oss.py`

This script produced large mismatches in the current tree:

- dense comparison: `84.2624%` mismatched positions
- sparse comparison: `86.4035%` mismatched positions
- end-to-end comparison: `48.5779%` mismatched positions

That matches its status as a legacy optional script instead of a default validation target.

### `test/test_torch_cuda.py`

Measured command:

```bash
RUN_TORCH_CUDA_BENCH=1 /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v moe_src.test.test_torch_cuda
```

This is the seed-fixed rerun, meaning the benchmark seed is fixed per `(batch_size, maxnnz)` subplot and does not depend on `t_d`.

Observed summary:

- overall: `256` cases, custom CUDA wins `173`, `torch.bmm` wins `83`
- overall mean `cuda_ms / bmm_ms = 0.9416`
- overall geometric mean `cuda_ms / bmm_ms = 0.8643`
- `batch=64, maxnnz=4`: custom CUDA wins `56`, `torch.bmm` wins `8`, first `torch.bmm` win at `t_d=57`
- `batch=64, maxnnz=8`: custom CUDA wins `32`, `torch.bmm` wins `32`, first `torch.bmm` win at `t_d=33`
- `batch=128, maxnnz=4`: custom CUDA wins `64`, `torch.bmm` wins `0`
- `batch=128, maxnnz=8`: custom CUDA wins `21`, `torch.bmm` wins `43`, first `torch.bmm` win at `t_d=22`

The `torch.bmm` line is now visibly flatter within each subplot. In the seed-fixed CSV it stayed within these narrow bands:

- `batch=64, maxnnz=4`: about `0.8737 ms` to `0.9378 ms`
- `batch=64, maxnnz=8`: about `0.8735 ms` to `0.8990 ms`
- `batch=128, maxnnz=4`: about `1.1095 ms` to `1.1199 ms`
- `batch=128, maxnnz=8`: about `1.1092 ms` to `1.1202 ms`

This residual variation is still measurement noise, not a real algorithmic dependence on `t_d`. Likely contributors are:

- GPU clock / DVFS fluctuation
- CUDA launch and synchronization jitter
- allocator/cache state
- small cuBLAS kernel-selection and cache effects
- the fact that each point is still a real runtime measurement instead of a symbolic estimate

### `test/test_torch_cuda_hybrid.py`

Measured command:

```bash
RUN_TORCH_CUDA_HYBRID_BENCH=1 /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python -s -m unittest -v moe_src.test.test_torch_cuda_hybrid
```

Observed summary:

- overall: `256` cases, hybrid wins `151`, `torch.bmm` wins `105`
- overall mean `hybrid_ms / bmm_ms = 1.0274`
- overall geometric mean `hybrid_ms / bmm_ms = 0.9565`
- `batch=64, maxnnz=4`: `actual_t_d=49`, hybrid wins `52`, `torch.bmm` wins `12`, first `torch.bmm` win at `t_d=53`
- `batch=64, maxnnz=8`: `actual_t_d=2`, hybrid wins `27`, `torch.bmm` wins `37`, first `torch.bmm` win at `t_d=28`
- `batch=128, maxnnz=4`: `actual_t_d=111`, hybrid wins `54`, `torch.bmm` wins `10`, first `torch.bmm` win at `t_d=55`
- `batch=128, maxnnz=8`: `actual_t_d=60`, hybrid wins `18`, `torch.bmm` wins `46`, first `torch.bmm` win at `t_d=16`

The extra probe step matters most when the configured `t_d` is already large. In particular, the hybrid path loses substantially earlier than the plain CUDA path for the larger-`maxnnz` settings, because it pays the probe overhead and still launches the same CUDA kernel with the configured `t_d`.

## Artifact Paths

Current generated artifacts are:

- `moe_src/test/torch_cuda_td_sweep.csv`
- `moe_src/test/torch_cuda_td_sweep_metadata.txt`
- `moe_src/test/torch_cuda_td_sweep.png`
- `moe_src/test/torch_cuda_hybrid_td_sweep.csv`
- `moe_src/test/torch_cuda_hybrid_td_sweep_metadata.txt`
- `moe_src/test/torch_cuda_hybrid_td_sweep.png`
