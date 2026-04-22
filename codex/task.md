# Mixed Dense-Sparse Triton SDDMM Task Plan

## Goal
Implement a Triton validation kernel for the MoE SDDMM stage with an adjustable sparse width, so the dense part width and sparse part width can be tuned independently. The kernel must:

- support increasing sparse width or reducing it down to a pure dense fallback
- assign both dense and sparse work within one launch
- let each block detect whether it is serving the dense or sparse region and run the corresponding path
- use tensor-core-friendly dense compute
- use exact sparse gathers to avoid extra memory traffic
- include correctness and speed tests

## Current Constraint To Remove
The current code ties dense width and sparse width together:

- `moe_src/test/test.py` builds `mask_c` as `t_d + t_d // sp_pd`, with `sp_pd = 1`, so sparse width is forced to equal dense width.
- `moe_test/mlp.py` also allocates `ir`, `mask_v`, and `result` assuming one shared `t_d`.
- `moe_src/mlp_kernel/csrc/cuda/sddmm.cu` and the paired `spmm.cu` interpret the launch metadata with that same coupling.

The first design change is to split this into two independent knobs, for example:

- `dense_width`
- `sparse_width`
- `maxnnz`

## Plan

| Step | Work | Expected Output | Validation |
| --- | --- | --- | --- |
| 1 | [Done] Freeze the new semantics and tensor layout. Define independent `dense_width` and `sparse_width`, decide whether dense and sparse expert indices stay concatenated or are carried as separate tensors, and define the output layout for dense `ir` and sparse `mask_v`. | A small design section in code comments plus a shared metadata builder API, likely in a new Triton-side helper module. | Shape assertions for `mask_c_dense`, `mask_c_sparse`, `mask_r_sparse`, `ir_dense`, and `mask_v_sparse`; verify `sparse_width = 0` is legal. |
| 2 | [Done] Build a pure PyTorch reference oracle for the new behavior. This reference should compute `ref = (x @ up.T) * silu(x @ gate.T)` first, then slice dense columns and sparse sampled rows exactly according to the new metadata. | A reference implementation in a new test/helper file, likely under `moe_src/test/`. | Compare dense slices and sparse sampled slices against the full dense reference for multiple `(bs, dense_width, sparse_width, maxnnz)` combinations. |
| 3 | [Done] Implement a Triton mixed SDDMM prototype. One launch grid covers both dense and sparse regions. Each block derives its region from `program_id`, then dispatches to either the dense path or sparse path. | A new Triton kernel module, likely `moe_src/triton/mixed_sddmm.py`. | For small shapes, compare Triton outputs against the PyTorch oracle with max-abs-diff and mismatch-count checks. Include cases `sparse_width > dense_width`, `sparse_width < dense_width`, and `sparse_width = 0`. |
| 4 | [Done] Implement dense-region block mapping and tensor-core compute. Dense blocks should read contiguous `x` and weight tiles and use Triton `dot` patterns that lower to tensor cores on fp16/bf16-capable GPUs. | Dense path in the Triton kernel plus launch-time block-count calculation. | Inspect output correctness first, then benchmark dense-only cases against the current dense baseline. The acceptance target is that the `sparse_width = 0` mode is not slower than the baseline by an unacceptable margin. |
| 5 | [Done] Implement sparse-region block mapping and exact sparse memory access. Sparse blocks should load only the selected token rows from `mask_r_sparse` and only the selected expert blocks from `mask_c_sparse`, without materializing full dense intermediate tiles. | Sparse path in the Triton kernel plus sparse metadata preparation helpers. | Validate that sparse outputs match the oracle exactly on sampled rows. Confirm no extra dense-sized temporary tensor is allocated for the sparse path. |
| 6 | Build the unified scheduler metadata. Host-side code should compute the number of dense blocks and sparse blocks, pass compact metadata to the Triton kernel, and let blocks self-dispatch. | Metadata builder and launcher wrapper, likely in the same Triton module or a nearby wrapper file. | For small debug cases, print or assert the block-to-region mapping and verify there are no overlapping writes or uncovered tiles. |
| 7 | Add a correctness and speed test harness. The test should generate random inputs and routing, run the reference path and Triton path, and report both numerical error and latency. | A new executable test script, likely `moe_src/test/test_triton_sddmm.py`. | Correctness: max abs diff, relative diff, mismatch ratio. Speed: warmup + timed iterations, median/mean latency, and throughput across several width settings. |
| 8 | Add an integration path for validation use. Wire the Triton launcher behind an explicit dev path so it can be exercised from the existing MoE test harness without replacing the production CUDA kernel immediately. | A guarded integration path, likely a new forward mode or a dev flag in `moe_test/mlp.py` or a dedicated test harness. | Run the existing local benchmark flow with the Triton path enabled and confirm end-to-end execution does not fail for supported batch sizes. |

## Step 1 Completion Notes

Step 1 is complete in the current branch.

- Added a shared metadata helper in `moe_src/sddmm_validation/mixed_sddmm.py`.
- The helper keeps dense and sparse expert indices separate as `mask_c_dense` and `mask_c_sparse`, instead of reusing one concatenated `mask_c`.
- The logical output layout is frozen as:
  - `ir_dense[batch, dense_width, expert_block_size]`
  - `mask_v_sparse[maxnnz, sparse_width, expert_block_size]`
- Added shape/range validation and output buffer allocation helpers around that layout.
- Added `moe_src/test/test_mixed_sddmm_metadata.py` to validate:
  - dense/sparse metadata shapes
  - disjoint dense/sparse expert selections
  - sparse row-topk selection
  - `sparse_width = 0`
- The Step 1 metadata test is now device-parameterized and runs on:
  - CPU
  - CUDA, when available

## Step 2 Completion Notes

Step 2 is complete in the current branch.

- Added a pure PyTorch oracle in `moe_src/sddmm_validation/mixed_sddmm_reference.py`.
- The oracle first materializes the full intermediate as:
  - `full_intermediate[batch, num_experts, expert_block_size] = (x @ up.T) * silu(x @ gate.T)`
- Dense outputs are then sliced as:
  - `ir_dense[:, dense_slot, :] = full_intermediate[:, mask_c_dense[dense_slot], :]`
- Sparse outputs are gathered exactly as:
  - `mask_v_sparse[row_slot, sparse_slot, :] = full_intermediate[mask_r_sparse[row_slot, sparse_slot], mask_c_sparse[sparse_slot], :]`
- Exported the new reference helpers from `moe_src/sddmm_validation/__init__.py`.
- Added `moe_src/test/test_mixed_sddmm_reference.py` to validate multiple `(bs, dense_width, sparse_width, maxnnz)` combinations, including:
  - `sparse_width = 0`
  - `sparse_width < dense_width`
  - `sparse_width > dense_width`
- The Step 2 reference test is now device-parameterized and runs on:
  - CPU
  - CUDA, when available

## Step 3 Completion Notes

Step 3 is complete in the current branch.

- Added a Triton mixed SDDMM validation kernel in:
  - `moe_src/triton_kernels/mixed_sddmm.py`
- The launcher uses one program-id space for both regions:
  - dense programs occupy the first contiguous range
  - sparse programs occupy the following contiguous range
  - each Triton program self-dispatches to the dense or sparse path from `program_id`
- The Triton path writes directly into the frozen Step 1 output layout:
  - `ir_dense[batch, dense_width, expert_block_size]`
  - `mask_v_sparse[maxnnz, sparse_width, expert_block_size]`
- The implementation keeps Step 3 scoped to SDDMM validation only and does not modify the existing CUDA `spmm` integration path.
- The local Triton validation package was placed under `moe_src/triton_kernels/` instead of `moe_src/triton/`.
  - This avoids shadowing the upstream Python `triton` package when `moe_src/` is added to `PYTHONPATH`.
- The Step 1 and Step 2 unit tests were merged into one file and extended with Step 3 validation:
  - `moe_src/test/test_mixed_sddmm.py`
- The Step 3 Triton unit test validates:
  - `sparse_width < dense_width`
  - `sparse_width > dense_width`
  - `sparse_width = 0`
- The Step 3 unit test currently uses scaled fp16 inputs for stable comparison against the PyTorch oracle under the prototype kernel.

## Step 4 Completion Notes

Step 4 is complete in the current branch.

- Dense-region launch metadata is now built explicitly in:
  - `moe_src/triton_kernels/mixed_sddmm.py`
- The dense path uses grouped block scheduling and Triton `tl.dot` tiles intended for tensor-core-friendly fp16/bf16 execution.
- Added a specialized dense-only Triton kernel and launcher:
  - `_dense_only_sddmm_kernel(...)`
  - `launch_dense_only_sddmm_triton(...)`
- The mixed launcher keeps the one-launch path for mixed dense+sparse execution, but when `sparse_width = 0` it now routes to the specialized dense-only kernel automatically.
- The dense-only Triton path computes the full activation output directly:
  - `output = (x @ up.T) * silu(x @ gate.T)`
  - it does not return a raw matmul intermediate
- The dense-only baseline used for benchmarking is the PyTorch implementation of the same selected-dense semantics:
  - `dense_ref = (x @ dense_up_flat.T) * silu(x @ dense_gate_flat.T)`
  - on CUDA, this is expected to lower through PyTorch to cuBLAS/cuBLASLt-backed GEMM paths
- Added dense-only correctness and speed coverage in `moe_src/test/test_mixed_sddmm.py` for:
  - `m = bs`, `expert_n = 512`, `k = 2048`, `dense_width = 4`, so `total_n = 2048`
  - `m = bs`, `expert_n = 512`, `k = 2048`, `dense_width = 16`, so `total_n = 8192`
  - `bs in {32, 64, 128, 256}`
- Added a reproducible dense-only launch-parameter sweep across:
  - `block_m`
  - `block_n`
  - `block_k`
  - `dense_group_m`
  - `num_warps`
  - `num_stages`
- The best observed tuned configs in the current environment are:
  - `total_n = 2048`: `block_m=32, block_n=128, block_k=64, dense_group_m=4, num_warps=8, num_stages=2`
  - `total_n = 8192`: `block_m=64, block_n=128, block_k=32, dense_group_m=4, num_warps=8, num_stages=3`
- Performance observations from the current machine:
  - for `total_n = 2048`, Triton dense-only is still slower than the PyTorch dense baseline by roughly `1.8x` to `2.1x`
  - for `total_n = 8192`, the tuned Triton dense-only path is faster than the PyTorch dense baseline, with ratios around `0.84x` to `0.90x`

## Step 5 Completion Notes

Step 5 is complete in the current branch.

- Added a specialized sparse-only Triton kernel and launcher:
  - `_sparse_only_sddmm_kernel(...)`
  - `launch_sparse_only_sddmm_triton(...)`
- The sparse path in the Triton validation kernel now reads only:
  - selected token rows from `mask_r_sparse`
  - selected sparse expert blocks from `mask_c_sparse`
- The sparse path does not materialize a full dense intermediate tile for sparse experts.
  - In the sparse-only path, `ir_dense` is allocated with logical shape `(batch, 0, expert_block_size)` and zero storage.
- Added a metadata split helper in `moe_src/sddmm_validation/mixed_sddmm.py`:
  - `split_mixed_sddmm_metadata(...)`
  - this produces exact dense-only and sparse-only metadata views from one mixed metadata object
- Added sparse and mixed correctness coverage in `moe_src/test/test_mixed_sddmm.py` for:
  - `dense_width = 0` metadata legality
  - sparse-only Triton vs PyTorch oracle
  - sparse-only specialized kernel vs mixed-kernel sparse subpath
  - ragged mixed shapes
  - mixed kernel vs dense-only and sparse-only specialized subpaths
  - explicit mixed-only launcher vs wrapper consistency
- The sparse and mixed correctness coverage now exercises:
  - sparse-only cases with `hidden_dim = 128`, `expert_block_size = 96`, `sparse_width = 4`, `maxnnz = 8`, and `batch_size in {16, 32, 64}`
  - mixed cases with varying `(dense_width, sparse_width, maxnnz)` and non-uniform `(batch_size, hidden_dim, expert_block_size)`

## Test Runner Notes

- `moe_src/run_test.sh` now treats `moe_src/test/test_mixed_sddmm.py` as the default mixed SDDMM validation test.
- That unified test file covers:
  - Step 1 metadata validation
  - Step 2 PyTorch oracle validation
  - Step 3 Triton-vs-oracle validation
  - Step 4 dense-only correctness and speed validation
  - Step 5 sparse-only and mixed-path correctness validation
- The default unit tests execute on CPU first and also cover CUDA automatically when CUDA is available.
- GPU-backed extension tests remain separate and are only run when CUDA is available.
- `moe_src/test/test_oss.py` is intentionally kept as a legacy reference script, but it is not part of the default regression path and is skipped by `moe_src/run_test.sh`.
- `moe_src/run_test.sh` now hides successful extension build warnings by default.
  - Set `SHOW_BUILD_LOG=1` to print the full compiler output.
  - On build failure, the full build log is replayed automatically.
- `moe_src/run.sh` now accepts:
  - `bash moe_src/run.sh`
    - runs correctness validation only
  - `bash moe_src/run.sh --speed`
    - enables `RUN_MIXED_SDDMM_BENCH=1`
    - enables `RUN_MIXED_SDDMM_TUNE=1`
    - runs dense-only speed benchmarks, mixed-vs-dense `torch.bmm` benchmarks, and the launch-parameter sweep in addition to correctness tests

## Current Validation Status

- `python -m unittest moe_src.test.test_mixed_sddmm` passes.
  - this covers Step 1 through Step 5 in one unit test file
  - when CUDA is available, this exercises the CPU path, the CUDA path, and the Triton validation path
- `RUN_MIXED_SDDMM_BENCH=1 python -m unittest moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_step4_benchmark moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_large_total_n_benchmark` passes.
  - this measures the dense-only Triton path against the PyTorch dense baseline for:
    - `total_n = 2048`
    - `total_n = 8192`
- `RUN_MIXED_SDDMM_BENCH=1 python -m unittest moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_mixed_vs_dense_torch_bmm_benchmark` passes.
  - this measures the mixed Triton path against a dense `torch.bmm` baseline for:
    - `batch_size in {32, 64, 128, 256}`
    - `num_experts = 16`
    - `expert_block_size = 512`
    - `hidden_dim = 2048`
    - split cases `(dense_width, sparse_width) in {(12, 4), (8, 8), (4, 12)}`
    - `maxnnz in {4, 8, 16, 32, batch_size}` after clipping to `<= batch_size`
- `RUN_MIXED_SDDMM_TUNE=1 python -m unittest moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_parameter_sweep` passes.
  - this prints the ranked launch-parameter sweep results for the current GPU
- `CUDA_VISIBLE_DEVICES=-1 bash moe_src/run_test.sh` passes.
  - this confirms the unified mixed SDDMM test file runs correctly in CPU-only mode
  - this also confirms `test/test_oss.py` is reported as a skipped legacy non-default test
- `bash moe_src/run.sh` passes.
  - this rebuilds the extension, runs the unified mixed SDDMM test file, and then runs `moe_src/test/test.py`
- `bash moe_src/run.sh --speed` passes.
  - this runs correctness validation, the Step 4 dense-only benchmarks, the launch-parameter sweep, and the existing GPU-backed integration test path
- `moe_src/test/test.py` was updated to:
  - remove the stale `transformers` dependency
  - replace hardcoded `448` widths with `expert_w`
  - validate both the selected-row oracle and the full-MoE oracle separately
- With `expert_w = 512`, the current CUDA path matches the selected-row oracle closely, while the full-MoE oracle still differs because the current sparse path only keeps `maxnnz` rows per sparse expert by design.

## Suggested File Outputs

- `moe_src/triton_kernels/mixed_sddmm.py`
- `moe_src/triton_kernels/__init__.py`
- `moe_src/test/test_mixed_sddmm.py`
- optional shared metadata helper if the launcher logic becomes non-trivial

## How To Validate

Correctness:

- `python -m unittest moe_src.test.test_mixed_sddmm`
- `bash moe_src/run.sh`

Speed:

- `bash moe_src/run.sh --speed`
- direct benchmark only:
  - `RUN_MIXED_SDDMM_BENCH=1 python -m unittest moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_step4_benchmark moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_large_total_n_benchmark`
  - `RUN_MIXED_SDDMM_BENCH=1 python -m unittest moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_mixed_vs_dense_torch_bmm_benchmark`
- direct sweep only:
  - `RUN_MIXED_SDDMM_TUNE=1 python -m unittest moe_src.test.test_mixed_sddmm.MixedSDDMMTritonTest.test_dense_only_parameter_sweep`

## Validation Matrix

- batch sizes: `32`, `64`, `128`, `256`
- dense widths: small, medium, full-dense fallback
- sparse widths: `0`, smaller than dense, equal to dense, larger than dense
- `maxnnz`: at least `1`, `2`, `4`, `8`
- dtypes: start with `fp16`

For each point in the matrix, validate:

- output correctness against the PyTorch oracle
- stable execution with no shape/layout assertion failures
- latency relative to the current dense baseline

## Exit Criteria

The task is complete when the repository has:

- a Triton SDDMM validation kernel with independently adjustable dense and sparse widths
- unified block scheduling inside one launch
- tensor-core dense compute and exact sparse gathers
- a reproducible correctness and speed test
- a documented dense-only fallback path that can be used when sparse settings are not beneficial

## Notes

- The first validation target should be the SDDMM stage only. Keep the current CUDA `spmm` path unchanged until the new SDDMM layout and scheduling are proven correct.
- If the single-kernel dense/sparse branch causes unacceptable divergence, keep the unified launch prototype for validation but be ready to split it later. That is an optimization fallback, not the first implementation target.
