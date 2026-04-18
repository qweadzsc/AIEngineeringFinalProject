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
| 2 | Build a pure PyTorch reference oracle for the new behavior. This reference should compute `ref = (x @ up.T) * silu(x @ gate.T)` first, then slice dense columns and sparse sampled rows exactly according to the new metadata. | A reference implementation in a new test/helper file, likely under `moe_src/test/`. | Compare dense slices and sparse sampled slices against the full dense reference for multiple `(bs, dense_width, sparse_width, maxnnz)` combinations. |
| 3 | Implement a Triton mixed SDDMM prototype. One launch grid covers both dense and sparse regions. Each block derives its region from `program_id`, then dispatches to either the dense path or sparse path. | A new Triton kernel module, likely `moe_src/triton/mixed_sddmm.py`. | For small shapes, compare Triton outputs against the PyTorch oracle with max-abs-diff and mismatch-count checks. Include cases `sparse_width > dense_width`, `sparse_width < dense_width`, and `sparse_width = 0`. |
| 4 | Implement dense-region block mapping and tensor-core compute. Dense blocks should read contiguous `x` and weight tiles and use Triton `dot` patterns that lower to tensor cores on fp16/bf16-capable GPUs. | Dense path in the Triton kernel plus launch-time block-count calculation. | Inspect output correctness first, then benchmark dense-only cases against the current dense baseline. The acceptance target is that the `sparse_width = 0` mode is not slower than the baseline by an unacceptable margin. |
| 5 | Implement sparse-region block mapping and exact sparse memory access. Sparse blocks should load only the selected token rows from `mask_r_sparse` and only the selected expert blocks from `mask_c_sparse`, without materializing full dense intermediate tiles. | Sparse path in the Triton kernel plus sparse metadata preparation helpers. | Validate that sparse outputs match the oracle exactly on sampled rows. Confirm no extra dense-sized temporary tensor is allocated for the sparse path. |
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

## Current Validation Status

- `python moe_src/test/test_mixed_sddmm_metadata.py` passes.
- `bash moe_src/run.sh` now defaults to `CUDA_VISIBLE_DEVICES=6`, rebuilds the extension, and runs the active test set.
- `moe_src/test/test.py` was updated to:
  - remove the stale `transformers` dependency
  - replace hardcoded `448` widths with `expert_w`
  - validate both the selected-row oracle and the full-MoE oracle separately
- With `expert_w = 512`, the current CUDA path matches the selected-row oracle closely, while the full-MoE oracle still differs because the current sparse path only keeps `maxnnz` rows per sparse expert by design.

## Suggested File Outputs

- `moe_src/triton/mixed_sddmm.py`
- `moe_src/triton/__init__.py`
- `moe_src/test/test_triton_sddmm.py`
- optional shared metadata helper if the launcher logic becomes non-trivial

## Validation Matrix

- batch sizes: `32`, `64`, `128`
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
