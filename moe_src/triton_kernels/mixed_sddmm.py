from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_src.sddmm_validation import MixedSDDMMMetadata

import triton
import triton.language as tl


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class MixedSDDMMTritonResult:
    ir_dense: torch.Tensor
    mask_v_sparse: torch.Tensor
    dense_programs: int
    sparse_programs: int

    @property
    def total_programs(self) -> int:
        return self.dense_programs + self.sparse_programs


if triton is not None:

    @triton.jit
    def _mixed_sddmm_kernel(
        x_ptr,
        up_ptr,
        gate_ptr,
        mask_c_dense_ptr,
        mask_c_sparse_ptr,
        mask_r_sparse_ptr,
        ir_ptr,
        mask_v_ptr,
        batch_size,
        hidden_dim,
        expert_block_size,
        dense_width,
        sparse_width,
        maxnnz,
        dense_tiles_m,
        sparse_tiles_m,
        tiles_n,
        stride_x_batch,
        stride_x_hidden,
        stride_proj_out,
        stride_proj_hidden,
        stride_mask_r_row,
        stride_mask_r_col,
        stride_ir_batch,
        stride_ir_width,
        stride_ir_col,
        stride_mask_v_row,
        stride_mask_v_width,
        stride_mask_v_col,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dense_region_programs = dense_width * dense_tiles_m * tiles_n

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        if pid < dense_region_programs:
            programs_per_expert = dense_tiles_m * tiles_n
            dense_slot = pid // programs_per_expert
            local_pid = pid % programs_per_expert
            tile_m = local_pid // tiles_n
            tile_n = local_pid % tiles_n

            row_ids = tile_m * BLOCK_M + offs_m
            col_ids = tile_n * BLOCK_N + offs_n
            row_mask = row_ids < batch_size
            col_mask = col_ids < expert_block_size

            token_rows = row_ids
            expert_idx = tl.load(mask_c_dense_ptr + dense_slot)
            output_ptr = (
                ir_ptr
                + row_ids[:, None] * stride_ir_batch
                + dense_slot * stride_ir_width
                + col_ids[None, :] * stride_ir_col
            )
        else:
            sparse_pid = pid - dense_region_programs
            programs_per_expert = sparse_tiles_m * tiles_n
            sparse_slot = sparse_pid // programs_per_expert
            local_pid = sparse_pid % programs_per_expert
            tile_m = local_pid // tiles_n
            tile_n = local_pid % tiles_n

            row_ids = tile_m * BLOCK_M + offs_m
            col_ids = tile_n * BLOCK_N + offs_n
            row_mask = row_ids < maxnnz
            col_mask = col_ids < expert_block_size

            token_rows = tl.load(
                mask_r_sparse_ptr + row_ids * stride_mask_r_row + sparse_slot * stride_mask_r_col,
                mask=row_mask,
                other=0,
            )
            expert_idx = tl.load(mask_c_sparse_ptr + sparse_slot)
            output_ptr = (
                mask_v_ptr
                + row_ids[:, None] * stride_mask_v_row
                + sparse_slot * stride_mask_v_width
                + col_ids[None, :] * stride_mask_v_col
            )

        expert_offset = expert_idx * expert_block_size
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, hidden_dim, BLOCK_K):
            k_ids = k_start + offs_k
            k_mask = k_ids < hidden_dim

            x_tile = tl.load(
                x_ptr + token_rows[:, None] * stride_x_batch + k_ids[None, :] * stride_x_hidden,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            up_tile = tl.load(
                up_ptr
                + (expert_offset + col_ids[None, :]) * stride_proj_out
                + k_ids[:, None] * stride_proj_hidden,
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            gate_tile = tl.load(
                gate_ptr
                + (expert_offset + col_ids[None, :]) * stride_proj_out
                + k_ids[:, None] * stride_proj_hidden,
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            up_acc += tl.dot(x_tile, up_tile)
            gate_acc += tl.dot(x_tile, gate_tile)

        output = up_acc * (gate_acc * tl.sigmoid(gate_acc))
        tl.store(output_ptr, output, mask=row_mask[:, None] & col_mask[None, :])


def launch_mixed_sddmm_triton(
    x: torch.Tensor,
    up_proj: torch.Tensor,
    gate_proj: torch.Tensor,
    metadata: MixedSDDMMMetadata,
    *,
    block_m: int = 16,
    block_n: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> MixedSDDMMTritonResult:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise AssertionError("x must be a CUDA tensor")

    metadata.assert_valid()
    _check(x.dim() == 2, f"x must be rank-2 [batch, hidden_dim], got shape {tuple(x.shape)}")
    _check(up_proj.dim() == 2, f"up_proj must be rank-2 [num_experts * expert_block_size, hidden_dim]")
    _check(gate_proj.dim() == 2, f"gate_proj must be rank-2 [num_experts * expert_block_size, hidden_dim]")
    _check(x.dtype in (torch.float16, torch.bfloat16), "Triton prototype currently supports fp16/bf16 inputs")
    _check(up_proj.dtype == x.dtype, "up_proj dtype must match x dtype")
    _check(gate_proj.dtype == x.dtype, "gate_proj dtype must match x dtype")
    _check(x.device == up_proj.device == gate_proj.device, "x, up_proj, and gate_proj must be on the same device")
    _check(metadata.mask_c_dense.device == x.device, "metadata tensors must be on the same CUDA device as x")
    _check(metadata.mask_c_sparse.device == x.device, "metadata tensors must be on the same CUDA device as x")
    _check(metadata.mask_r_sparse.device == x.device, "metadata tensors must be on the same CUDA device as x")
    _check(x.is_contiguous(), "x must be contiguous")
    _check(up_proj.is_contiguous(), "up_proj must be contiguous")
    _check(gate_proj.is_contiguous(), "gate_proj must be contiguous")

    batch_size, hidden_dim = x.shape
    expected_proj_shape = (metadata.num_experts * metadata.expert_block_size, hidden_dim)
    _check(tuple(up_proj.shape) == expected_proj_shape, f"up_proj shape mismatch: expected {expected_proj_shape}")
    _check(tuple(gate_proj.shape) == expected_proj_shape, f"gate_proj shape mismatch: expected {expected_proj_shape}")

    mask_c_dense = metadata.mask_c_dense.contiguous().to(dtype=torch.int32)
    mask_c_sparse = metadata.mask_c_sparse.contiguous().to(dtype=torch.int32)
    mask_r_sparse = metadata.mask_r_sparse.contiguous().to(dtype=torch.int32)
    ir_dense, mask_v_sparse = metadata.allocate_output_buffers(dtype=x.dtype, device=x.device)

    tiles_n = _ceil_div(metadata.expert_block_size, block_n)
    dense_tiles_m = _ceil_div(metadata.batch_size, block_m) if metadata.dense_width > 0 else 0
    sparse_tiles_m = _ceil_div(metadata.maxnnz, block_m) if metadata.sparse_width > 0 and metadata.maxnnz > 0 else 0
    dense_programs = metadata.dense_width * dense_tiles_m * tiles_n
    sparse_programs = metadata.sparse_width * sparse_tiles_m * tiles_n
    total_programs = dense_programs + sparse_programs

    if total_programs == 0:
        return MixedSDDMMTritonResult(
            ir_dense=ir_dense,
            mask_v_sparse=mask_v_sparse,
            dense_programs=dense_programs,
            sparse_programs=sparse_programs,
        )

    grid = (total_programs,)
    _mixed_sddmm_kernel[grid](
        x,
        up_proj,
        gate_proj,
        mask_c_dense,
        mask_c_sparse,
        mask_r_sparse,
        ir_dense,
        mask_v_sparse,
        metadata.batch_size,
        hidden_dim,
        metadata.expert_block_size,
        metadata.dense_width,
        metadata.sparse_width,
        metadata.maxnnz,
        dense_tiles_m,
        sparse_tiles_m,
        tiles_n,
        x.stride(0),
        x.stride(1),
        up_proj.stride(0),
        up_proj.stride(1),
        mask_r_sparse.stride(0),
        mask_r_sparse.stride(1),
        ir_dense.stride(0),
        ir_dense.stride(1),
        ir_dense.stride(2),
        mask_v_sparse.stride(0),
        mask_v_sparse.stride(1),
        mask_v_sparse.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    metadata.assert_output_shapes(ir_dense=ir_dense, mask_v_sparse=mask_v_sparse)
    return MixedSDDMMTritonResult(
        ir_dense=ir_dense,
        mask_v_sparse=mask_v_sparse,
        dense_programs=dense_programs,
        sparse_programs=sparse_programs,
    )
