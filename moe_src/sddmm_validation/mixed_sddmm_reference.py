from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .mixed_sddmm import MixedSDDMMMetadata


def _expect_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}")


@dataclass(frozen=True)
class MixedSDDMMReference:
    full_intermediate: torch.Tensor
    ir_dense: torch.Tensor
    mask_v_sparse: torch.Tensor

    @property
    def full_intermediate_flat(self) -> torch.Tensor:
        return self.full_intermediate.reshape(self.full_intermediate.shape[0], -1)

    def assert_matches_metadata(self, metadata: MixedSDDMMMetadata) -> None:
        _expect_shape(
            self.full_intermediate,
            (metadata.batch_size, metadata.num_experts, metadata.expert_block_size),
            "full_intermediate",
        )
        metadata.assert_output_shapes(ir_dense=self.ir_dense, mask_v_sparse=self.mask_v_sparse)


def build_full_sddmm_reference(
    x: torch.Tensor,
    up_proj: torch.Tensor,
    gate_proj: torch.Tensor,
    *,
    num_experts: int,
    expert_block_size: int,
) -> torch.Tensor:
    if x.dim() != 2:
        raise AssertionError(f"x must be rank-2 [batch, hidden_dim], got shape {tuple(x.shape)}")
    if expert_block_size <= 0:
        raise AssertionError("expert_block_size must be positive")
    if num_experts < 0:
        raise AssertionError("num_experts must be non-negative")

    batch_size, hidden_dim = x.shape
    expected_proj_shape = (num_experts * expert_block_size, hidden_dim)
    _expect_shape(up_proj, expected_proj_shape, "up_proj")
    _expect_shape(gate_proj, expected_proj_shape, "gate_proj")

    full_intermediate = (x @ up_proj.T) * F.silu(x @ gate_proj.T)
    return full_intermediate.view(batch_size, num_experts, expert_block_size).contiguous()


def slice_mixed_sddmm_reference(
    full_intermediate: torch.Tensor,
    metadata: MixedSDDMMMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata.assert_valid()
    _expect_shape(
        full_intermediate,
        (metadata.batch_size, metadata.num_experts, metadata.expert_block_size),
        "full_intermediate",
    )

    device = full_intermediate.device
    dense_index = metadata.mask_c_dense.to(device=device)
    ir_dense = full_intermediate.index_select(1, dense_index).contiguous()

    mask_v_sparse = torch.empty(
        metadata.mask_v_sparse_shape,
        dtype=full_intermediate.dtype,
        device=device,
    )
    for sparse_slot, expert_idx in enumerate(metadata.mask_c_sparse.cpu().tolist()):
        row_index = metadata.mask_r_sparse[:, sparse_slot].to(device=device)
        expert_values = full_intermediate[:, expert_idx, :]
        mask_v_sparse[:, sparse_slot, :] = expert_values.index_select(0, row_index)

    metadata.assert_output_shapes(ir_dense=ir_dense, mask_v_sparse=mask_v_sparse)
    return ir_dense, mask_v_sparse


def build_mixed_sddmm_reference(
    x: torch.Tensor,
    up_proj: torch.Tensor,
    gate_proj: torch.Tensor,
    metadata: MixedSDDMMMetadata,
) -> MixedSDDMMReference:
    full_intermediate = build_full_sddmm_reference(
        x,
        up_proj,
        gate_proj,
        num_experts=metadata.num_experts,
        expert_block_size=metadata.expert_block_size,
    )
    ir_dense, mask_v_sparse = slice_mixed_sddmm_reference(full_intermediate, metadata)

    reference = MixedSDDMMReference(
        full_intermediate=full_intermediate,
        ir_dense=ir_dense,
        mask_v_sparse=mask_v_sparse,
    )
    reference.assert_matches_metadata(metadata)
    return reference
