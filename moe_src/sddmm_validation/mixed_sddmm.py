from __future__ import annotations

from dataclasses import dataclass

import torch


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}")


@dataclass(frozen=True)
class MixedSDDMMMetadata:
    """
    Step-1 metadata freeze for the mixed dense/sparse SDDMM validation path.

    The old CUDA path packs dense and sparse expert indices into one `mask_c`.
    For the Triton validation path we split them up explicitly:

    - `mask_c_dense[dense_width]` stores the experts handled by the dense path.
    - `mask_c_sparse[sparse_width]` stores the experts handled by the sparse path.
    - `mask_r_sparse[maxnnz, sparse_width]` stores the selected token rows for each
      sparse expert.

    Logical outputs are also kept separate:

    - `ir_dense[batch, dense_width, expert_block_size]`
    - `mask_v_sparse[maxnnz, sparse_width, expert_block_size]`

    Future launchers can flatten the trailing dimensions at the ABI boundary, but
    this module freezes the logical layout so tests and the reference path share
    one interpretation.
    """

    batch_size: int
    num_experts: int
    expert_block_size: int
    dense_width: int
    sparse_width: int
    maxnnz: int
    mask_c_dense: torch.Tensor
    mask_c_sparse: torch.Tensor
    mask_r_sparse: torch.Tensor

    @property
    def ir_dense_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.dense_width, self.expert_block_size)

    @property
    def mask_v_sparse_shape(self) -> tuple[int, int, int]:
        return (self.maxnnz, self.sparse_width, self.expert_block_size)

    @property
    def ir_dense_flat_shape(self) -> tuple[int, int]:
        return (self.batch_size, self.dense_width * self.expert_block_size)

    @property
    def mask_v_sparse_flat_shape(self) -> tuple[int, int]:
        return (self.maxnnz, self.sparse_width * self.expert_block_size)

    def assert_valid(self) -> None:
        _expect_shape(self.mask_c_dense, (self.dense_width,), "mask_c_dense")
        _expect_shape(self.mask_c_sparse, (self.sparse_width,), "mask_c_sparse")
        _expect_shape(self.mask_r_sparse, (self.maxnnz, self.sparse_width), "mask_r_sparse")

        _check(self.mask_c_dense.dtype == torch.long, "mask_c_dense must be torch.long")
        _check(self.mask_c_sparse.dtype == torch.long, "mask_c_sparse must be torch.long")
        _check(self.mask_r_sparse.dtype == torch.long, "mask_r_sparse must be torch.long")

        if self.dense_width + self.sparse_width > 0:
            merged = torch.cat((self.mask_c_dense, self.mask_c_sparse))
            _check(
                torch.unique(merged).numel() == merged.numel(),
                "dense and sparse expert selections must not overlap",
            )
            _check(
                bool(((merged >= 0) & (merged < self.num_experts)).all().item()),
                "expert indices must stay inside [0, num_experts)",
            )

        if self.mask_r_sparse.numel() > 0:
            _check(
                bool(((self.mask_r_sparse >= 0) & (self.mask_r_sparse < self.batch_size)).all().item()),
                "sparse row indices must stay inside [0, batch_size)",
            )

    def allocate_output_buffers(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.mask_c_dense.device

        ir_dense = torch.zeros(self.ir_dense_shape, dtype=dtype, device=device)
        mask_v_sparse = torch.zeros(self.mask_v_sparse_shape, dtype=dtype, device=device)
        self.assert_output_shapes(ir_dense=ir_dense, mask_v_sparse=mask_v_sparse)
        return ir_dense, mask_v_sparse

    def assert_output_shapes(
        self,
        *,
        ir_dense: torch.Tensor,
        mask_v_sparse: torch.Tensor,
    ) -> None:
        _expect_shape(ir_dense, self.ir_dense_shape, "ir_dense")
        _expect_shape(mask_v_sparse, self.mask_v_sparse_shape, "mask_v_sparse")


def build_mixed_sddmm_metadata(
    routing_weights: torch.Tensor,
    *,
    dense_width: int,
    sparse_width: int,
    maxnnz: int,
    expert_block_size: int,
) -> MixedSDDMMMetadata:
    if routing_weights.dim() != 2:
        raise AssertionError(
            f"routing_weights must be rank-2 [batch, experts], got shape {tuple(routing_weights.shape)}"
        )
    if dense_width < 0 or sparse_width < 0:
        raise AssertionError("dense_width and sparse_width must be non-negative")
    if maxnnz < 0:
        raise AssertionError("maxnnz must be non-negative")
    if expert_block_size <= 0:
        raise AssertionError("expert_block_size must be positive")

    batch_size, num_experts = routing_weights.shape
    if dense_width + sparse_width > num_experts:
        raise AssertionError(
            "dense_width + sparse_width must not exceed the number of experts"
        )
    if maxnnz > batch_size:
        raise AssertionError("maxnnz must not exceed batch_size")

    routing_activity = torch.count_nonzero(routing_weights, dim=0)
    sorted_experts = torch.argsort(routing_activity, descending=True)

    mask_c_dense = sorted_experts[:dense_width].to(dtype=torch.long).contiguous()
    sparse_end = dense_width + sparse_width
    mask_c_sparse = sorted_experts[dense_width:sparse_end].to(dtype=torch.long).contiguous()

    if sparse_width == 0 or maxnnz == 0:
        mask_r_sparse = torch.empty((maxnnz, sparse_width), dtype=torch.long, device=routing_weights.device)
    else:
        sparse_scores = routing_weights[:, mask_c_sparse]
        mask_r_sparse = torch.topk(sparse_scores, k=maxnnz, dim=0, largest=True, sorted=True).indices.contiguous()

    metadata = MixedSDDMMMetadata(
        batch_size=batch_size,
        num_experts=num_experts,
        expert_block_size=expert_block_size,
        dense_width=dense_width,
        sparse_width=sparse_width,
        maxnnz=maxnnz,
        mask_c_dense=mask_c_dense,
        mask_c_sparse=mask_c_sparse,
        mask_r_sparse=mask_r_sparse,
    )
    metadata.assert_valid()
    return metadata


def split_mixed_sddmm_metadata(
    metadata: MixedSDDMMMetadata,
) -> tuple[MixedSDDMMMetadata, MixedSDDMMMetadata]:
    metadata.assert_valid()
    device = metadata.mask_c_dense.device

    dense_only = MixedSDDMMMetadata(
        batch_size=metadata.batch_size,
        num_experts=metadata.num_experts,
        expert_block_size=metadata.expert_block_size,
        dense_width=metadata.dense_width,
        sparse_width=0,
        maxnnz=0,
        mask_c_dense=metadata.mask_c_dense.clone(),
        mask_c_sparse=torch.empty((0,), dtype=torch.long, device=device),
        mask_r_sparse=torch.empty((0, 0), dtype=torch.long, device=device),
    )
    sparse_only = MixedSDDMMMetadata(
        batch_size=metadata.batch_size,
        num_experts=metadata.num_experts,
        expert_block_size=metadata.expert_block_size,
        dense_width=0,
        sparse_width=metadata.sparse_width,
        maxnnz=metadata.maxnnz,
        mask_c_dense=torch.empty((0,), dtype=torch.long, device=device),
        mask_c_sparse=metadata.mask_c_sparse.clone(),
        mask_r_sparse=metadata.mask_r_sparse.clone(),
    )

    dense_only.assert_valid()
    sparse_only.assert_valid()
    return dense_only, sparse_only
