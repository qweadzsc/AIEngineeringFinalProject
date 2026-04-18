from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from moe_src.sddmm_validation import (
    build_full_sddmm_reference,
    build_mixed_sddmm_metadata,
    build_mixed_sddmm_reference,
    slice_mixed_sddmm_reference,
)


class MixedSDDMMReferenceTest(unittest.TestCase):
    def _devices(self) -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

    def _build_case_tensors(
        self,
        *,
        batch_size: int,
        num_experts: int,
        hidden_dim: int,
        expert_block_size: int,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device=device)
        up_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=torch.float32, device=device)
        gate_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=torch.float32, device=device)

        router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32, device=device)
        router_probs = torch.softmax(router_logits, dim=1)
        topk = min(3, num_experts)
        values, indices = torch.topk(router_probs, k=topk, dim=1)
        routing_weights = torch.zeros_like(router_probs)
        routing_weights.scatter_(1, indices, values)

        return x, up_proj, gate_proj, routing_weights

    def _build_expected_dense(
        self,
        full_intermediate: torch.Tensor,
        dense_experts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, expert_block_size = full_intermediate.shape
        expected = torch.empty(
            (batch_size, dense_experts.numel(), expert_block_size),
            dtype=full_intermediate.dtype,
            device=full_intermediate.device,
        )
        for dense_slot, expert_idx in enumerate(dense_experts.cpu().tolist()):
            expected[:, dense_slot, :] = full_intermediate[:, expert_idx, :]
        return expected

    def _build_expected_sparse(
        self,
        full_intermediate: torch.Tensor,
        mask_c_sparse: torch.Tensor,
        mask_r_sparse: torch.Tensor,
    ) -> torch.Tensor:
        maxnnz, sparse_width = tuple(mask_r_sparse.shape)
        expert_block_size = full_intermediate.shape[-1]
        expected = torch.empty(
            (maxnnz, sparse_width, expert_block_size),
            dtype=full_intermediate.dtype,
            device=full_intermediate.device,
        )
        for sparse_slot, expert_idx in enumerate(mask_c_sparse.cpu().tolist()):
            for row_slot in range(maxnnz):
                row_idx = int(mask_r_sparse[row_slot, sparse_slot].item())
                expected[row_slot, sparse_slot, :] = full_intermediate[row_idx, expert_idx, :]
        return expected

    def _run_case(
        self,
        *,
        batch_size: int,
        dense_width: int,
        sparse_width: int,
        maxnnz: int,
        device: torch.device,
    ) -> None:
        num_experts = 6
        hidden_dim = 5
        expert_block_size = 8

        x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
            batch_size=batch_size,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            expert_block_size=expert_block_size,
            seed=(batch_size * 1000) + (dense_width * 100) + (sparse_width * 10) + maxnnz,
            device=device,
        )
        metadata = build_mixed_sddmm_metadata(
            routing_weights,
            dense_width=dense_width,
            sparse_width=sparse_width,
            maxnnz=maxnnz,
            expert_block_size=expert_block_size,
        )

        reference = build_mixed_sddmm_reference(x, up_proj, gate_proj, metadata)
        full_intermediate = build_full_sddmm_reference(
            x,
            up_proj,
            gate_proj,
            num_experts=num_experts,
            expert_block_size=expert_block_size,
        )
        sliced_dense, sliced_sparse = slice_mixed_sddmm_reference(full_intermediate, metadata)

        self.assertEqual(tuple(reference.full_intermediate.shape), (batch_size, num_experts, expert_block_size))
        self.assertEqual(tuple(reference.full_intermediate_flat.shape), (batch_size, num_experts * expert_block_size))
        self.assertEqual(tuple(reference.ir_dense.shape), metadata.ir_dense_shape)
        self.assertEqual(tuple(reference.mask_v_sparse.shape), metadata.mask_v_sparse_shape)
        self.assertEqual(reference.full_intermediate.device.type, device.type)
        self.assertEqual(reference.ir_dense.device.type, device.type)
        self.assertEqual(reference.mask_v_sparse.device.type, device.type)

        expected_dense = self._build_expected_dense(full_intermediate, metadata.mask_c_dense)
        expected_sparse = self._build_expected_sparse(
            full_intermediate,
            metadata.mask_c_sparse,
            metadata.mask_r_sparse,
        )

        self.assertTrue(torch.allclose(reference.full_intermediate, full_intermediate))
        self.assertTrue(torch.allclose(reference.ir_dense, expected_dense))
        self.assertTrue(torch.allclose(reference.mask_v_sparse, expected_sparse))
        self.assertTrue(torch.allclose(reference.ir_dense, sliced_dense))
        self.assertTrue(torch.allclose(reference.mask_v_sparse, sliced_sparse))

    def test_reference_matches_full_reference_slices_across_width_settings(self) -> None:
        cases = [
            {"batch_size": 4, "dense_width": 2, "sparse_width": 1, "maxnnz": 2},
            {"batch_size": 5, "dense_width": 1, "sparse_width": 3, "maxnnz": 2},
            {"batch_size": 6, "dense_width": 3, "sparse_width": 0, "maxnnz": 4},
            {"batch_size": 6, "dense_width": 1, "sparse_width": 4, "maxnnz": 3},
        ]

        for device in self._devices():
            for case in cases:
                with self.subTest(device=device.type, **case):
                    self._run_case(device=device, **case)


if __name__ == "__main__":
    unittest.main()
