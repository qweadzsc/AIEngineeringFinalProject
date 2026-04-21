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
from moe_src.triton_kernels import launch_mixed_sddmm_triton


class _DeviceMixin:
    def _devices(self) -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices


class MixedSDDMMMetadataTest(_DeviceMixin, unittest.TestCase):
    def test_build_metadata_splits_dense_and_sparse_regions(self) -> None:
        routing_weights_cpu = torch.tensor(
            [
                [0.6, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.9, 0.2, 0.0, 0.0, 0.0],
                [0.4, 0.8, 0.3, 0.7, 0.0, 0.0],
                [0.3, 0.7, 0.4, 0.6, 0.2, 0.0],
                [0.2, 0.6, 0.5, 0.5, 0.1, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.3],
            ],
            dtype=torch.float32,
        )
        expected_mask_c_dense = torch.tensor([1, 0], dtype=torch.long)
        expected_mask_c_sparse = torch.tensor([2, 3, 4], dtype=torch.long)
        expected_mask_r_sparse = torch.tensor(
            [
                [4, 2, 3],
                [3, 3, 4],
            ],
            dtype=torch.long,
        )

        for device in self._devices():
            with self.subTest(device=device.type):
                routing_weights = routing_weights_cpu.to(device=device)
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=2,
                    sparse_width=3,
                    maxnnz=2,
                    expert_block_size=448,
                )

                self.assertTrue(torch.equal(metadata.mask_c_dense, expected_mask_c_dense.to(device=device)))
                self.assertTrue(torch.equal(metadata.mask_c_sparse, expected_mask_c_sparse.to(device=device)))
                self.assertTrue(torch.equal(metadata.mask_r_sparse, expected_mask_r_sparse.to(device=device)))
                self.assertEqual(metadata.mask_c_dense.device.type, device.type)
                self.assertEqual(metadata.mask_c_sparse.device.type, device.type)
                self.assertEqual(metadata.mask_r_sparse.device.type, device.type)
                self.assertEqual(metadata.ir_dense_shape, (6, 2, 448))
                self.assertEqual(metadata.mask_v_sparse_shape, (2, 3, 448))
                self.assertEqual(metadata.ir_dense_flat_shape, (6, 896))
                self.assertEqual(metadata.mask_v_sparse_flat_shape, (2, 1344))

                ir_dense, mask_v_sparse = metadata.allocate_output_buffers(dtype=torch.float16)
                metadata.assert_output_shapes(ir_dense=ir_dense, mask_v_sparse=mask_v_sparse)
                self.assertEqual(ir_dense.device.type, device.type)
                self.assertEqual(mask_v_sparse.device.type, device.type)

    def test_sparse_width_zero_is_legal(self) -> None:
        routing_weights_cpu = torch.tensor(
            [
                [0.9, 0.2, 0.0, 0.0],
                [0.8, 0.0, 0.4, 0.0],
                [0.7, 0.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        expected_mask_c_dense = torch.tensor([0, 2], dtype=torch.long)

        for device in self._devices():
            with self.subTest(device=device.type):
                routing_weights = routing_weights_cpu.to(device=device)
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=2,
                    sparse_width=0,
                    maxnnz=2,
                    expert_block_size=128,
                )

                self.assertTrue(torch.equal(metadata.mask_c_dense, expected_mask_c_dense.to(device=device)))
                self.assertEqual(metadata.mask_c_dense.device.type, device.type)
                self.assertEqual(metadata.mask_c_sparse.device.type, device.type)
                self.assertEqual(metadata.mask_r_sparse.device.type, device.type)
                self.assertEqual(tuple(metadata.mask_c_sparse.shape), (0,))
                self.assertEqual(tuple(metadata.mask_r_sparse.shape), (2, 0))
                self.assertEqual(metadata.ir_dense_shape, (3, 2, 128))
                self.assertEqual(metadata.mask_v_sparse_shape, (2, 0, 128))

                ir_dense, mask_v_sparse = metadata.allocate_output_buffers(dtype=torch.float16)
                self.assertEqual(ir_dense.device.type, device.type)
                self.assertEqual(mask_v_sparse.device.type, device.type)
                self.assertEqual(tuple(ir_dense.shape), (3, 2, 128))
                self.assertEqual(tuple(mask_v_sparse.shape), (2, 0, 128))


class MixedSDDMMReferenceTest(_DeviceMixin, unittest.TestCase):
    def _build_case_tensors(
        self,
        *,
        batch_size: int,
        num_experts: int,
        hidden_dim: int,
        expert_block_size: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
        up_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=dtype, device=device)
        gate_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=dtype, device=device)

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton validation")
class MixedSDDMMTritonTest(unittest.TestCase):
    def _build_case_tensors(
        self,
        *,
        batch_size: int,
        num_experts: int,
        hidden_dim: int,
        expert_block_size: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        scale = 0.2
        x = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device) * scale
        up_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=torch.float16, device=device) * scale
        gate_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=torch.float16, device=device) * scale

        router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32, device=device)
        router_probs = torch.softmax(router_logits, dim=1)
        values, indices = torch.topk(router_probs, k=min(3, num_experts), dim=1)
        routing_weights = torch.zeros_like(router_probs)
        routing_weights.scatter_(1, indices, values)
        return x, up_proj, gate_proj, routing_weights

    def _diff_stats(self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float) -> tuple[int, float]:
        if actual.numel() == 0:
            return 0, 0.0
        diff = (actual - expected).abs()
        mismatches = int((diff > atol).sum().item())
        max_diff = float(diff.max().item())
        return mismatches, max_diff

    def _run_case(
        self,
        *,
        batch_size: int,
        dense_width: int,
        sparse_width: int,
        maxnnz: int,
    ) -> None:
        num_experts = 6
        hidden_dim = 64
        expert_block_size = 64

        x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
            batch_size=batch_size,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            expert_block_size=expert_block_size,
            seed=(batch_size * 1000) + (dense_width * 100) + (sparse_width * 10) + maxnnz,
        )
        metadata = build_mixed_sddmm_metadata(
            routing_weights,
            dense_width=dense_width,
            sparse_width=sparse_width,
            maxnnz=maxnnz,
            expert_block_size=expert_block_size,
        )
        reference = build_mixed_sddmm_reference(x, up_proj, gate_proj, metadata)
        result = launch_mixed_sddmm_triton(x, up_proj, gate_proj, metadata)
        torch.cuda.synchronize()

        dense_mismatches, dense_max_diff = self._diff_stats(result.ir_dense, reference.ir_dense, atol=5e-2)
        sparse_mismatches, sparse_max_diff = self._diff_stats(
            result.mask_v_sparse,
            reference.mask_v_sparse,
            atol=5e-2,
        )

        self.assertEqual(tuple(result.ir_dense.shape), metadata.ir_dense_shape)
        self.assertEqual(tuple(result.mask_v_sparse.shape), metadata.mask_v_sparse_shape)
        self.assertEqual(result.total_programs, result.dense_programs + result.sparse_programs)
        self.assertGreater(result.dense_programs, 0)
        if sparse_width == 0:
            self.assertEqual(result.sparse_programs, 0)
        else:
            self.assertGreater(result.sparse_programs, 0)

        self.assertEqual(dense_mismatches, 0, msg=f"dense mismatches={dense_mismatches}, max_diff={dense_max_diff}")
        self.assertLessEqual(dense_max_diff, 5e-2)
        self.assertEqual(
            sparse_mismatches,
            0,
            msg=f"sparse mismatches={sparse_mismatches}, max_diff={sparse_max_diff}",
        )
        self.assertLessEqual(sparse_max_diff, 5e-2)

    def test_triton_matches_reference_across_mixed_width_settings(self) -> None:
        cases = [
            {"batch_size": 8, "dense_width": 2, "sparse_width": 1, "maxnnz": 2},
            {"batch_size": 8, "dense_width": 1, "sparse_width": 3, "maxnnz": 2},
            {"batch_size": 8, "dense_width": 3, "sparse_width": 0, "maxnnz": 4},
        ]

        for case in cases:
            with self.subTest(**case):
                self._run_case(**case)


if __name__ == "__main__":
    unittest.main()
