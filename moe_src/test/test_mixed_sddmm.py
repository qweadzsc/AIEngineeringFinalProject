from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from moe_src.sddmm_validation import (
    build_full_sddmm_reference,
    build_mixed_sddmm_metadata,
    build_mixed_sddmm_reference,
    slice_mixed_sddmm_reference,
)
from moe_src.triton_kernels import (
    build_mixed_sddmm_launch_metadata,
    launch_dense_only_sddmm_triton,
    launch_mixed_sddmm_triton,
)


DENSE_ONLY_TUNING_CONFIGS = [
    {"block_m": 32, "block_n": 64, "block_k": 32, "dense_group_m": 4, "num_warps": 4, "num_stages": 2},
    {"block_m": 32, "block_n": 128, "block_k": 32, "dense_group_m": 4, "num_warps": 4, "num_stages": 2},
    {"block_m": 32, "block_n": 128, "block_k": 64, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 64, "block_k": 32, "dense_group_m": 4, "num_warps": 4, "num_stages": 2},
    {"block_m": 64, "block_n": 64, "block_k": 64, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 128, "block_k": 32, "dense_group_m": 4, "num_warps": 4, "num_stages": 2},
    {"block_m": 64, "block_n": 128, "block_k": 32, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 128, "block_k": 64, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 256, "block_k": 32, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 256, "block_k": 64, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 128, "block_n": 64, "block_k": 32, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 128, "block_n": 128, "block_k": 32, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 128, "block_n": 128, "block_k": 64, "dense_group_m": 4, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 128, "block_k": 32, "dense_group_m": 1, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 128, "block_k": 32, "dense_group_m": 8, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 256, "block_k": 32, "dense_group_m": 1, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 256, "block_k": 32, "dense_group_m": 8, "num_warps": 8, "num_stages": 2},
    {"block_m": 64, "block_n": 128, "block_k": 32, "dense_group_m": 4, "num_warps": 8, "num_stages": 3},
    {"block_m": 64, "block_n": 256, "block_k": 32, "dense_group_m": 4, "num_warps": 8, "num_stages": 3},
]

TUNED_SMALL_TOTAL_N_CONFIG = {
    "block_m": 32,
    "block_n": 128,
    "block_k": 64,
    "dense_group_m": 4,
    "num_warps": 8,
    "num_stages": 2,
}

TUNED_LARGE_TOTAL_N_CONFIG = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 32,
    "dense_group_m": 4,
    "num_warps": 8,
    "num_stages": 3,
}


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
        scale: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        x = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device=device) * scale
        up_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=torch.float16, device=device) * scale
        gate_proj = torch.randn(num_experts * expert_block_size, hidden_dim, dtype=torch.float16, device=device) * scale

        router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32, device=device)
        router_probs = torch.softmax(router_logits, dim=1)
        values, indices = torch.topk(router_probs, k=min(3, num_experts), dim=1)
        routing_weights = torch.zeros_like(router_probs)
        routing_weights.scatter_(1, indices, values)
        return x, up_proj, gate_proj, routing_weights

    def _build_selected_dense_baseline(
        self,
        x: torch.Tensor,
        up_proj: torch.Tensor,
        gate_proj: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        hidden_dim = x.shape[1]
        dense_up = up_proj.view(metadata.num_experts, metadata.expert_block_size, hidden_dim).index_select(
            0, metadata.mask_c_dense
        )
        dense_gate = gate_proj.view(metadata.num_experts, metadata.expert_block_size, hidden_dim).index_select(
            0, metadata.mask_c_dense
        )
        dense_up_flat = dense_up.reshape(metadata.dense_width * metadata.expert_block_size, hidden_dim)
        dense_gate_flat = dense_gate.reshape(metadata.dense_width * metadata.expert_block_size, hidden_dim)
        dense_ref = (x @ dense_up_flat.T) * F.silu(x @ dense_gate_flat.T)
        return dense_ref.view(metadata.batch_size, metadata.dense_width, metadata.expert_block_size).contiguous()

    def _diff_stats(self, actual: torch.Tensor, expected: torch.Tensor, *, atol: float) -> tuple[int, float]:
        if actual.numel() == 0:
            return 0, 0.0
        diff = (actual - expected).abs()
        mismatches = int((diff > atol).sum().item())
        max_diff = float(diff.max().item())
        return mismatches, max_diff

    def _measure_cuda_ms(
        self,
        fn,
        *,
        warmup: int = 20,
        iters: int = 50,
    ) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            start_events[i].record()
            fn()
            end_events[i].record()
        torch.cuda.synchronize()

        timings = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
        timings.sort()
        return float(timings[len(timings) // 2])

    def _measure_dense_only_ms(
        self,
        x: torch.Tensor,
        up_proj: torch.Tensor,
        gate_proj: torch.Tensor,
        metadata,
        *,
        launch_kwargs: dict | None = None,
    ) -> float:
        launch_kwargs = launch_kwargs or {}
        return self._measure_cuda_ms(
            lambda: launch_dense_only_sddmm_triton(x, up_proj, gate_proj, metadata, **launch_kwargs).ir_dense
        )

    def _verify_dense_only_config(
        self,
        x: torch.Tensor,
        up_proj: torch.Tensor,
        gate_proj: torch.Tensor,
        metadata,
        expected_dense: torch.Tensor,
        *,
        launch_kwargs: dict | None = None,
    ) -> None:
        launch_kwargs = launch_kwargs or {}
        result = launch_dense_only_sddmm_triton(x, up_proj, gate_proj, metadata, **launch_kwargs)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(result.ir_dense, expected_dense, atol=5e-2, rtol=5e-2))

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

    def test_dense_only_triton_matches_selected_dense_baseline_across_step4_batch_sizes(self) -> None:
        num_experts = 8
        hidden_dim = 2048
        expert_block_size = 512
        dense_width = 4
        sparse_width = 0
        maxnnz = 0

        for batch_size in [32, 64, 128, 256]:
            with self.subTest(batch_size=batch_size):
                x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
                    batch_size=batch_size,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    expert_block_size=expert_block_size,
                    seed=9000 + batch_size,
                    scale=0.05,
                )
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=dense_width,
                    sparse_width=sparse_width,
                    maxnnz=maxnnz,
                    expert_block_size=expert_block_size,
                )
                expected_dense = self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                result = launch_mixed_sddmm_triton(x, up_proj, gate_proj, metadata)
                torch.cuda.synchronize()

                dense_mismatches, dense_max_diff = self._diff_stats(result.ir_dense, expected_dense, atol=5e-2)
                self.assertEqual(dense_mismatches, 0, msg=f"dense mismatches={dense_mismatches}, max_diff={dense_max_diff}")
                self.assertLessEqual(dense_max_diff, 5e-2)
                self.assertEqual(result.sparse_programs, 0)
                self.assertEqual(result.kernel_variant, "dense_only")
                self.assertEqual(result.launch_metadata.block_m, 32)
                self.assertEqual(result.launch_metadata.block_n, 64)
                self.assertEqual(result.launch_metadata.block_k, 32)
                self.assertGreaterEqual(result.launch_metadata.dense_group_m, 1)
                self.assertEqual(
                    result.dense_programs,
                    dense_width
                    * result.launch_metadata.dense_tiles_m
                    * result.launch_metadata.tiles_n,
                )

                launch_metadata = build_mixed_sddmm_launch_metadata(
                    metadata,
                    block_m=32,
                    block_n=64,
                    block_k=32,
                    dense_group_m=4,
                    num_warps=4,
                    num_stages=2,
                )
                self.assertEqual(result.launch_metadata, launch_metadata)

    def test_dense_only_specialized_kernel_matches_mixed_kernel(self) -> None:
        num_experts = 8
        hidden_dim = 2048
        expert_block_size = 512
        dense_width = 4
        sparse_width = 0
        maxnnz = 0

        for batch_size in [32, 64, 128, 256]:
            with self.subTest(batch_size=batch_size):
                x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
                    batch_size=batch_size,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    expert_block_size=expert_block_size,
                    seed=15000 + batch_size,
                    scale=0.05,
                )
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=dense_width,
                    sparse_width=sparse_width,
                    maxnnz=maxnnz,
                    expert_block_size=expert_block_size,
                )
                expected_dense = self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                mixed_result = launch_mixed_sddmm_triton(
                    x,
                    up_proj,
                    gate_proj,
                    metadata,
                    use_dense_only_fast_path=False,
                )
                dense_only_result = launch_dense_only_sddmm_triton(x, up_proj, gate_proj, metadata)
                torch.cuda.synchronize()

                self.assertEqual(mixed_result.kernel_variant, "mixed")
                self.assertEqual(dense_only_result.kernel_variant, "dense_only")
                self.assertTrue(torch.allclose(mixed_result.ir_dense, expected_dense, atol=5e-2, rtol=5e-2))
                self.assertTrue(torch.allclose(dense_only_result.ir_dense, expected_dense, atol=5e-2, rtol=5e-2))
                self.assertTrue(torch.allclose(dense_only_result.ir_dense, mixed_result.ir_dense, atol=5e-2, rtol=5e-2))
                self.assertEqual(dense_only_result.launch_metadata, mixed_result.launch_metadata)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.getenv("RUN_MIXED_SDDMM_BENCH", "0") == "1",
        "Set RUN_MIXED_SDDMM_BENCH=1 to run the dense-only Step 4 benchmark",
    )
    def test_dense_only_step4_benchmark(self) -> None:
        num_experts = 8
        hidden_dim = 2048
        expert_block_size = 512
        dense_width = 4
        sparse_width = 0
        maxnnz = 0

        for batch_size in [32, 64, 128, 256]:
            with self.subTest(batch_size=batch_size):
                x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
                    batch_size=batch_size,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    expert_block_size=expert_block_size,
                    seed=12000 + batch_size,
                    scale=0.05,
                )
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=dense_width,
                    sparse_width=sparse_width,
                    maxnnz=maxnnz,
                    expert_block_size=expert_block_size,
                )
                expected_dense = self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                mixed_result = launch_mixed_sddmm_triton(
                    x,
                    up_proj,
                    gate_proj,
                    metadata,
                    use_dense_only_fast_path=False,
                )
                dense_only_result = launch_dense_only_sddmm_triton(x, up_proj, gate_proj, metadata)
                torch.cuda.synchronize()
                self.assertTrue(torch.allclose(mixed_result.ir_dense, expected_dense, atol=5e-2, rtol=5e-2))
                self.assertTrue(torch.allclose(dense_only_result.ir_dense, expected_dense, atol=5e-2, rtol=5e-2))
                self._verify_dense_only_config(
                    x,
                    up_proj,
                    gate_proj,
                    metadata,
                    expected_dense,
                    launch_kwargs=TUNED_SMALL_TOTAL_N_CONFIG,
                )

                baseline_ms = self._measure_cuda_ms(
                    lambda: self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                )
                mixed_ms = self._measure_cuda_ms(
                    lambda: launch_mixed_sddmm_triton(
                        x,
                        up_proj,
                        gate_proj,
                        metadata,
                        use_dense_only_fast_path=False,
                    ).ir_dense
                )
                dense_only_ms = self._measure_cuda_ms(
                    lambda: launch_dense_only_sddmm_triton(x, up_proj, gate_proj, metadata).ir_dense
                )
                tuned_ms = self._measure_dense_only_ms(
                    x,
                    up_proj,
                    gate_proj,
                    metadata,
                    launch_kwargs=TUNED_SMALL_TOTAL_N_CONFIG,
                )
                mixed_ratio = mixed_ms / baseline_ms
                dense_only_ratio = dense_only_ms / baseline_ms
                tuned_ratio = tuned_ms / baseline_ms
                print(
                    f"[step4-bench] m={batch_size} expert_n={expert_block_size} total_n={dense_width * expert_block_size} "
                    f"k={hidden_dim} dense_width={dense_width} baseline_ms={baseline_ms:.3f} "
                    f"mixed_ms={mixed_ms:.3f} dense_only_ms={dense_only_ms:.3f} tuned_ms={tuned_ms:.3f} "
                    f"mixed_ratio={mixed_ratio:.3f} dense_only_ratio={dense_only_ratio:.3f} "
                    f"tuned_ratio={tuned_ratio:.3f} tuned_cfg={TUNED_SMALL_TOTAL_N_CONFIG}"
                )
                self.assertGreater(baseline_ms, 0.0)
                self.assertGreater(mixed_ms, 0.0)
                self.assertGreater(dense_only_ms, 0.0)
                self.assertGreater(tuned_ms, 0.0)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.getenv("RUN_MIXED_SDDMM_BENCH", "0") == "1",
        "Set RUN_MIXED_SDDMM_BENCH=1 to run the larger total-N dense-only benchmark",
    )
    def test_dense_only_large_total_n_benchmark(self) -> None:
        num_experts = 24
        hidden_dim = 2048
        expert_block_size = 512
        dense_width = 16
        sparse_width = 0
        maxnnz = 0

        for batch_size in [32, 64, 128, 256]:
            with self.subTest(batch_size=batch_size):
                x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
                    batch_size=batch_size,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    expert_block_size=expert_block_size,
                    seed=18000 + batch_size,
                    scale=0.05,
                )
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=dense_width,
                    sparse_width=sparse_width,
                    maxnnz=maxnnz,
                    expert_block_size=expert_block_size,
                )
                expected_dense = self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                self._verify_dense_only_config(x, up_proj, gate_proj, metadata, expected_dense)
                self._verify_dense_only_config(
                    x,
                    up_proj,
                    gate_proj,
                    metadata,
                    expected_dense,
                    launch_kwargs=TUNED_LARGE_TOTAL_N_CONFIG,
                )

                baseline_ms = self._measure_cuda_ms(
                    lambda: self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                )
                default_ms = self._measure_dense_only_ms(x, up_proj, gate_proj, metadata)
                tuned_ms = self._measure_dense_only_ms(
                    x,
                    up_proj,
                    gate_proj,
                    metadata,
                    launch_kwargs=TUNED_LARGE_TOTAL_N_CONFIG,
                )
                default_ratio = default_ms / baseline_ms
                tuned_ratio = tuned_ms / baseline_ms
                print(
                    f"[step4-large-n-bench] m={batch_size} expert_n={expert_block_size} "
                    f"total_n={dense_width * expert_block_size} k={hidden_dim} dense_width={dense_width} "
                    f"baseline_ms={baseline_ms:.3f} default_ms={default_ms:.3f} tuned_ms={tuned_ms:.3f} "
                    f"default_ratio={default_ratio:.3f} tuned_ratio={tuned_ratio:.3f} "
                    f"tuned_cfg={TUNED_LARGE_TOTAL_N_CONFIG}"
                )
                self.assertGreater(baseline_ms, 0.0)
                self.assertGreater(default_ms, 0.0)
                self.assertGreater(tuned_ms, 0.0)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.getenv("RUN_MIXED_SDDMM_TUNE", "0") == "1",
        "Set RUN_MIXED_SDDMM_TUNE=1 to run the dense-only Triton parameter sweep",
    )
    def test_dense_only_parameter_sweep(self) -> None:
        cases = [
            {
                "name": "small-total-n",
                "batch_sizes": [32, 64, 128, 256],
                "num_experts": 8,
                "dense_width": 4,
                "expert_block_size": 512,
                "hidden_dim": 2048,
                "seed_base": 22000,
            },
            {
                "name": "large-total-n",
                "batch_sizes": [32, 64, 128, 256],
                "num_experts": 24,
                "dense_width": 16,
                "expert_block_size": 512,
                "hidden_dim": 2048,
                "seed_base": 26000,
            },
        ]

        for case in cases:
            baselines = {}
            tensors = {}
            for batch_size in case["batch_sizes"]:
                x, up_proj, gate_proj, routing_weights = self._build_case_tensors(
                    batch_size=batch_size,
                    num_experts=case["num_experts"],
                    hidden_dim=case["hidden_dim"],
                    expert_block_size=case["expert_block_size"],
                    seed=case["seed_base"] + batch_size,
                    scale=0.05,
                )
                metadata = build_mixed_sddmm_metadata(
                    routing_weights,
                    dense_width=case["dense_width"],
                    sparse_width=0,
                    maxnnz=0,
                    expert_block_size=case["expert_block_size"],
                )
                expected_dense = self._build_selected_dense_baseline(x, up_proj, gate_proj, metadata)
                self._verify_dense_only_config(x, up_proj, gate_proj, metadata, expected_dense)
                tensors[batch_size] = (x, up_proj, gate_proj, metadata, expected_dense)
                baselines[batch_size] = self._measure_cuda_ms(
                    lambda x=x, up_proj=up_proj, gate_proj=gate_proj, metadata=metadata: self._build_selected_dense_baseline(
                        x, up_proj, gate_proj, metadata
                    ),
                    warmup=5,
                    iters=15,
                )

            rows = []
            for launch_kwargs in DENSE_ONLY_TUNING_CONFIGS:
                ratios = []
                timings = []
                for batch_size in case["batch_sizes"]:
                    x, up_proj, gate_proj, metadata, expected_dense = tensors[batch_size]
                    self._verify_dense_only_config(
                        x,
                        up_proj,
                        gate_proj,
                        metadata,
                        expected_dense,
                        launch_kwargs=launch_kwargs,
                    )
                    triton_ms = self._measure_dense_only_ms(
                        x,
                        up_proj,
                        gate_proj,
                        metadata,
                        launch_kwargs=launch_kwargs,
                    )
                    timings.append(triton_ms)
                    ratios.append(triton_ms / baselines[batch_size])

                geom_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
                rows.append((geom_ratio, sum(timings) / len(timings), launch_kwargs, timings, ratios))

            rows.sort(key=lambda row: row[0])
            print(
                f"[dense-only-sweep] case={case['name']} expert_n={case['expert_block_size']} "
                f"total_n={case['dense_width'] * case['expert_block_size']} k={case['hidden_dim']}"
            )
            for rank, (geom_ratio, avg_ms, launch_kwargs, timings, ratios) in enumerate(rows[:8], 1):
                detail = ", ".join(
                    f"bs={batch_size}:t={timing:.3f},r={ratio:.3f}"
                    for batch_size, timing, ratio in zip(case["batch_sizes"], timings, ratios)
                )
                print(
                    f"  rank={rank} geom_ratio={geom_ratio:.3f} avg_ms={avg_ms:.3f} "
                    f"cfg={launch_kwargs} details={detail}"
                )


if __name__ == "__main__":
    unittest.main()
