from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from moe_src.sddmm_validation.mixed_sddmm import build_mixed_sddmm_metadata


class MixedSDDMMMetadataTest(unittest.TestCase):
    def _devices(self) -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

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

                self.assertTrue(
                    torch.equal(metadata.mask_c_dense, expected_mask_c_dense.to(device=device))
                )
                self.assertTrue(
                    torch.equal(metadata.mask_c_sparse, expected_mask_c_sparse.to(device=device))
                )
                self.assertTrue(
                    torch.equal(metadata.mask_r_sparse, expected_mask_r_sparse.to(device=device))
                )

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

                self.assertTrue(
                    torch.equal(metadata.mask_c_dense, expected_mask_c_dense.to(device=device))
                )
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


if __name__ == "__main__":
    unittest.main()
