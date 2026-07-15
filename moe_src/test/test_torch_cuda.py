from __future__ import annotations

import csv
import math
import os
import sys
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
PLOT_PATH = TEST_DIR / "torch_cuda_td_sweep.png"
CSV_PATH = TEST_DIR / "torch_cuda_td_sweep.csv"
METADATA_PATH = TEST_DIR / "torch_cuda_td_sweep_metadata.txt"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from moe_src.test.test import TestCUDAMoe, set_all_seed


@dataclass(frozen=True)
class TorchCudaBenchmarkCase:
    batch_size: int
    sequence_length: int
    hid_dim: int
    num_experts: int
    expert_w: int
    top_k: int
    sp_pd: int
    t_d: int
    maxnnz: int

    @property
    def tokens(self) -> int:
        return self.batch_size * self.sequence_length

    @property
    def sparse_width(self) -> int:
        return self.t_d // self.sp_pd

    @property
    def selected_experts(self) -> int:
        return self.t_d + self.sparse_width

    @property
    def work_ratio(self) -> float:
        dense_work = self.tokens * self.t_d
        sparse_work = self.maxnnz * self.sparse_width
        full_dense_work = self.tokens * self.num_experts
        return (dense_work + sparse_work) / full_dense_work

    @property
    def sparsity(self) -> float:
        return 1.0 - self.work_ratio

    @property
    def name(self) -> str:
        return f"bs{self.batch_size}_seq{self.sequence_length}_td{self.t_d}_nnz{self.maxnnz}"


@dataclass(frozen=True)
class TorchCudaBenchmarkResult:
    case: TorchCudaBenchmarkCase
    cuda_ms: float
    bmm_ms: float

    @property
    def winner(self) -> str:
        return "cuda" if self.cuda_ms < self.bmm_ms else "bmm"

    @property
    def delta_ms(self) -> float:
        return abs(self.cuda_ms - self.bmm_ms)

    @property
    def speedup(self) -> float:
        faster = min(self.cuda_ms, self.bmm_ms)
        slower = max(self.cuda_ms, self.bmm_ms)
        return slower / faster

    @property
    def cuda_vs_bmm(self) -> float:
        return self.cuda_ms / self.bmm_ms


def _validate_case(case: TorchCudaBenchmarkCase) -> None:
    if case.batch_size <= 0:
        raise AssertionError("batch_size must be positive")
    if case.sequence_length <= 0:
        raise AssertionError("sequence_length must be positive")
    if case.hid_dim <= 0:
        raise AssertionError("hid_dim must be positive")
    if case.num_experts <= 0:
        raise AssertionError("num_experts must be positive")
    if case.expert_w <= 0:
        raise AssertionError("expert_w must be positive")
    if case.top_k <= 0 or case.top_k > case.num_experts:
        raise AssertionError("top_k must be in [1, num_experts]")
    if case.sp_pd <= 0:
        raise AssertionError("sp_pd must be positive")
    if case.t_d <= 0:
        raise AssertionError("t_d must be positive")
    if case.t_d % case.sp_pd != 0:
        raise AssertionError("t_d must be divisible by sp_pd")
    if case.maxnnz < 0 or case.maxnnz > case.tokens:
        raise AssertionError("maxnnz must satisfy 0 <= maxnnz <= tokens")
    if case.selected_experts > case.num_experts:
        raise AssertionError("dense + sparse selected experts must not exceed num_experts")
    if case.expert_w % 128 != 0:
        raise AssertionError("expert_w must be divisible by 128 for the current CUDA layout assumption")


def _build_td_sweep_cases() -> list[TorchCudaBenchmarkCase]:
    base = {
        "sequence_length": 1,
        "hid_dim": 2048,
        "num_experts": 128,
        "expert_w": 512,
        "top_k": 8,
        "sp_pd": 1,
    }
    cases: list[TorchCudaBenchmarkCase] = []
    for batch_size in (64, 128):
        for maxnnz in (4, 8):
            for t_d in range(1, 65):
                case = TorchCudaBenchmarkCase(
                    batch_size=batch_size,
                    t_d=t_d,
                    maxnnz=maxnnz,
                    **base,
                )
                _validate_case(case)
                cases.append(case)
    return cases


def _write_csv(results: list[TorchCudaBenchmarkResult]) -> None:
    fieldnames = [
        "batch_size",
        "sequence_length",
        "tokens",
        "hid_dim",
        "num_experts",
        "expert_w",
        "top_k",
        "sp_pd",
        "t_d",
        "sparse_width",
        "selected_experts",
        "maxnnz",
        "work_ratio",
        "sparsity",
        "cuda_ms",
        "bmm_ms",
        "winner",
        "delta_ms",
        "speedup",
        "cuda_vs_bmm",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            case = result.case
            writer.writerow(
                {
                    "batch_size": case.batch_size,
                    "sequence_length": case.sequence_length,
                    "tokens": case.tokens,
                    "hid_dim": case.hid_dim,
                    "num_experts": case.num_experts,
                    "expert_w": case.expert_w,
                    "top_k": case.top_k,
                    "sp_pd": case.sp_pd,
                    "t_d": case.t_d,
                    "sparse_width": case.sparse_width,
                    "selected_experts": case.selected_experts,
                    "maxnnz": case.maxnnz,
                    "work_ratio": f"{case.work_ratio:.8f}",
                    "sparsity": f"{case.sparsity:.8f}",
                    "cuda_ms": f"{result.cuda_ms:.8f}",
                    "bmm_ms": f"{result.bmm_ms:.8f}",
                    "winner": result.winner,
                    "delta_ms": f"{result.delta_ms:.8f}",
                    "speedup": f"{result.speedup:.8f}",
                    "cuda_vs_bmm": f"{result.cuda_vs_bmm:.8f}",
                }
            )


def _load_results_from_csv() -> list[TorchCudaBenchmarkResult]:
    results: list[TorchCudaBenchmarkResult] = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case = TorchCudaBenchmarkCase(
                batch_size=int(row["batch_size"]),
                sequence_length=int(row["sequence_length"]),
                hid_dim=int(row["hid_dim"]),
                num_experts=int(row["num_experts"]),
                expert_w=int(row["expert_w"]),
                top_k=int(row["top_k"]),
                sp_pd=int(row["sp_pd"]),
                t_d=int(row["t_d"]),
                maxnnz=int(row["maxnnz"]),
            )
            results.append(
                TorchCudaBenchmarkResult(
                    case=case,
                    cuda_ms=float(row["cuda_ms"]),
                    bmm_ms=float(row["bmm_ms"]),
                )
            )
    return results


def _write_metadata(*, warmup: int, iters: int) -> None:
    lines = [
        "torch.bmm vs custom CUDA t_d sweep",
        f"plot_path={PLOT_PATH}",
        f"csv_path={CSV_PATH}",
        "",
        "Fixed hyperparameters:",
        "- batch_size in {64, 128}",
        "- sequence_length = 1",
        "- tokens = batch_size * sequence_length",
        "- maxnnz in {4, 8}",
        "- t_d in [1, 64]",
        "- hidden_size = 2048",
        "- num_experts = 128",
        "- expert_w = 512",
        "- top_k = 8",
        "- sp_pd = 1",
        "- dtype = float16",
        "- device = cuda",
        "- benchmark_seed = batch_size * 1000 + maxnnz * 100 (fixed within each subplot)",
        f"- warmup = {warmup}",
        f"- iters = {iters}",
        "",
        "Definitions:",
        "- custom CUDA path = TestCUDAMoe.forward_cuda(...) from moe_src/test/test.py",
        "- torch.bmm path = TestCUDAMoe.dense_forward(...) from moe_src/test/test.py",
        "- sparse_width = t_d // sp_pd",
        "- selected_experts = t_d + sparse_width",
        "- work_ratio = (tokens * t_d + maxnnz * sparse_width) / (tokens * num_experts)",
        "- sparsity = 1 - work_ratio",
        "",
        "Important semantic note:",
        "- forward_cuda computes the selected dense+sparse expert path.",
        "- dense_forward computes the full dense MoE path with batched torch.bmm.",
        "- Random seed is fixed per (batch_size, maxnnz) group and does not depend on t_d.",
        "",
        "Artifact note:",
        "- If TORCH_CUDA_REUSE_EXISTING=1 and the CSV already exists, the benchmark test will reuse the saved measurements and regenerate the figure without rerunning the full sweep.",
    ]
    METADATA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_multiline_text(draw, x: int, y: int, text: str, font, fill: str, width: int) -> int:
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [paragraph])
    cursor_y = y
    for line in lines:
        draw.text((x, cursor_y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, cursor_y), line or " ", font=font)
        cursor_y = bbox[3] + 4
    return cursor_y


def _plot_results(results: list[TorchCudaBenchmarkResult], *, warmup: int, iters: int) -> None:
    from PIL import Image, ImageDraw

    image_width = 1700
    image_height = 1180
    background = "#ffffff"
    stroke = "#1f1f1f"
    cuda_color = "#d1495b"
    bmm_color = "#2b59c3"
    grid_color = "#d7d3c7"

    image = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(30)
    subtitle_font = _load_font(18)
    axis_font = _load_font(16)
    tick_font = _load_font(14)
    legend_font = _load_font(16)
    note_font = _load_font(14)

    draw.text((60, 24), "Custom CUDA vs torch.bmm while sweeping t_d", fill=stroke, font=title_font)
    draw.text(
        (60, 62),
        "Each subplot uses one (batch_size, maxnnz) pair. X axis is t_d from 1 to 64. Y axis is median latency in ms.",
        fill=stroke,
        font=subtitle_font,
    )

    legend_y = 92
    draw.line((60, legend_y + 10, 120, legend_y + 10), fill=cuda_color, width=4)
    draw.text((130, legend_y), "custom CUDA", fill=stroke, font=legend_font)
    draw.line((290, legend_y + 10, 350, legend_y + 10), fill=bmm_color, width=4)
    draw.text((360, legend_y), "torch.bmm", fill=stroke, font=legend_font)

    outer_left = 60
    outer_right = 60
    outer_top = 140
    outer_bottom = 180
    gap_x = 40
    gap_y = 48
    subplot_width = (image_width - outer_left - outer_right - gap_x) // 2
    subplot_height = (image_height - outer_top - outer_bottom - gap_y) // 2

    x_ticks = [1, 16, 32, 48, 64]
    batch_sizes = (64, 128)
    maxnnz_values = (4, 8)

    for row, batch_size in enumerate(batch_sizes):
        for col, maxnnz in enumerate(maxnnz_values):
            subset = [
                result
                for result in results
                if result.case.batch_size == batch_size and result.case.maxnnz == maxnnz
            ]
            subset.sort(key=lambda result: result.case.t_d)
            if not subset:
                continue

            x0 = outer_left + col * (subplot_width + gap_x)
            y0 = outer_top + row * (subplot_height + gap_y)
            x1 = x0 + subplot_width
            y1 = y0 + subplot_height
            draw.rectangle((x0, y0, x1, y1), outline=stroke, width=2)

            title = (
                f"batch={batch_size}, seq=1, tokens={batch_size}, maxnnz={maxnnz}\n"
                f"sparsity {subset[0].case.sparsity:.3f} -> {subset[-1].case.sparsity:.3f}, "
                f"wins cuda/bmm={sum(1 for r in subset if r.winner == 'cuda')}/{sum(1 for r in subset if r.winner == 'bmm')}"
            )
            _draw_multiline_text(draw, x0 + 16, y0 + 10, title, axis_font, stroke, width=52)

            plot_left = x0 + 68
            plot_right = x1 - 18
            plot_top = y0 + 76
            plot_bottom = y1 - 56
            draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=stroke, width=1)

            all_y = [result.cuda_ms for result in subset] + [result.bmm_ms for result in subset]
            y_min = min(all_y)
            y_max = max(all_y)
            y_pad = max((y_max - y_min) * 0.08, 0.02)
            y_min -= y_pad
            y_max += y_pad
            if y_max <= y_min:
                y_max = y_min + 1.0

            for idx in range(5):
                frac = idx / 4
                y_value = y_max - frac * (y_max - y_min)
                py = plot_top + frac * (plot_bottom - plot_top)
                draw.line((plot_left, py, plot_right, py), fill=grid_color, width=1)
                label = f"{y_value:.2f}"
                bbox = draw.textbbox((0, 0), label, font=tick_font)
                draw.text((plot_left - bbox[2] - 8, py - bbox[3] / 2), label, fill=stroke, font=tick_font)

            for t_d in x_ticks:
                px = plot_left + (t_d - 1) * (plot_right - plot_left) / 63
                draw.line((px, plot_bottom, px, plot_bottom + 5), fill=stroke, width=1)
                label = str(t_d)
                bbox = draw.textbbox((0, 0), label, font=tick_font)
                draw.text((px - bbox[2] / 2, plot_bottom + 8), label, fill=stroke, font=tick_font)

            def project_x(t_d: int) -> float:
                return plot_left + (t_d - 1) * (plot_right - plot_left) / 63

            def project_y(value: float) -> float:
                return plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)

            cuda_points = [(project_x(result.case.t_d), project_y(result.cuda_ms)) for result in subset]
            bmm_points = [(project_x(result.case.t_d), project_y(result.bmm_ms)) for result in subset]
            draw.line(cuda_points, fill=cuda_color, width=3)
            draw.line(bmm_points, fill=bmm_color, width=3)

            draw.text((plot_left, plot_bottom + 30), "t_d", fill=stroke, font=axis_font)
            draw.text((x0 + 16, plot_top - 22), "Latency (ms)", fill=stroke, font=axis_font)

    note = (
        "Fixed hyperparameters: hidden_size=2048, num_experts=128, expert_w=512, top_k=8, sp_pd=1, dtype=float16, device=cuda, sequence_length=1. "
        f"Measurement uses median CUDA event time with warmup={warmup}, iters={iters}. Seed is fixed per (batch_size, maxnnz) subplot. "
        "Custom CUDA is the selected dense+sparse kernel path from forward_cuda(...); torch.bmm is the full dense MoE path from dense_forward(...)."
    )
    _draw_multiline_text(draw, 60, image_height - 140, note, note_font, stroke, width=150)

    image.save(PLOT_PATH)


@unittest.skipUnless(
    torch.cuda.is_available() and os.getenv("RUN_TORCH_CUDA_BENCH", "0") == "1",
    "Set RUN_TORCH_CUDA_BENCH=1 to run the torch.bmm vs custom CUDA benchmark",
)
class TorchCudaBenchmarkTest(unittest.TestCase):
    def _run_case(
        self,
        case: TorchCudaBenchmarkCase,
        *,
        warmup: int,
        iters: int,
    ) -> TorchCudaBenchmarkResult:
        _validate_case(case)

        seed = case.batch_size * 1000 + case.maxnnz * 100
        set_all_seed(seed)

        model = TestCUDAMoe(
            case.hid_dim,
            case.t_d,
            case.maxnnz,
            num_experts=case.num_experts,
            expert_w=case.expert_w,
            top_k=case.top_k,
            sp_pd=case.sp_pd,
        )
        model.rand_init()
        model = model.to("cuda")

        x = (
            torch.rand(
                case.batch_size,
                case.sequence_length,
                case.hid_dim,
                device="cuda",
                dtype=torch.float16,
            )
            - 0.5
        ) / 2

        cuda_ms = model._measure_cuda_ms(
            lambda: model.forward_cuda(x, verbose=False, record_time=False),
            warmup=warmup,
            iters=iters,
        )
        bmm_ms = model._measure_cuda_ms(
            lambda: model.dense_forward(x),
            warmup=warmup,
            iters=iters,
        )

        result = TorchCudaBenchmarkResult(case=case, cuda_ms=cuda_ms, bmm_ms=bmm_ms)
        print(
            "[torch-vs-cuda] "
            f"case={case.name} tokens={case.tokens} t_d={case.t_d} sparse_width={case.sparse_width} "
            f"maxnnz={case.maxnnz} work_ratio={case.work_ratio:.4f} sparsity={case.sparsity:.4f} "
            f"cuda_ms={result.cuda_ms:.4f} bmm_ms={result.bmm_ms:.4f} "
            f"winner={result.winner} delta_ms={result.delta_ms:.4f} "
            f"speedup={result.speedup:.4f}x cuda_vs_bmm={result.cuda_vs_bmm:.4f}"
        )

        del x, model
        torch.cuda.empty_cache()
        return result

    def test_torch_bmm_vs_custom_cuda_td_sweep(self) -> None:
        warmup = int(os.getenv("TORCH_CUDA_BENCH_WARMUP", "10"))
        iters = int(os.getenv("TORCH_CUDA_BENCH_ITERS", "20"))
        reuse_existing = os.getenv("TORCH_CUDA_REUSE_EXISTING", "0") == "1"

        if reuse_existing and CSV_PATH.exists():
            results = _load_results_from_csv()
        else:
            cases = _build_td_sweep_cases()
            results: list[TorchCudaBenchmarkResult] = []
            for case in cases:
                with self.subTest(case=case.name):
                    result = self._run_case(case, warmup=warmup, iters=iters)
                    self.assertGreater(result.cuda_ms, 0.0)
                    self.assertGreater(result.bmm_ms, 0.0)
                    results.append(result)
            _write_csv(results)

        self.assertEqual(len(results), 256)
        _write_metadata(warmup=warmup, iters=iters)
        _plot_results(results, warmup=warmup, iters=iters)

        cuda_wins = sum(1 for result in results if result.winner == "cuda")
        bmm_wins = len(results) - cuda_wins
        mean_ratio = sum(result.cuda_vs_bmm for result in results) / len(results)
        geometric_ratio = math.exp(sum(math.log(result.cuda_vs_bmm) for result in results) / len(results))

        for batch_size in (64, 128):
            for maxnnz in (4, 8):
                subset = [
                    result
                    for result in results
                    if result.case.batch_size == batch_size and result.case.maxnnz == maxnnz
                ]
                subset.sort(key=lambda result: result.case.t_d)
                first_bmm_td = next((result.case.t_d for result in subset if result.winner == "bmm"), None)
                print(
                    "[torch-vs-cuda-group] "
                    f"batch={batch_size} maxnnz={maxnnz} cuda_wins={sum(1 for r in subset if r.winner == 'cuda')} "
                    f"bmm_wins={sum(1 for r in subset if r.winner == 'bmm')} "
                    f"first_bmm_win_td={first_bmm_td} plot={PLOT_PATH.name}"
                )

        print(
            "[torch-vs-cuda-summary] "
            f"cases={len(results)} cuda_wins={cuda_wins} bmm_wins={bmm_wins} "
            f"mean_cuda_vs_bmm={mean_ratio:.4f} geom_cuda_vs_bmm={geometric_ratio:.4f} "
            f"plot={PLOT_PATH.name} csv={CSV_PATH.name} meta={METADATA_PATH.name}"
        )

        self.assertTrue(PLOT_PATH.exists())
        self.assertTrue(CSV_PATH.exists())
        self.assertTrue(METADATA_PATH.exists())


if __name__ == "__main__":
    unittest.main()
