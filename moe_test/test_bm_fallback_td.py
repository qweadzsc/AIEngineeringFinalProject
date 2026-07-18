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
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
MOE_SRC_DIR = ROOT_DIR / "moe_src"
MOE_TEST_DIR = ROOT_DIR / "moe_test"
PLOT_PATH = MOE_TEST_DIR / "bm_fallback_td_sweep.png"
CSV_PATH = MOE_TEST_DIR / "bm_fallback_td_sweep.csv"
METADATA_PATH = MOE_TEST_DIR / "bm_fallback_td_sweep_metadata.txt"

for path in (ROOT_DIR, MOE_SRC_DIR, MOE_TEST_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from mlp import SPMLP
from moe_src.test.test import set_all_seed


@dataclass(frozen=True)
class BMFallbackBenchmarkCase:
    batch_size: int
    sequence_length: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    maxnnz: int
    t_d: int

    @property
    def tokens(self) -> int:
        return self.batch_size * self.sequence_length

    @property
    def total_assignments(self) -> int:
        return self.tokens * self.top_k

    @property
    def max_reachable_t_d(self) -> int:
        return min(64, self.num_experts, self.total_assignments // (self.maxnnz + 1))

    @property
    def name(self) -> str:
        return f"bs{self.batch_size}_seq{self.sequence_length}_nnz{self.maxnnz}_td{self.t_d}"


@dataclass(frozen=True)
class BMFallbackBenchmarkResult:
    case: BMFallbackBenchmarkCase
    main_ms: float
    bm_fallback_ms: float

    @property
    def winner(self) -> str:
        return "bm_fallback" if self.bm_fallback_ms < self.main_ms else "main"

    @property
    def delta_ms(self) -> float:
        return abs(self.main_ms - self.bm_fallback_ms)

    @property
    def main_vs_bm(self) -> float:
        return self.main_ms / self.bm_fallback_ms


class DummyExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, device: str, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = F.silu


class DummyOriginMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.experts = nn.ModuleList(
            [DummyExpert(hidden_size, intermediate_size, device=device, dtype=dtype) for _ in range(num_experts)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -0.02, 0.02)


class FixedGate(nn.Module):
    def __init__(self, router_logits: torch.Tensor):
        super().__init__()
        self.register_buffer("router_logits", router_logits.contiguous())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] != self.router_logits.shape[0]:
            raise ValueError(
                f"FixedGate expected {self.router_logits.shape[0]} rows but got {hidden_states.shape[0]}"
            )
        return self.router_logits


BASE_CASE = {
    "sequence_length": 1,
    "hidden_size": 2048,
    "intermediate_size": 768,
    "num_experts": 128,
    "top_k": 8,
}
DTYPE = torch.float16
DEVICE = "cuda"


def _validate_case(case: BMFallbackBenchmarkCase) -> None:
    if case.batch_size <= 0:
        raise AssertionError("batch_size must be positive")
    if case.sequence_length <= 0:
        raise AssertionError("sequence_length must be positive")
    if case.hidden_size <= 0:
        raise AssertionError("hidden_size must be positive")
    if case.intermediate_size <= 0:
        raise AssertionError("intermediate_size must be positive")
    if case.num_experts <= 0:
        raise AssertionError("num_experts must be positive")
    if case.top_k <= 0 or case.top_k > case.num_experts:
        raise AssertionError("top_k must be in [1, num_experts]")
    if case.maxnnz <= 0 or case.maxnnz > case.tokens:
        raise AssertionError("maxnnz must be in [1, tokens]")
    if case.t_d <= 0:
        raise AssertionError("t_d must be positive")
    if case.t_d > case.max_reachable_t_d:
        raise AssertionError(
            f"t_d={case.t_d} exceeds the reachable maximum {case.max_reachable_t_d} for {case.batch_size=} {case.maxnnz=}"
        )


def _build_cases() -> list[BMFallbackBenchmarkCase]:
    cases: list[BMFallbackBenchmarkCase] = []
    for batch_size in (64, 128):
        for maxnnz in (4, 8):
            template = BMFallbackBenchmarkCase(batch_size=batch_size, maxnnz=maxnnz, t_d=1, **BASE_CASE)
            max_t_d = template.max_reachable_t_d
            for t_d in range(1, max_t_d + 1):
                case = BMFallbackBenchmarkCase(batch_size=batch_size, maxnnz=maxnnz, t_d=t_d, **BASE_CASE)
                _validate_case(case)
                if _is_reachable_t_d(case):
                    cases.append(case)
    return cases


def _is_reachable_t_d(case: BMFallbackBenchmarkCase) -> bool:
    if case.t_d <= 0 or case.t_d > case.max_reachable_t_d:
        return False
    min_required = case.t_d * (case.maxnnz + 1)
    max_supported = case.t_d * case.tokens + (case.num_experts - case.t_d) * case.maxnnz
    return min_required <= case.total_assignments <= max_supported


def _build_target_counts(case: BMFallbackBenchmarkCase) -> list[int]:
    if not _is_reachable_t_d(case):
        raise RuntimeError(f"Unreachable routing target for {case}")

    counts = [0 for _ in range(case.num_experts)]
    for expert in range(case.t_d):
        counts[expert] = case.maxnnz + 1

    remaining = case.total_assignments - case.t_d * (case.maxnnz + 1)
    hot_experts = list(range(case.t_d))
    cold_experts = list(range(case.t_d, case.num_experts))

    hot_cursor = 0
    while remaining > 0 and hot_experts:
        expert = hot_experts[hot_cursor % len(hot_experts)]
        if counts[expert] < case.tokens:
            counts[expert] += 1
            remaining -= 1
        hot_cursor += 1
        if hot_cursor > len(hot_experts) * case.tokens and all(counts[e] >= case.tokens for e in hot_experts):
            break

    cold_cursor = 0
    while remaining > 0 and cold_experts:
        expert = cold_experts[cold_cursor % len(cold_experts)]
        if counts[expert] < case.maxnnz:
            counts[expert] += 1
            remaining -= 1
        cold_cursor += 1
        if cold_cursor > len(cold_experts) * max(1, case.maxnnz) and all(counts[e] >= case.maxnnz for e in cold_experts):
            break

    if remaining != 0:
        raise RuntimeError(f"Could not allocate target counts for {case}; remaining={remaining}")

    return counts


def _build_selected_experts(case: BMFallbackBenchmarkCase) -> torch.Tensor:
    remaining_counts = _build_target_counts(case)
    rows: list[list[int]] = []

    for _ in range(case.tokens):
        candidates = [(count, expert) for expert, count in enumerate(remaining_counts) if count > 0]
        if len(candidates) < case.top_k:
            raise RuntimeError(f"Not enough experts with remaining capacity to fill a row for {case}")
        candidates.sort(key=lambda item: (-item[0], item[1]))
        row = [expert for _, expert in candidates[: case.top_k]]
        for expert in row:
            remaining_counts[expert] -= 1
        rows.append(row)

    if any(remaining_counts):
        raise RuntimeError(f"Unconsumed routing counts remain for {case}: {remaining_counts[:case.t_d + 4]}")

    selected_experts = torch.tensor(rows, dtype=torch.long, device=DEVICE)
    counts = torch.bincount(selected_experts.reshape(-1), minlength=case.num_experts)
    measured_t_d = int(torch.count_nonzero(counts > case.maxnnz).item())
    if measured_t_d != case.t_d:
        raise AssertionError(f"Expected t_d={case.t_d}, but constructed routing gives {measured_t_d}")
    return selected_experts


def _build_router_logits(case: BMFallbackBenchmarkCase) -> torch.Tensor:
    selected_experts = _build_selected_experts(case)
    base_weights = torch.linspace(float(case.top_k), 1.0, steps=case.top_k, device=DEVICE, dtype=torch.float32)
    base_weights /= base_weights.sum()
    router_logits = torch.full(
        (case.tokens, case.num_experts),
        -50.0,
        device=DEVICE,
        dtype=DTYPE,
    )
    router_logits.scatter_(1, selected_experts, base_weights.log().to(DTYPE).unsqueeze(0).expand(case.tokens, -1))
    return router_logits


def _build_hidden_states(case: BMFallbackBenchmarkCase) -> torch.Tensor:
    seed = case.batch_size * 1000 + case.maxnnz * 100
    set_all_seed(seed)
    hidden_states = (
        torch.rand(
            case.batch_size,
            case.sequence_length,
            case.hidden_size,
            device=DEVICE,
            dtype=DTYPE,
        )
        - 0.5
    ) / 2
    return hidden_states.contiguous()


def _build_layer(case: BMFallbackBenchmarkCase) -> SPMLP:
    set_all_seed(20260715)
    origin = DummyOriginMLP(
        hidden_size=case.hidden_size,
        intermediate_size=case.intermediate_size,
        num_experts=case.num_experts,
        top_k=case.top_k,
        norm_topk_prob=True,
        device=DEVICE,
        dtype=DTYPE,
    )
    layer = SPMLP(origin, t_d=32, forward_mode="main", bm_fallback_t_d=32)
    layer.maxnnz = case.maxnnz
    layer.eval()
    return layer


def _run_main_branch(layer: SPMLP, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    original_threshold = layer.bm_fallback_t_d
    layer.bm_fallback_t_d = layer.num_experts + 1
    try:
        with torch.inference_mode():
            return layer.main_forward(hidden_states)
    finally:
        layer.bm_fallback_t_d = original_threshold


def _run_bm_fallback_branch(layer: SPMLP, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    original_threshold = layer.bm_fallback_t_d
    layer.bm_fallback_t_d = 0
    try:
        with torch.inference_mode():
            return layer.main_forward(hidden_states)
    finally:
        layer.bm_fallback_t_d = original_threshold


def _measure_cuda_ms(fn, *, warmup: int, iters: int) -> float:
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


def _write_csv(results: list[BMFallbackBenchmarkResult]) -> None:
    fieldnames = [
        "batch_size",
        "sequence_length",
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "maxnnz",
        "t_d",
        "max_reachable_t_d",
        "main_ms",
        "bm_fallback_ms",
        "winner",
        "delta_ms",
        "main_vs_bm",
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
                    "hidden_size": case.hidden_size,
                    "intermediate_size": case.intermediate_size,
                    "num_experts": case.num_experts,
                    "top_k": case.top_k,
                    "maxnnz": case.maxnnz,
                    "t_d": case.t_d,
                    "max_reachable_t_d": case.max_reachable_t_d,
                    "main_ms": f"{result.main_ms:.8f}",
                    "bm_fallback_ms": f"{result.bm_fallback_ms:.8f}",
                    "winner": result.winner,
                    "delta_ms": f"{result.delta_ms:.8f}",
                    "main_vs_bm": f"{result.main_vs_bm:.8f}",
                }
            )


def _load_results_from_csv() -> list[BMFallbackBenchmarkResult]:
    results: list[BMFallbackBenchmarkResult] = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case = BMFallbackBenchmarkCase(
                batch_size=int(row["batch_size"]),
                sequence_length=int(row["sequence_length"]),
                hidden_size=int(row["hidden_size"]),
                intermediate_size=int(row["intermediate_size"]),
                num_experts=int(row["num_experts"]),
                top_k=int(row["top_k"]),
                maxnnz=int(row["maxnnz"]),
                t_d=int(row["t_d"]),
            )
            results.append(
                BMFallbackBenchmarkResult(
                    case=case,
                    main_ms=float(row["main_ms"]),
                    bm_fallback_ms=float(row["bm_fallback_ms"]),
                )
            )
    return results


def _write_metadata(*, warmup: int, iters: int) -> None:
    lines = [
        "bm fallback t_d sweep for moe_test/SPMLP",
        f"plot_path={PLOT_PATH}",
        f"csv_path={CSV_PATH}",
        "",
        "Fixed hyperparameters:",
        "- model shape from /share/public/public_models/Qwen3-30B-A3B/config.json",
        "- hidden_size = 2048",
        "- moe_intermediate_size = 768",
        "- num_experts = 128",
        "- top_k = 8",
        "- dtype = float16 (to match moe_test/main.py)",
        "- device = cuda",
        "- batch_size in {64, 128}",
        "- sequence_length = 1",
        "- maxnnz in {4, 8}",
        "- t_d sweeps from 1 to min(64, total_assignments // (maxnnz + 1), num_experts)",
        f"- warmup = {warmup}",
        f"- iters = {iters}",
        "",
        "Benchmark semantics:",
        "- This benchmark exercises the current SPMLP implementation in moe_test/mlp.py.",
        "- Routing is synthesized so that count_nonzero(expert_hit_count > maxnnz) equals the requested t_d exactly.",
        "- main line: call SPMLP.main_forward(...) with bm_fallback_t_d forced above the reachable range, so it stays on the kernel path.",
        "- bm_fallback line: call SPMLP.main_forward(...) with bm_fallback_t_d=0, so the same routing/probe work runs first and then it falls through to bm_forward_with_extra_input(...).",
        "- Therefore the bm_fallback measurement includes the current mixer/probe overhead that exists before the branch decision.",
        "",
        "Artifact note:",
        "- If RUN_BM_FALLBACK_REUSE_EXISTING=1 and the CSV already exists, the benchmark will reuse the saved timings and only regenerate the metadata and figure.",
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


def _plot_results(results: list[BMFallbackBenchmarkResult], *, warmup: int, iters: int) -> None:
    from PIL import Image, ImageDraw

    image_width = 1700
    image_height = 1180
    background = "#ffffff"
    stroke = "#1f1f1f"
    main_color = "#d1495b"
    bm_color = "#2b59c3"
    grid_color = "#d7d3c7"

    image = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(30)
    subtitle_font = _load_font(18)
    axis_font = _load_font(16)
    tick_font = _load_font(14)
    legend_font = _load_font(16)
    note_font = _load_font(14)

    draw.text((60, 24), "SPMLP main path vs bm fallback while sweeping t_d", fill=stroke, font=title_font)
    draw.text(
        (60, 62),
        "Each subplot uses one (batch_size, maxnnz) pair. X axis is the constructed runtime t_d. Y axis is median latency in ms.",
        fill=stroke,
        font=subtitle_font,
    )

    legend_y = 92
    draw.line((60, legend_y + 10, 120, legend_y + 10), fill=main_color, width=4)
    draw.text((130, legend_y), "main kernel path", fill=stroke, font=legend_font)
    draw.line((360, legend_y + 10, 420, legend_y + 10), fill=bm_color, width=4)
    draw.text((430, legend_y), "bm fallback path", fill=stroke, font=legend_font)

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

            first_bm_td = next((result.case.t_d for result in subset if result.winner == "bm_fallback"), None)
            x0 = outer_left + col * (subplot_width + gap_x)
            y0 = outer_top + row * (subplot_height + gap_y)
            x1 = x0 + subplot_width
            y1 = y0 + subplot_height
            draw.rectangle((x0, y0, x1, y1), outline=stroke, width=2)

            title = (
                f"batch={batch_size}, seq=1, tokens={batch_size}, maxnnz={maxnnz}\n"
                f"t_d range=1..{subset[-1].case.max_reachable_t_d}, first bm win={first_bm_td}, wins main/bm="
                f"{sum(1 for r in subset if r.winner == 'main')}/{sum(1 for r in subset if r.winner == 'bm_fallback')}"
            )
            _draw_multiline_text(draw, x0 + 16, y0 + 10, title, axis_font, stroke, width=54)

            plot_left = x0 + 68
            plot_right = x1 - 18
            plot_top = y0 + 76
            plot_bottom = y1 - 56
            draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=stroke, width=1)

            all_y = [result.main_ms for result in subset] + [result.bm_fallback_ms for result in subset]
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

            main_points = [(project_x(result.case.t_d), project_y(result.main_ms)) for result in subset]
            bm_points = [(project_x(result.case.t_d), project_y(result.bm_fallback_ms)) for result in subset]
            draw.line(main_points, fill=main_color, width=3)
            draw.line(bm_points, fill=bm_color, width=3)

            draw.text((plot_left, plot_bottom + 30), "t_d", fill=stroke, font=axis_font)
            draw.text((x0 + 16, plot_top - 22), "Latency (ms)", fill=stroke, font=axis_font)

    note = (
        "Fixed hyperparameters: hidden_size=2048, intermediate_size=768, num_experts=128, top_k=8, dtype=float16, device=cuda. "
        f"Measurement uses median CUDA event time with warmup={warmup}, iters={iters}. "
        "The routing pattern is synthesized so that the runtime count_nonzero(expert_hit_count > maxnnz) equals the requested t_d exactly. "
        "Red line stays on the current SPMLP kernel path; blue line forces the current bm fallback branch after the same probe work."
    )
    _draw_multiline_text(draw, 60, image_height - 140, note, note_font, stroke, width=150)

    image.save(PLOT_PATH)


@unittest.skipUnless(
    torch.cuda.is_available() and os.getenv("RUN_BM_FALLBACK_BENCH", "0") == "1",
    "Set RUN_BM_FALLBACK_BENCH=1 to run the SPMLP main-vs-bm fallback benchmark",
)
class BMFallbackBenchmarkTest(unittest.TestCase):
    def _run_case(
        self,
        layer: SPMLP,
        case: BMFallbackBenchmarkCase,
        *,
        warmup: int,
        iters: int,
        hidden_states: torch.Tensor,
    ) -> BMFallbackBenchmarkResult:
        _validate_case(case)
        layer.maxnnz = case.maxnnz
        layer.gate = FixedGate(_build_router_logits(case))

        # Validate the constructed route before timing.
        _run_main_branch(layer, hidden_states)
        if layer.t_d != case.t_d:
            raise AssertionError(f"Expected runtime t_d={case.t_d}, but SPMLP observed {layer.t_d}")

        main_ms = _measure_cuda_ms(
            lambda: _run_main_branch(layer, hidden_states),
            warmup=warmup,
            iters=iters,
        )
        bm_fallback_ms = _measure_cuda_ms(
            lambda: _run_bm_fallback_branch(layer, hidden_states),
            warmup=warmup,
            iters=iters,
        )

        result = BMFallbackBenchmarkResult(case=case, main_ms=main_ms, bm_fallback_ms=bm_fallback_ms)
        print(
            "[bm-fallback-bench] "
            f"case={case.name} tokens={case.tokens} maxnnz={case.maxnnz} t_d={case.t_d} "
            f"main_ms={result.main_ms:.4f} bm_fallback_ms={result.bm_fallback_ms:.4f} "
            f"winner={result.winner} delta_ms={result.delta_ms:.4f} main_vs_bm={result.main_vs_bm:.4f}"
        )
        return result

    def test_main_vs_bm_fallback_td_sweep(self) -> None:
        warmup = int(os.getenv("BM_FALLBACK_WARMUP", "5"))
        iters = int(os.getenv("BM_FALLBACK_ITERS", "10"))
        reuse_existing = os.getenv("RUN_BM_FALLBACK_REUSE_EXISTING", "0") == "1"

        if reuse_existing and CSV_PATH.exists():
            results = _load_results_from_csv()
        else:
            cases = _build_cases()
            results: list[BMFallbackBenchmarkResult] = []
            layer = _build_layer(cases[0])
            hidden_cache: dict[int, torch.Tensor] = {}
            for case in cases:
                if case.batch_size not in hidden_cache:
                    hidden_cache[case.batch_size] = _build_hidden_states(case)
                with self.subTest(case=case.name):
                    result = self._run_case(
                        layer,
                        case,
                        warmup=warmup,
                        iters=iters,
                        hidden_states=hidden_cache[case.batch_size],
                    )
                    self.assertGreater(result.main_ms, 0.0)
                    self.assertGreater(result.bm_fallback_ms, 0.0)
                    results.append(result)
            _write_csv(results)

        expected_cases = len(_build_cases())
        self.assertEqual(len(results), expected_cases)
        _write_metadata(warmup=warmup, iters=iters)
        _plot_results(results, warmup=warmup, iters=iters)

        main_wins = sum(1 for result in results if result.winner == "main")
        bm_wins = len(results) - main_wins
        mean_ratio = sum(result.main_vs_bm for result in results) / len(results)
        geometric_ratio = math.exp(sum(math.log(result.main_vs_bm) for result in results) / len(results))

        for batch_size in (64, 128):
            for maxnnz in (4, 8):
                subset = [
                    result
                    for result in results
                    if result.case.batch_size == batch_size and result.case.maxnnz == maxnnz
                ]
                subset.sort(key=lambda result: result.case.t_d)
                first_bm_td = next((result.case.t_d for result in subset if result.winner == "bm_fallback"), None)
                print(
                    "[bm-fallback-group] "
                    f"batch={batch_size} maxnnz={maxnnz} cases={len(subset)} "
                    f"first_bm_win_t_d={first_bm_td} "
                    f"main_wins={sum(1 for r in subset if r.winner == 'main')} "
                    f"bm_wins={sum(1 for r in subset if r.winner == 'bm_fallback')}"
                )

        print(
            "[bm-fallback-summary] "
            f"cases={len(results)} main_wins={main_wins} bm_wins={bm_wins} "
            f"mean_main_vs_bm={mean_ratio:.4f} geom_main_vs_bm={geometric_ratio:.4f} "
            f"plot={PLOT_PATH.name} csv={CSV_PATH.name} meta={METADATA_PATH.name}"
        )

        self.assertTrue(PLOT_PATH.exists())
        self.assertTrue(CSV_PATH.exists())
        self.assertTrue(METADATA_PATH.exists())


if __name__ == "__main__":
    unittest.main()
