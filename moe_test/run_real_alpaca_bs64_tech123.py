from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOE_TEST_DIR = PROJECT_ROOT / "moe_test"
DEFAULT_DATASET_INDEX = 0
DEFAULT_NUM_PROMPTS = 10
DEFAULT_EA_TOTAL_TOKEN = 64
DEFAULT_T_D_CFG = 64


@dataclass(frozen=True)
class MethodConfig:
    name: str
    method: str
    description: str
    extra_args: tuple[str, ...]


@dataclass
class MethodResult:
    name: str
    method: str
    description: str
    command: list[str]
    returncode: int
    average_time_seconds: float | None
    average_time_ms: float | None
    average_al: float | None
    peak_gpu_gb: float | None
    spmlp_layers: int | None
    kernel_path_calls: int | None
    unsupported_batch_fallback_calls: int | None
    runtime_t_d_fallback_calls: int | None
    original_forward_calls: int | None
    bm_forward_calls: int | None
    raw_log_path: str


def _python_bin() -> str:
    venv_python = PROJECT_ROOT / '.venv' / 'bin' / 'python'
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _build_method_configs(tech3_threshold: int) -> list[MethodConfig]:
    common_original_fallback = (
        '--spmlp-unsupported-fallback-mode',
        'original',
    )
    return [
        MethodConfig(
            name='tech1',
            method='mtp',
            description='MTP, fixed t_d=64, no adaptive_t_d, runtime fallback disabled',
            extra_args=(
                '--no-spmlp-adaptive-td',
                '--spmlp-bm-fallback-td',
                '-1',
                *common_original_fallback,
                '--spmlp-runtime-fallback-mode',
                'none',
            ),
        ),
        MethodConfig(
            name='tech2',
            method='mtp',
            description='MTP, adaptive_t_d enabled, runtime fallback disabled',
            extra_args=(
                '--spmlp-adaptive-td',
                '--spmlp-bm-fallback-td',
                '-1',
                *common_original_fallback,
                '--spmlp-runtime-fallback-mode',
                'none',
            ),
        ),
        MethodConfig(
            name='tech3',
            method='mtp',
            description=f'MTP, adaptive_t_d enabled, runtime fallback to bm when runtime_t_d > {tech3_threshold}',
            extra_args=(
                '--spmlp-adaptive-td',
                '--spmlp-bm-fallback-td',
                str(tech3_threshold),
                *common_original_fallback,
                '--spmlp-runtime-fallback-mode',
                'bm',
            ),
        ),
    ]


def _parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return None if match is None else float(match.group(1))


def _parse_spmlp_stats(text: str) -> dict[str, int | None]:
    stats = {
        'spmlp_layers': None,
        'kernel_path_calls': None,
        'unsupported_batch_fallback_calls': None,
        'runtime_t_d_fallback_calls': None,
        'original_forward_calls': None,
        'bm_forward_calls': None,
    }
    match = re.search(r'SPMLP path stats:\s+(.*)', text)
    if match is None:
        return stats
    aliases = {'layers': 'spmlp_layers'}
    for token in match.group(1).split():
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        key = aliases.get(key, key)
        if key in stats:
            stats[key] = int(value)
    return stats


def _run_one(config: MethodConfig, *, dataset: int, num_prompts: int, ea_total_token: int, maxnnz: int, t_d_cfg: int, gpu: str, output_dir: Path) -> MethodResult:
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        _python_bin(),
        '-s',
        'main.py',
        '--dataset', str(dataset),
        '--method', config.method,
        '--num-prompts', str(num_prompts),
        '--ea-total-token', str(ea_total_token),
        '--spmlp-maxnnz', str(maxnnz),
        '--spmlp-t-d', str(t_d_cfg),
        *config.extra_args,
    ]

    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONPATH'] = str(PROJECT_ROOT / 'moe_src')
    env['CUDA_VISIBLE_DEVICES'] = gpu

    completed = subprocess.run(
        cmd,
        cwd=MOE_TEST_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    raw_log_path = logs_dir / f'{config.name}.log'
    raw_log_path.write_text(completed.stdout, encoding='utf-8')

    average_time_seconds = _parse_float(r'Average time across \d+ prompts: ([0-9.]+) seconds', completed.stdout)
    average_al = _parse_float(r'Average AL \(Acceptance Length\) across \d+ prompts: ([0-9.]+)', completed.stdout)
    peak_gpu_gb = _parse_float(r'Peak GPU memory usage: ([0-9.]+) GB', completed.stdout)
    stats = _parse_spmlp_stats(completed.stdout)

    return MethodResult(
        name=config.name,
        method=config.method,
        description=config.description,
        command=cmd,
        returncode=completed.returncode,
        average_time_seconds=average_time_seconds,
        average_time_ms=None if average_time_seconds is None else average_time_seconds * 1000.0,
        average_al=average_al,
        peak_gpu_gb=peak_gpu_gb,
        spmlp_layers=stats['spmlp_layers'],
        kernel_path_calls=stats['kernel_path_calls'],
        unsupported_batch_fallback_calls=stats['unsupported_batch_fallback_calls'],
        runtime_t_d_fallback_calls=stats['runtime_t_d_fallback_calls'],
        original_forward_calls=stats['original_forward_calls'],
        bm_forward_calls=stats['bm_forward_calls'],
        raw_log_path=str(raw_log_path),
    )


def _write_csv(results: list[MethodResult], output_dir: Path) -> Path:
    csv_path = output_dir / 'summary.csv'
    fieldnames = [
        'name', 'method', 'description', 'returncode', 'average_time_seconds', 'average_time_ms', 'average_al',
        'peak_gpu_gb', 'spmlp_layers', 'kernel_path_calls', 'unsupported_batch_fallback_calls',
        'runtime_t_d_fallback_calls', 'original_forward_calls', 'bm_forward_calls', 'raw_log_path',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row.pop('command')
            writer.writerow(row)
    return csv_path


def _write_json(results: list[MethodResult], output_dir: Path, args: argparse.Namespace) -> Path:
    json_path = output_dir / 'summary.json'
    payload = {
        'date': '2026-07-18',
        'dataset_index': args.dataset,
        'num_prompts': args.num_prompts,
        'ea_total_token': args.ea_total_token,
        'maxnnz': args.maxnnz,
        't_d_cfg': args.t_d_cfg,
        'tech3_threshold': args.tech3_threshold,
        'gpu': args.gpu,
        'results': [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return json_path


def _write_markdown(results: list[MethodResult], output_dir: Path, args: argparse.Namespace) -> Path:
    md_path = output_dir / 'summary.md'
    by_name = {result.name: result for result in results}
    lines: list[str] = []
    lines.append(f'# Alpaca 真实 10-prompt ablation：ea_total_token={args.ea_total_token}, maxnnz={args.maxnnz}')
    lines.append('')
    lines.append('## 配置')
    lines.append('')
    lines.append('- 日期：2026-07-18')
    lines.append(f'- 数据集：alpaca（dataset index = {args.dataset}）')
    lines.append(f'- prompt 数：{args.num_prompts}')
    lines.append(f'- `ea_total_token`：{args.ea_total_token}')
    lines.append(f'- `maxnnz`：{args.maxnnz}')
    lines.append(f'- `t_d_cfg`：{args.t_d_cfg}')
    lines.append(f'- `tech3` 的 `bm_fallback_t_d`：{args.tech3_threshold}')
    lines.append('- `max_new_tokens`：128')
    lines.append('- 说明：这里的“bs64”按当前真实实验口径，对应 `ea_total_token=64`，不是把每次 MLP 调用固定成单一 batch size。')
    lines.append('')
    lines.append('## 结果')
    lines.append('')
    lines.append('| method | avg latency (s) | avg latency (ms) | avg AL | peak GPU (GB) | kernel calls | unsupported fallback | runtime bm fallback | bm calls |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for result in results:
        lines.append(
            f"| {result.name} | {result.average_time_seconds if result.average_time_seconds is not None else 'NA'} | {result.average_time_ms if result.average_time_ms is not None else 'NA'} | {result.average_al if result.average_al is not None else 'NA'} | {result.peak_gpu_gb if result.peak_gpu_gb is not None else 'NA'} | {result.kernel_path_calls if result.kernel_path_calls is not None else 'NA'} | {result.unsupported_batch_fallback_calls if result.unsupported_batch_fallback_calls is not None else 'NA'} | {result.runtime_t_d_fallback_calls if result.runtime_t_d_fallback_calls is not None else 'NA'} | {result.bm_forward_calls if result.bm_forward_calls is not None else 'NA'} |"
        )
    lines.append('')
    lines.append('## 相对速度')
    lines.append('')
    def add_speedup(faster: str, slower: str):
        a = by_name.get(faster)
        b = by_name.get(slower)
        if a is None or b is None or a.average_time_seconds in (None, 0.0) or b.average_time_seconds is None:
            return
        lines.append(f'- `{faster}` 相对 `{slower}`：`{b.average_time_seconds / a.average_time_seconds:.4f}x`')
    add_speedup('tech2', 'tech1')
    add_speedup('tech3', 'tech2')
    lines.append('')
    lines.append('## 原始日志')
    lines.append('')
    for result in results:
        lines.append(f'- `{result.name}`: `{result.raw_log_path}`')
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Run real alpaca benchmark for the bs64 tech1/2/3 ablation setting.')
    parser.add_argument('--dataset', type=int, default=DEFAULT_DATASET_INDEX)
    parser.add_argument('--num-prompts', type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument('--ea-total-token', type=int, default=DEFAULT_EA_TOTAL_TOKEN)
    parser.add_argument('--maxnnz', type=int, required=True)
    parser.add_argument('--t-d-cfg', type=int, default=DEFAULT_T_D_CFG)
    parser.add_argument('--tech3-threshold', type=int, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[MethodResult] = []
    for config in _build_method_configs(args.tech3_threshold):
        print(f'[run] {config.name}: {config.description}', flush=True)
        result = _run_one(
            config,
            dataset=args.dataset,
            num_prompts=args.num_prompts,
            ea_total_token=args.ea_total_token,
            maxnnz=args.maxnnz,
            t_d_cfg=args.t_d_cfg,
            gpu=args.gpu,
            output_dir=output_dir,
        )
        results.append(result)
        if result.returncode != 0:
            print(f'[error] {config.name} failed. See {result.raw_log_path}', flush=True)
            _write_csv(results, output_dir)
            _write_json(results, output_dir, args)
            _write_markdown(results, output_dir, args)
            return result.returncode
        print(
            f'[done] {config.name}: avg={result.average_time_seconds}s AL={result.average_al} '
            f'kernel={result.kernel_path_calls} bm_calls={result.bm_forward_calls}',
            flush=True,
        )

    csv_path = _write_csv(results, output_dir)
    json_path = _write_json(results, output_dir, args)
    md_path = _write_markdown(results, output_dir, args)
    print(f'[saved] csv={csv_path}')
    print(f'[saved] json={json_path}')
    print(f'[saved] md={md_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
