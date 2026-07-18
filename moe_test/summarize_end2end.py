#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

RESULTS_BASE = Path('/share/zhouyongkang/projects/sc/results/end2end_bs64_maxnnz4')
DATASET_ORDER = [
    ('csqa', 'CSQA'),
    ('gsm8k', 'GSM8K'),
    ('hellaswag', 'HellaSwag'),
    ('piqa', 'PIQA'),
    ('siqa', 'SIQA'),
    ('sst2', 'SST-2'),
    ('alpaca', 'Alpaca'),
    ('sum', 'SUM'),
]
METHOD_ORDER = ['hf', 'eagle', 'mtp', 'bm', 'bmeagle']

TIME_RE = re.compile(r'Average time across (\d+) prompts: ([0-9.]+) seconds')
AL_RE = re.compile(r'Average AL \(Acceptance Length\) across \d+ prompts: ([0-9.]+)')
PEAK_RE = re.compile(r'Peak GPU memory usage: ([0-9.]+) GB')
STATS_RE = re.compile(r'SPMLP path stats:\s+(.*)')
EXIT_RE = re.compile(r'exit_code=(\d+)')


def resolve_result_root(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'result root does not exist: {path}')
        return path
    if not RESULTS_BASE.exists():
        raise FileNotFoundError(f'no default result directory found: {RESULTS_BASE}')
    candidates = [path for path in RESULTS_BASE.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f'no run directories found under: {RESULTS_BASE}')
    return max(candidates, key=lambda item: item.stat().st_mtime)


def parse_stats(stats_blob: str | None) -> dict[str, int | None]:
    parsed = {
        'spmlp_layers': None,
        'kernel_path_calls': None,
        'unsupported_batch_fallback_calls': None,
        'runtime_t_d_fallback_calls': None,
        'original_forward_calls': None,
        'bm_forward_calls': None,
    }
    if not stats_blob:
        return parsed
    for token in stats_blob.split():
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        if key == 'layers':
            key = 'spmlp_layers'
        if key in parsed:
            parsed[key] = int(value)
    return parsed


def parse_log(log_path: Path, dataset_slug: str, dataset_display: str, method: str) -> dict[str, object]:
    result: dict[str, object] = {
        'dataset_slug': dataset_slug,
        'dataset_display': dataset_display,
        'method': method,
        'log_path': str(log_path),
        'status': 'missing',
        'num_prompts': None,
        'average_time_seconds': None,
        'average_al': None,
        'peak_gpu_gb': None,
        'spmlp_layers': None,
        'kernel_path_calls': None,
        'unsupported_batch_fallback_calls': None,
        'runtime_t_d_fallback_calls': None,
        'original_forward_calls': None,
        'bm_forward_calls': None,
        'exit_code': None,
    }
    if not log_path.exists():
        return result

    text = log_path.read_text(encoding='utf-8', errors='replace')
    result['status'] = 'present'

    time_match = TIME_RE.search(text)
    if time_match:
        result['num_prompts'] = int(time_match.group(1))
        result['average_time_seconds'] = float(time_match.group(2))

    al_match = AL_RE.search(text)
    if al_match:
        result['average_al'] = float(al_match.group(1))

    peak_match = PEAK_RE.search(text)
    if peak_match:
        result['peak_gpu_gb'] = float(peak_match.group(1))

    stats_match = STATS_RE.search(text)
    result.update(parse_stats(stats_match.group(1) if stats_match else None))

    exit_match = EXIT_RE.search(text)
    if exit_match:
        result['exit_code'] = int(exit_match.group(1))

    if result['average_time_seconds'] is not None and result['exit_code'] == 0:
        result['status'] = 'ok'
    elif result['exit_code'] is not None:
        result['status'] = 'failed'

    return result


def safe_speedup(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate == 0.0:
        return None
    return baseline / candidate


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def geometric_mean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0.0]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def write_outputs(result_root: Path, rows: list[dict[str, object]]) -> None:
    summary_dir = result_root / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    long_csv_path = summary_dir / 'summary_long.csv'
    fieldnames = [
        'dataset_slug',
        'dataset_display',
        'method',
        'status',
        'num_prompts',
        'average_time_seconds',
        'average_al',
        'peak_gpu_gb',
        'spmlp_layers',
        'kernel_path_calls',
        'unsupported_batch_fallback_calls',
        'runtime_t_d_fallback_calls',
        'original_forward_calls',
        'bm_forward_calls',
        'exit_code',
        'log_path',
    ]
    with long_csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    by_dataset: dict[str, dict[str, dict[str, object]]] = {slug: {} for slug, _ in DATASET_ORDER}
    for row in rows:
        by_dataset[row['dataset_slug']][row['method']] = row

    wide_csv_path = summary_dir / 'summary_wide.csv'
    wide_fieldnames = [
        'dataset_slug',
        'dataset_display',
        'hf_seconds',
        'eagle_seconds',
        'mtp_seconds',
        'bm_seconds',
        'bmeagle_seconds',
        'hf_al',
        'eagle_al',
        'mtp_al',
        'bm_al',
        'bmeagle_al',
        'hf_vs_eagle',
        'eagle_vs_mtp',
        'eagle_vs_bm',
        'eagle_vs_bmeagle',
    ]
    wide_rows: list[dict[str, object]] = []
    for slug, display in DATASET_ORDER:
        method_rows = by_dataset.get(slug, {})
        hf = method_rows.get('hf', {})
        eagle = method_rows.get('eagle', {})
        mtp = method_rows.get('mtp', {})
        bm = method_rows.get('bm', {})
        bmeagle = method_rows.get('bmeagle', {})
        wide_rows.append(
            {
                'dataset_slug': slug,
                'dataset_display': display,
                'hf_seconds': hf.get('average_time_seconds'),
                'eagle_seconds': eagle.get('average_time_seconds'),
                'mtp_seconds': mtp.get('average_time_seconds'),
                'bm_seconds': bm.get('average_time_seconds'),
                'bmeagle_seconds': bmeagle.get('average_time_seconds'),
                'hf_al': hf.get('average_al'),
                'eagle_al': eagle.get('average_al'),
                'mtp_al': mtp.get('average_al'),
                'bm_al': bm.get('average_al'),
                'bmeagle_al': bmeagle.get('average_al'),
                'hf_vs_eagle': safe_speedup(hf.get('average_time_seconds'), eagle.get('average_time_seconds')),
                'eagle_vs_mtp': safe_speedup(eagle.get('average_time_seconds'), mtp.get('average_time_seconds')),
                'eagle_vs_bm': safe_speedup(eagle.get('average_time_seconds'), bm.get('average_time_seconds')),
                'eagle_vs_bmeagle': safe_speedup(eagle.get('average_time_seconds'), bmeagle.get('average_time_seconds')),
            }
        )
    with wide_csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fieldnames)
        writer.writeheader()
        for row in wide_rows:
            writer.writerow(row)

    overall_latency: dict[str, list[float]] = {method: [] for method in METHOD_ORDER}
    eagle_vs_mtp_values: list[float] = []
    eagle_vs_bm_values: list[float] = []
    eagle_vs_bmeagle_values: list[float] = []
    for row in wide_rows:
        for method in METHOD_ORDER:
            latency_value = row.get(f'{method}_seconds')
            if isinstance(latency_value, float):
                overall_latency[method].append(latency_value)
        if isinstance(row.get('eagle_vs_mtp'), float):
            eagle_vs_mtp_values.append(row['eagle_vs_mtp'])
        if isinstance(row.get('eagle_vs_bm'), float):
            eagle_vs_bm_values.append(row['eagle_vs_bm'])
        if isinstance(row.get('eagle_vs_bmeagle'), float):
            eagle_vs_bmeagle_values.append(row['eagle_vs_bmeagle'])

    summary_json_path = summary_dir / 'summary.json'
    summary_payload = {
        'result_root': str(result_root),
        'datasets': wide_rows,
        'overall': {
            'mean_latency_seconds': {
                method: (mean(values) if values else None) for method, values in overall_latency.items()
            },
            'geomean_speedup_vs_eagle': {
                'mtp': geometric_mean(eagle_vs_mtp_values),
                'bm': geometric_mean(eagle_vs_bm_values),
                'bmeagle': geometric_mean(eagle_vs_bmeagle_values),
            },
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    md_lines: list[str] = []
    md_lines.append('# End-to-End Summary: bs64 + maxnnz4')
    md_lines.append('')
    md_lines.append(f'- result root: `{result_root}`')
    md_lines.append('')
    md_lines.append('## Average Latency (seconds per prompt)')
    md_lines.append('')
    md_lines.append('| dataset | hf | eagle | mtp | bm | bmeagle |')
    md_lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for row in wide_rows:
        md_lines.append(
            f"| {row['dataset_display']} | {format_float(row['hf_seconds'])} | {format_float(row['eagle_seconds'])} | {format_float(row['mtp_seconds'])} | {format_float(row['bm_seconds'])} | {format_float(row['bmeagle_seconds'])} |"
        )
    md_lines.append('')
    md_lines.append('## Average Acceptance Length')
    md_lines.append('')
    md_lines.append('| dataset | eagle | mtp | bmeagle |')
    md_lines.append('| --- | ---: | ---: | ---: |')
    for row in wide_rows:
        md_lines.append(
            f"| {row['dataset_display']} | {format_float(row['eagle_al'])} | {format_float(row['mtp_al'])} | {format_float(row['bmeagle_al'])} |"
        )
    md_lines.append('')
    md_lines.append('## Speedup')
    md_lines.append('')
    md_lines.append('| dataset | hf / eagle | eagle / mtp | eagle / bm | eagle / bmeagle |')
    md_lines.append('| --- | ---: | ---: | ---: | ---: |')
    for row in wide_rows:
        md_lines.append(
            f"| {row['dataset_display']} | {format_float(row['hf_vs_eagle'])}x | {format_float(row['eagle_vs_mtp'])}x | {format_float(row['eagle_vs_bm'])}x | {format_float(row['eagle_vs_bmeagle'])}x |"
        )
    md_lines.append('')
    md_lines.append('## Overall')
    md_lines.append('')
    md_lines.append('- Mean latency across datasets:')
    for method in METHOD_ORDER:
        md_lines.append(f"  - `{method}`: `{format_float(summary_payload['overall']['mean_latency_seconds'][method])} s`")
    md_lines.append('- Geometric mean speedup vs eagle:')
    md_lines.append(f"  - `mtp`: `{format_float(summary_payload['overall']['geomean_speedup_vs_eagle']['mtp'])}x`")
    md_lines.append(f"  - `bm`: `{format_float(summary_payload['overall']['geomean_speedup_vs_eagle']['bm'])}x`")
    md_lines.append(f"  - `bmeagle`: `{format_float(summary_payload['overall']['geomean_speedup_vs_eagle']['bmeagle'])}x`")
    md_lines.append('')
    md_lines.append('## Files')
    md_lines.append('')
    md_lines.append(f'- long csv: `{long_csv_path}`')
    md_lines.append(f'- wide csv: `{wide_csv_path}`')
    md_lines.append(f'- json: `{summary_json_path}`')

    summary_md_path = summary_dir / 'summary.md'
    summary_md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize end-to-end benchmark logs.')
    parser.add_argument('--result-root', type=str, default=None, help='Path to a specific run directory.')
    args = parser.parse_args()

    result_root = resolve_result_root(args.result_root)
    rows: list[dict[str, object]] = []
    for dataset_slug, dataset_display in DATASET_ORDER:
        dataset_dir = result_root / dataset_slug
        for method in METHOD_ORDER:
            rows.append(parse_log(dataset_dir / f'{method}.log', dataset_slug, dataset_display, method))

    write_outputs(result_root, rows)
    print(f'summarized results under {result_root / "summary"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
