from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
MOE_TEST_DIR = Path(__file__).resolve().parent
MOE_SRC_DIR = ROOT_DIR / "moe_src"
SWEEP_CSV_PATH = ROOT_DIR / "moe_src" / "test" / "torch_cuda_hybrid_td_sweep.csv"

BASE_MODEL_PATH = "/share/public/public_models/Qwen3-30B-A3B"
EAGLE_MODEL_PATH = "/share/zhouyongkang/models/qwen3_30b_moe_eagle3"
DATASET_NAMES = ["alpaca", "commonsense_qa", "gsm8k", "hellaswag", "piqa", "siqa", "sst2", "sum"]

if str(MOE_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(MOE_TEST_DIR))
if str(MOE_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MOE_SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import CustomTextDataset, resolve_dataset_path
from EAGLE.eagle.model.ea_model import EaModel
from mlp import SPMLP


@dataclass(frozen=True)
class SweepEntry:
    batch_size: int
    maxnnz: int
    t_d: int
    actual_t_d: int
    bm_ms: float
    ours_ms: float
    winner: str


class RuntimeTDCollector:
    def __init__(self) -> None:
        self.current_prompt_idx = -1
        self.global_call_idx = 0
        self.per_layer: dict[int, list[int]] = defaultdict(list)
        self.records: list[tuple[int, int, int, int]] = []

    def set_prompt_idx(self, prompt_idx: int) -> None:
        self.current_prompt_idx = prompt_idx

    def install(self, model) -> None:
        for layer_idx, layer in enumerate(model.base_model.model.layers):
            mlp = layer.mlp
            original = mlp.mixer_with_fallback

            def wrapped(router_logits, _orig=original, _layer_idx=layer_idx):
                output = _orig(router_logits)
                runtime_t_d = int(output["runtime_t_d"])
                self.per_layer[_layer_idx].append(runtime_t_d)
                self.records.append(
                    (
                        self.current_prompt_idx,
                        _layer_idx,
                        self.global_call_idx,
                        runtime_t_d,
                    )
                )
                self.global_call_idx += 1
                return output

            mlp.mixer_with_fallback = wrapped


def _percentile_higher(values: Iterable[int], ratio: float) -> int:
    ordered = sorted(int(v) for v in values)
    if not ordered:
        raise ValueError("Cannot compute percentile of an empty sequence")
    rank = max(1, math.ceil(ratio * len(ordered)))
    return ordered[rank - 1]


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values))


def _median(values: list[int]) -> float:
    return float(statistics.median(values))


def _load_sweep_entries(path: Path) -> list[SweepEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {path}")

    entries: list[SweepEntry] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                SweepEntry(
                    batch_size=int(row["batch_size"]),
                    maxnnz=int(row["maxnnz"]),
                    t_d=int(row["t_d"]),
                    actual_t_d=int(row["actual_t_d"]),
                    bm_ms=float(row["bmm_ms"]),
                    ours_ms=float(row["hybrid_ms"]),
                    winner=row["winner"],
                )
            )
    return entries


def _build_mtp_model(*, t_d: int, adaptive_t_d: bool, bm_fallback_t_d: int | None):
    model = EaModel.from_pretrained(
        base_model_path=BASE_MODEL_PATH,
        ea_model_path=EAGLE_MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=64,
    )
    model.eval()
    model.device = model.base_model.device

    for layer in tqdm(model.base_model.model.layers, desc="Applying SPMLP to layers"):
        layer.mlp = SPMLP(
            layer.mlp,
            forward_mode="main",
            t_d=t_d,
            adaptive_t_d=adaptive_t_d,
            bm_fallback_t_d=bm_fallback_t_d,
        )

    return model


def _collect_runtime_t_d(
    *,
    dataset_idx: int,
    num_prompts: int,
    max_new_tokens: int,
    t_d: int,
    adaptive_t_d: bool,
    bm_fallback_t_d: int | None,
):
    dataset_name = DATASET_NAMES[dataset_idx]
    dataset_path = resolve_dataset_path(f"benchmark/{dataset_name}")
    dataset = CustomTextDataset(dataset_path)

    model = _build_mtp_model(
        t_d=t_d,
        adaptive_t_d=adaptive_t_d,
        bm_fallback_t_d=bm_fallback_t_d,
    )
    tokenizer = model.tokenizer
    collector = RuntimeTDCollector()
    collector.install(model)

    prompt_count = min(num_prompts, len(dataset))
    for prompt_idx in tqdm(range(prompt_count), desc="Collecting runtime_t_d"):
        prompt = dataset[prompt_idx]
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        collector.set_prompt_idx(prompt_idx)
        with torch.no_grad():
            model.eagenerate(input_ids, max_new_tokens=max_new_tokens)

    first_layer = model.base_model.model.layers[0].mlp
    hyperparams = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "num_prompts": prompt_count,
        "max_new_tokens": max_new_tokens,
        "spmlp_t_d_arg": t_d,
        "adaptive_t_d": adaptive_t_d,
        "bm_fallback_t_d": bm_fallback_t_d,
        "hidden_size": int(first_layer.hidden_size),
        "num_experts": int(first_layer.num_experts),
        "top_k": int(first_layer.top_k),
        "intermediate_size_per_expert": int(first_layer.intermediate_size),
        "maxnnz": int(first_layer.maxnnz),
        "num_layers": int(len(model.base_model.model.layers)),
        "base_model_path": BASE_MODEL_PATH,
        "eagle_model_path": EAGLE_MODEL_PATH,
    }

    del model
    torch.cuda.empty_cache()
    return collector, hyperparams


def _build_layer_stats(collector: RuntimeTDCollector) -> list[dict[str, float | int]]:
    layer_stats: list[dict[str, float | int]] = []
    for layer_idx in sorted(collector.per_layer):
        values = collector.per_layer[layer_idx]
        layer_stats.append(
            {
                "layer_idx": layer_idx,
                "mean_runtime_t_d": _mean(values),
                "median_runtime_t_d": _median(values),
                "p90_runtime_t_d": _percentile_higher(values, 0.90),
                "p95_runtime_t_d": _percentile_higher(values, 0.95),
                "p99_runtime_t_d": _percentile_higher(values, 0.99),
                "min_runtime_t_d": min(values),
                "max_runtime_t_d": max(values),
                "num_samples": len(values),
            }
        )
    return layer_stats


def _write_runtime_raw_csv(path: Path, collector: RuntimeTDCollector) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_idx", "layer_idx", "global_call_idx", "runtime_t_d"])
        writer.writerows(collector.records)


def _write_layer_stats_csv(path: Path, layer_stats: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "layer_idx",
        "mean_runtime_t_d",
        "median_runtime_t_d",
        "p90_runtime_t_d",
        "p95_runtime_t_d",
        "p99_runtime_t_d",
        "min_runtime_t_d",
        "max_runtime_t_d",
        "num_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in layer_stats:
            writer.writerow(row)


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _excel_col(column_idx: int) -> str:
    letters: list[str] = []
    while column_idx > 0:
        column_idx, rem = divmod(column_idx - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def _cell_xml(row_idx: int, col_idx: int, value) -> str:
    cell_ref = f"{_excel_col(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, bool):
        value = "TRUE" if value else "FALSE"
    if isinstance(value, int) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    if isinstance(value, float):
        return f'<c r="{cell_ref}"><v>{value:.12g}</v></c>'
    text = _xml_escape(str(value))
    return (
        f'<c r="{cell_ref}" t="inlineStr">'
        f'<is><t xml:space="preserve">{text}</t></is>'
        f"</c>"
    )


def _sheet_xml(rows: list[list[object]]) -> str:
    row_xml: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cell_xml = "".join(_cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row, start=1))
        row_xml.append(f'<row r="{row_idx}">{cell_xml}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(row_xml)
        + '</sheetData></worksheet>'
    )


def _write_simple_xlsx(path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for sheet_idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{sheet_idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append('</Types>')

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for sheet_idx, (sheet_name, _) in enumerate(sheets, start=1):
        safe_name = _xml_escape(sheet_name)
        workbook_sheets.append(
            f'<sheet name="{safe_name}" sheetId="{sheet_idx}" r:id="rId{sheet_idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{sheet_idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{sheet_idx}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append('</Relationships>')

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        + ''.join(workbook_sheets)
        + '</sheets></workbook>'
    )

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1">
    <font>
      <sz val="11"/>
      <name val="Calibri"/>
      <family val="2"/>
    </font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>
"""

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', '\n'.join(content_types))
        zf.writestr('_rels/.rels', root_rels)
        zf.writestr('xl/workbook.xml', workbook_xml)
        zf.writestr('xl/_rels/workbook.xml.rels', '\n'.join(workbook_rels))
        zf.writestr('xl/styles.xml', styles_xml)
        for sheet_idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f'xl/worksheets/sheet{sheet_idx}.xml', _sheet_xml(rows))


def _write_workbook(
    path: Path,
    *,
    sweep_entries: list[SweepEntry],
    layer_stats: list[dict[str, float | int]],
    global_stats: dict[str, int | float | str],
    hyperparams: dict[str, int | str | bool | None],
) -> None:
    sheets: list[tuple[str, list[list[object]]]] = []

    ws_meta: list[list[object]] = [["key", "value"]]
    for key, value in hyperparams.items():
        ws_meta.append([key, value])
    ws_meta.append(["sweep_csv", str(SWEEP_CSV_PATH)])
    ws_meta.append(["note", "latency sheets use columns A/B as bm/ours for direct copy"])
    sheets.append(("metadata", ws_meta))

    ws_summary: list[list[object]] = [["metric", "value"]]
    for key, value in global_stats.items():
        ws_summary.append([key, value])
    sheets.append(("runtime_td_summary", ws_summary))

    layer_headers = [
        "layer_idx",
        "mean_runtime_t_d",
        "median_runtime_t_d",
        "p90_runtime_t_d",
        "p95_runtime_t_d",
        "p99_runtime_t_d",
        "min_runtime_t_d",
        "max_runtime_t_d",
        "num_samples",
    ]
    ws_layer: list[list[object]] = [layer_headers]
    for row in layer_stats:
        ws_layer.append([row[h] for h in layer_headers])
    sheets.append(("runtime_td_layer_mean", ws_layer))

    configs = [(64, 4), (64, 8), (128, 4), (128, 8)]
    wide_header = ["t_d"]
    for batch_size, maxnnz in configs:
        wide_header.extend([f"bs{batch_size}_nnz{maxnnz}_bm_ms", f"bs{batch_size}_nnz{maxnnz}_ours_ms"])
    ws_wide: list[list[object]] = [wide_header]

    grouped: dict[tuple[int, int], dict[int, SweepEntry]] = defaultdict(dict)
    for entry in sweep_entries:
        grouped[(entry.batch_size, entry.maxnnz)][entry.t_d] = entry

    for t_d in range(1, 65):
        row = [t_d]
        for config in configs:
            entry = grouped[config][t_d]
            row.extend([entry.bm_ms, entry.ours_ms])
        ws_wide.append(row)
    sheets.append(("latency_wide", ws_wide))

    for batch_size, maxnnz in configs:
        ws: list[list[object]] = [["bm_ms", "ours_ms", "t_d", "actual_t_d", "winner"]]
        for t_d in range(1, 65):
            entry = grouped[(batch_size, maxnnz)][t_d]
            ws.append([entry.bm_ms, entry.ours_ms, entry.t_d, entry.actual_t_d, entry.winner])
        sheets.append((f"bs{batch_size}_nnz{maxnnz}", ws))

    _write_simple_xlsx(path, sheets)


def _write_markdown(
    path: Path,
    *,
    sweep_entries: list[SweepEntry],
    layer_stats: list[dict[str, float | int]],
    global_stats: dict[str, int | float | str],
    hyperparams: dict[str, int | str | bool | None],
    workbook_path: Path,
    raw_csv_path: Path,
    layer_csv_path: Path,
) -> None:
    grouped: dict[tuple[int, int], list[SweepEntry]] = defaultdict(list)
    for entry in sweep_entries:
        grouped[(entry.batch_size, entry.maxnnz)].append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda item: item.t_d)

    lines: list[str] = []
    lines.append('# Paper Numbers')
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append(f'- Workbook: `{workbook_path}`')
    lines.append(f'- Raw runtime_t_d samples: `{raw_csv_path}`')
    lines.append(f'- Layer runtime_t_d stats: `{layer_csv_path}`')
    lines.append('')
    lines.append('## runtime_t_d Summary')
    lines.append('')
    for key, value in global_stats.items():
        lines.append(f'- {key}: {value}')
    lines.append('')
    lines.append('## runtime_t_d Hyperparameters')
    lines.append('')
    for key, value in hyperparams.items():
        lines.append(f'- {key}: {value}')
    lines.append('')
    lines.append('## runtime_t_d Layer Means')
    lines.append('')
    lines.append('```csv')
    lines.append(
        'layer_idx,mean_runtime_t_d,median_runtime_t_d,p90_runtime_t_d,p95_runtime_t_d,p99_runtime_t_d,min_runtime_t_d,max_runtime_t_d,num_samples'
    )
    for row in layer_stats:
        lines.append(
            ','.join(
                [
                    str(row['layer_idx']),
                    f"{row['mean_runtime_t_d']:.6f}",
                    f"{row['median_runtime_t_d']:.6f}",
                    str(row['p90_runtime_t_d']),
                    str(row['p95_runtime_t_d']),
                    str(row['p99_runtime_t_d']),
                    str(row['min_runtime_t_d']),
                    str(row['max_runtime_t_d']),
                    str(row['num_samples']),
                ]
            )
        )
    lines.append('```')
    lines.append('')
    lines.append('## t_d Sweep Latency')
    lines.append('')

    for batch_size, maxnnz in [(64, 4), (64, 8), (128, 4), (128, 8)]:
        lines.append(f'### batch_size={batch_size}, maxnnz={maxnnz}')
        lines.append('')
        lines.append('```csv')
        lines.append('t_d,bm_ms,ours_ms,actual_t_d,winner')
        for entry in grouped[(batch_size, maxnnz)]:
            lines.append(
                f"{entry.t_d},{entry.bm_ms:.8f},{entry.ours_ms:.8f},{entry.actual_t_d},{entry.winner}"
            )
        lines.append('```')
        lines.append('')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Export paper-friendly latency and runtime_t_d tables.')
    parser.add_argument('--dataset', type=int, default=0, choices=range(len(DATASET_NAMES)))
    parser.add_argument('--num-prompts', type=int, default=30)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--spmlp-t-d', type=int, default=64)
    parser.add_argument('--spmlp-adaptive-td', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--spmlp-bm-fallback-td', type=int, default=-1)
    parser.add_argument('--output-dir', type=Path, default=ROOT_DIR / 'results' / 'paper_numbers')
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = output_dir / 'paper_runtime_td_numbers.xlsx'
    markdown_path = output_dir / 'paper_runtime_td_numbers.md'
    raw_csv_path = output_dir / 'runtime_td_samples.csv'
    layer_csv_path = output_dir / 'runtime_td_layer_stats.csv'
    summary_json_path = output_dir / 'runtime_td_summary.json'

    bm_fallback_t_d = None if args.spmlp_bm_fallback_td < 0 else args.spmlp_bm_fallback_td

    print(f'Loading sweep data from {SWEEP_CSV_PATH}')
    sweep_entries = _load_sweep_entries(SWEEP_CSV_PATH)

    print('Collecting runtime_t_d samples from mtp generation')
    collector, hyperparams = _collect_runtime_t_d(
        dataset_idx=args.dataset,
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        t_d=args.spmlp_t_d,
        adaptive_t_d=args.spmlp_adaptive_td,
        bm_fallback_t_d=bm_fallback_t_d,
    )

    if not collector.records:
        raise RuntimeError('No runtime_t_d samples were collected')

    all_values = [runtime_t_d for _, _, _, runtime_t_d in collector.records]
    layer_stats = _build_layer_stats(collector)

    p90 = _percentile_higher(all_values, 0.90)
    p95 = _percentile_higher(all_values, 0.95)
    p99 = _percentile_higher(all_values, 0.99)
    global_stats: dict[str, int | float | str] = {
        'total_runtime_t_d_samples': len(all_values),
        'layer_count_with_samples': len(layer_stats),
        'global_mean_runtime_t_d': _mean(all_values),
        'global_median_runtime_t_d': _median(all_values),
        'p90_runtime_t_d_leq': p90,
        'p95_runtime_t_d_leq': p95,
        'p99_runtime_t_d_leq': p99,
        'global_min_runtime_t_d': min(all_values),
        'global_max_runtime_t_d': max(all_values),
    }

    _write_runtime_raw_csv(raw_csv_path, collector)
    _write_layer_stats_csv(layer_csv_path, layer_stats)
    _write_workbook(
        workbook_path,
        sweep_entries=sweep_entries,
        layer_stats=layer_stats,
        global_stats=global_stats,
        hyperparams=hyperparams,
    )
    _write_markdown(
        markdown_path,
        sweep_entries=sweep_entries,
        layer_stats=layer_stats,
        global_stats=global_stats,
        hyperparams=hyperparams,
        workbook_path=workbook_path,
        raw_csv_path=raw_csv_path,
        layer_csv_path=layer_csv_path,
    )
    summary_json_path.write_text(
        json.dumps(
            {
                'global_stats': global_stats,
                'hyperparams': hyperparams,
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )

    print(f'Workbook written to {workbook_path}')
    print(f'Markdown summary written to {markdown_path}')
    print(f'Raw runtime_t_d CSV written to {raw_csv_path}')
    print(f'Layer runtime_t_d CSV written to {layer_csv_path}')
    print(f'JSON summary written to {summary_json_path}')
    print(
        'runtime_t_d percentiles: '
        f'p90<={p90}, p95<={p95}, p99<={p99}'
    )


if __name__ == '__main__':
    main()
