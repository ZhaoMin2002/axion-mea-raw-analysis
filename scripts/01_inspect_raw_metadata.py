"""Inspect Axion MEA .raw metadata without loading the full recording into RAM.

This script is the first step of the simplified Plate 8 / 4039 workflow.
It does not filter electrodes, does not perform QC, and does not modify data.
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
from pathlib import Path
from typing import Any

import numpy as np


RAW_MARKER = b"Raw\x06\xac\x18\x9b"
FOOTER_SEARCH_BYTES = 1_000_000
DEFAULT_WELL_SHAPE = (6, 8)
DEFAULT_ELECTRODE_SHAPE = (4, 4)


def _parse_value_and_unit(field: str) -> tuple[float, str]:
    """Parse strings such as 'Sampling Frequency, 12.5 kHz'."""
    parts = field.split(",", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"无法解析元数据字段: {field!r}")

    value_unit = parts[1].strip().split()
    if len(value_unit) < 2:
        raise ValueError(f"元数据字段缺少数值或单位: {field!r}")

    return float(value_unit[0]), value_unit[1]


def _sampling_rate_hz(value: float, unit: str) -> float:
    unit_lower = unit.lower()
    if unit_lower == "khz":
        return value * 1000.0
    if unit_lower == "hz":
        return value
    raise ValueError(f"无法识别采样率单位: {unit!r}")


def _extract_well_labels(footer: bytes, well_shape: tuple[int, int]) -> list[list[str]] | None:
    """Use the same label pattern as the original read_raw implementation."""
    pattern = r"\\x0.\\x00\\x00\\x01\\x.[^3]\\x..\\x..\\x0.\\x00\\x00\\x00.*?x0"
    matches = re.findall(pattern, str(footer))
    labels = [item[44:-3] for item in matches]

    expected = int(np.prod(well_shape))
    if len(labels) != expected:
        print(f"[警告] well label 数量为 {len(labels)}，预期为 {expected}。")
        print("[警告] 当前 RAW 的标签正则解析没有得到完整 6×8 plate map。")
        return None

    return np.asarray(labels, dtype=object).reshape(well_shape).tolist()


def inspect_raw(
    raw_path: Path,
    output_path: Path,
    well_shape: tuple[int, int] = DEFAULT_WELL_SHAPE,
    electrode_shape: tuple[int, int] = DEFAULT_ELECTRODE_SHAPE,
) -> dict[str, Any]:
    """Inspect metadata and estimate recording duration using a memory map."""
    if not raw_path.is_file():
        raise FileNotFoundError(f"找不到 RAW 文件: {raw_path}")

    file_size = raw_path.stat().st_size
    n_wells = int(np.prod(well_shape))
    n_electrodes = int(np.prod(electrode_shape))
    n_channels = n_wells * n_electrodes

    with raw_path.open("rb") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            marker_pos = mapped.find(RAW_MARKER)
            if marker_pos < 0:
                raise ValueError("没有找到 Axion RAW 数据标记。此文件版本可能与原代码不兼容。")

            signal_start = marker_pos + len(RAW_MARKER)
            meta_raw = mapped[:marker_pos]
            meta_text = str(meta_raw)

            fs_match = re.search(r"Sampling Frequency,.*?Hz", meta_text)
            scale_match = re.search(r"Scale,.*?V/sample", meta_text)
            if fs_match is None:
                raise ValueError("没有在 RAW 元数据中找到 Sampling Frequency。")
            if scale_match is None:
                raise ValueError("没有在 RAW 元数据中找到 Scale。")

            fs_value, fs_unit = _parse_value_and_unit(fs_match.group(0))
            scale_value, scale_unit = _parse_value_and_unit(scale_match.group(0))
            fs_hz = _sampling_rate_hz(fs_value, fs_unit)

            tail_size = min(FOOTER_SEARCH_BYTES, file_size)
            tail = mapped[file_size - tail_size : file_size]
            footer_match = re.search(b".{17}\x81h\xackQ\?M", tail)

            if footer_match is None:
                print("[警告] 没有找到原代码使用的文件尾元数据标记。")
                print("[警告] 将暂时把文件末尾视为信号末尾，时长估计可能包含 footer。")
                footer_bytes = 0
                footer = b""
            else:
                footer_bytes = tail_size - footer_match.start()
                footer = mapped[file_size - footer_bytes : file_size]

            signal_end = file_size - footer_bytes
            signal_bytes = signal_end - signal_start
            if signal_bytes <= 0:
                raise ValueError("计算得到的信号数据长度不合理。")

            trailing_byte = signal_bytes % 2
            total_int16_values = signal_bytes // 2
            complete_timepoints = total_int16_values // n_channels
            channel_remainder = total_int16_values % n_channels
            duration_seconds = complete_timepoints / fs_hz

            well_labels = _extract_well_labels(footer, well_shape) if footer else None

    result: dict[str, Any] = {
        "raw_path": str(raw_path.resolve()),
        "file_size_bytes": file_size,
        "file_size_gib": file_size / (1024**3),
        "raw_marker_offset": marker_pos,
        "signal_start_offset": signal_start,
        "footer_bytes": footer_bytes,
        "signal_bytes": signal_bytes,
        "trailing_byte_after_int16": trailing_byte,
        "well_shape": list(well_shape),
        "electrode_shape": list(electrode_shape),
        "n_wells": n_wells,
        "n_electrodes_per_well": n_electrodes,
        "n_channels": n_channels,
        "sampling_rate_value": fs_value,
        "sampling_rate_unit": fs_unit,
        "sampling_rate_hz": fs_hz,
        "scale_value": scale_value,
        "scale_unit": scale_unit,
        "complete_timepoints_per_channel": complete_timepoints,
        "channel_remainder_int16_values": channel_remainder,
        "estimated_duration_seconds": duration_seconds,
        "estimated_duration_minutes": duration_seconds / 60.0,
        "well_labels": well_labels,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, ensure_ascii=False, indent=2)

    return result


def _print_result(result: dict[str, Any], output_path: Path) -> None:
    print("\n=== Axion RAW metadata inspection ===")
    print(f"RAW 文件: {result['raw_path']}")
    print(f"文件大小: {result['file_size_gib']:.3f} GiB")
    print(
        "采样率: "
        f"{result['sampling_rate_value']} {result['sampling_rate_unit']} "
        f"({result['sampling_rate_hz']:.1f} Hz)"
    )
    print(f"Scale: {result['scale_value']} {result['scale_unit']}")
    print(
        "板型: "
        f"{result['well_shape'][0]}×{result['well_shape'][1]} wells, "
        f"每孔 {result['electrode_shape'][0]}×{result['electrode_shape'][1]} electrodes"
    )
    print(f"估算时长: {result['estimated_duration_seconds']:.2f} 秒")
    print(f"估算时长: {result['estimated_duration_minutes']:.2f} 分钟")
    print(f"通道整除余数: {result['channel_remainder_int16_values']}")

    labels = result.get("well_labels")
    if labels is None:
        print("well_labels: 未成功解析。后续需要使用外部 plate map。")
    else:
        print("well_labels:")
        for row in labels:
            print("  " + " | ".join(str(value) for value in row))

    print(f"\n结果已保存: {output_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 Axion MEA RAW 元数据，不加载完整信号到内存。"
    )
    parser.add_argument("--raw", type=Path, required=True, help=".raw 文件路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/plate8_4039/raw_metadata.json"),
        help="JSON 输出路径",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = inspect_raw(raw_path=args.raw, output_path=args.output)
    _print_result(result, args.output)


if __name__ == "__main__":
    main()
