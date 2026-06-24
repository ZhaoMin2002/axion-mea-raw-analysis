"""Extract a short Axion MEA segment and create diagnostic electrode heatmaps.

This script deliberately does not remove or filter electrodes. It uses all 16
channels per well and generates activity summaries only for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


DEFAULT_METADATA = Path("outputs/plate8_4039/raw_metadata.json")
DEFAULT_CHANNEL_ARRAY = Path("ad_organoids/channel_array.mat")
DEFAULT_PLATE_MAP = Path("resources/plate_maps/plate8_4039_platemap.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/plate8_4039/short_segment")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_plate_map(path: Path | None, well_shape: tuple[int, int]) -> dict[str, dict[str, str]]:
    """Load a CSV with columns such as well, group, include and raw_label."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"找不到 plate map CSV: {path}")

    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None or "well" not in reader.fieldnames:
            raise ValueError("plate map CSV 至少需要 'well' 列。")
        for row in reader:
            well = str(row.get("well", "")).strip().upper()
            if not well:
                continue
            mapping[well] = {key: str(value or "").strip() for key, value in row.items()}

    expected = int(np.prod(well_shape))
    if len(mapping) != expected:
        print(f"[警告] plate map 包含 {len(mapping)} 个孔，预期为 {expected} 个孔。")
    return mapping


def load_channel_indices(
    channel_array_path: Path,
    well_shape: tuple[int, int],
    electrode_shape: tuple[int, int],
) -> np.ndarray:
    """Return raw-index coordinates for every logical well/electrode location."""
    if not channel_array_path.is_file():
        raise FileNotFoundError(f"找不到 channel_array.mat: {channel_array_path}")

    channel_array = loadmat(channel_array_path)
    if "out" not in channel_array:
        raise KeyError("channel_array.mat 中没有找到 'out' 结构体。")

    out = channel_array["out"]
    if out.dtype.names is None:
        raise ValueError("channel_array.mat 的 'out' 不是预期的 MATLAB struct。")

    row_names = list(out.dtype.names)
    channel_dict: dict[str, np.ndarray] = {}
    for name, values in zip(row_names, out[0][0]):
        channel_dict[name] = values[0]

    required = {"WellRow", "WellColumn", "ElectrodeRow", "ElectrodeColumn"}
    missing = required.difference(channel_dict)
    if missing:
        raise KeyError(f"channel_array.mat 缺少字段: {sorted(missing)}")

    n_rows_well, n_cols_well = well_shape
    n_rows_electrodes, n_cols_electrodes = electrode_shape
    shape_rev = (
        n_cols_electrodes,
        n_rows_electrodes,
        n_cols_well,
        n_rows_well,
    )

    wrow_inds = channel_dict["WellRow"].reshape(*shape_rev).T
    wcol_inds = channel_dict["WellColumn"].reshape(*shape_rev).T
    erow_inds = channel_dict["ElectrodeRow"].reshape(*shape_rev).T
    ecol_inds = channel_dict["ElectrodeColumn"].reshape(*shape_rev).T

    locs = np.zeros((n_rows_well, n_cols_well, n_rows_electrodes, n_cols_electrodes, 4), dtype=int)
    seen: set[tuple[int, int, int, int]] = set()

    for wr in range(n_rows_well):
        for wc in range(n_cols_well):
            for er in range(n_rows_electrodes):
                for ec in range(n_cols_electrodes):
                    loc = np.where(
                        (wrow_inds == wr + 1)
                        & (wcol_inds == wc + 1)
                        & (erow_inds == er + 1)
                        & (ecol_inds == ec + 1)
                    )
                    coordinates = np.array(loc).T
                    if coordinates.shape[0] != 1:
                        raise ValueError(
                            "channel_array 映射不是唯一的: "
                            f"well=({wr + 1},{wc + 1}), electrode=({er + 1},{ec + 1}), "
                            f"matches={coordinates.shape[0]}"
                        )
                    coordinate = tuple(int(value) for value in coordinates[0])
                    if coordinate in seen:
                        raise ValueError(f"channel_array 映射包含重复位置: {coordinate}")
                    seen.add(coordinate)
                    locs[wr, wc, er, ec] = coordinate

    return locs.reshape(-1, 4).T


def extract_segment(
    metadata: dict[str, Any],
    channel_array_path: Path,
    start_sec: float,
    duration_sec: float,
) -> tuple[np.ndarray, float, float]:
    """Read and reorder a short segment as microvolts."""
    raw_path = Path(metadata["raw_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"找不到 RAW 文件: {raw_path}")

    fs_hz = float(metadata["sampling_rate_hz"])
    scale_v_per_sample = float(metadata["scale_value"])
    signal_start_offset = int(metadata["signal_start_offset"])
    well_shape = tuple(int(value) for value in metadata["well_shape"])
    electrode_shape = tuple(int(value) for value in metadata["electrode_shape"])
    n_channels = int(metadata["n_channels"])
    max_timepoints = int(metadata["complete_timepoints_per_channel"])

    if start_sec < 0:
        raise ValueError("start_sec 不能小于 0。")
    if duration_sec <= 0:
        raise ValueError("duration_sec 必须大于 0。")

    start_sample = int(round(start_sec * fs_hz))
    n_samples = int(round(duration_sec * fs_hz))
    end_sample = start_sample + n_samples
    if end_sample > max_timepoints:
        raise ValueError(
            f"请求范围超出可用数据: end_sample={end_sample}, "
            f"available={max_timepoints}"
        )

    bytes_per_value = np.dtype("<i2").itemsize
    byte_offset = signal_start_offset + start_sample * n_channels * bytes_per_value
    needed_bytes = n_samples * n_channels * bytes_per_value
    if byte_offset + needed_bytes > raw_path.stat().st_size:
        raise ValueError("请求的 RAW 字节范围超出文件大小。")

    mapped = np.memmap(
        raw_path,
        dtype="<i2",
        mode="r",
        offset=byte_offset,
        shape=(n_samples, n_channels),
        order="C",
    )
    raw_values = np.asarray(mapped).copy()
    del mapped

    n_rows_well, n_cols_well = well_shape
    n_rows_electrodes, n_cols_electrodes = electrode_shape
    shape_rev = (
        n_cols_electrodes,
        n_rows_electrodes,
        n_cols_well,
        n_rows_well,
    )

    scrambled = raw_values.reshape(-1, *shape_rev).T
    del raw_values

    locs = load_channel_indices(channel_array_path, well_shape, electrode_shape)
    ordered_counts = scrambled[locs[0], locs[1], locs[2], locs[3], :]
    ordered_counts = ordered_counts.reshape(
        n_rows_well,
        n_cols_well,
        n_rows_electrodes,
        n_cols_electrodes,
        n_samples,
    )

    signal_uv = ordered_counts.astype(np.float32)
    signal_uv *= np.float32(scale_v_per_sample * 1_000_000.0)
    return signal_uv, fs_hz, scale_v_per_sample


def well_name(row_index: int, col_index: int) -> str:
    return f"{chr(ord('A') + row_index)}{col_index + 1}"


def electrode_name(erow: int, ecol: int) -> str:
    """Follow the original notebook convention: 11, 21, ..., 44."""
    return f"{ecol + 1}{erow + 1}"


def calculate_activity(signal_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute display-only RMS and peak-to-peak values for every electrode."""
    centered = signal_uv - signal_uv.mean(axis=-1, keepdims=True)
    rms_uv = np.sqrt(np.mean(np.square(centered, dtype=np.float64), axis=-1))
    ptp_uv = np.ptp(signal_uv, axis=-1)
    return rms_uv, ptp_uv


def write_electrode_csv(
    output_path: Path,
    rms_uv: np.ndarray,
    ptp_uv: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        fieldnames = ["well", "group", "include", "electrode", "rms_uv", "peak_to_peak_uv"]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for wr in range(rms_uv.shape[0]):
            for wc in range(rms_uv.shape[1]):
                well = well_name(wr, wc)
                row = plate_map.get(well, {})
                for er in range(rms_uv.shape[2]):
                    for ec in range(rms_uv.shape[3]):
                        writer.writerow(
                            {
                                "well": well,
                                "group": row.get("group", ""),
                                "include": row.get("include", ""),
                                "electrode": electrode_name(er, ec),
                                "rms_uv": f"{float(rms_uv[wr, wc, er, ec]):.9g}",
                                "peak_to_peak_uv": f"{float(ptp_uv[wr, wc, er, ec]):.9g}",
                            }
                        )


def plot_electrode_heatmap(
    output_path: Path,
    rms_uv: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> None:
    matrix = rms_uv.reshape(rms_uv.shape[0] * rms_uv.shape[1], -1)
    display = np.log10(np.maximum(matrix, np.finfo(float).tiny))

    well_labels = []
    for wr in range(rms_uv.shape[0]):
        for wc in range(rms_uv.shape[1]):
            well = well_name(wr, wc)
            group = plate_map.get(well, {}).get("group", "")
            well_labels.append(f"{well} {group}".strip())

    electrode_labels = [
        electrode_name(er, ec)
        for er in range(rms_uv.shape[2])
        for ec in range(rms_uv.shape[3])
    ]

    fig, ax = plt.subplots(figsize=(12, 16))
    image = ax.imshow(display, aspect="auto")
    ax.set_title("Electrode activity overview (diagnostic only, no filtering)")
    ax.set_xlabel("Electrode")
    ax.set_ylabel("Well")
    ax.set_xticks(np.arange(len(electrode_labels)), labels=electrode_labels, rotation=90)
    ax.set_yticks(np.arange(len(well_labels)), labels=well_labels)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10 RMS amplitude (µV)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_well_heatmap(
    output_path: Path,
    rms_uv: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> None:
    well_rms = rms_uv.mean(axis=(-2, -1))
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(well_rms, aspect="auto")
    ax.set_title("Mean RMS across all 16 electrodes per well")
    ax.set_xlabel("Well column")
    ax.set_ylabel("Well row")
    ax.set_xticks(np.arange(well_rms.shape[1]), labels=np.arange(1, well_rms.shape[1] + 1))
    ax.set_yticks(
        np.arange(well_rms.shape[0]),
        labels=[chr(ord("A") + index) for index in range(well_rms.shape[0])],
    )

    for wr in range(well_rms.shape[0]):
        for wc in range(well_rms.shape[1]):
            well = well_name(wr, wc)
            group = plate_map.get(well, {}).get("group", "")
            label = f"{well}\n{group}\n{well_rms[wr, wc]:.2f}"
            ax.text(wc, wr, label, ha="center", va="center", fontsize=6)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean electrode RMS (µV)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_example_trace(output_path: Path, signal_uv: np.ndarray, fs_hz: float) -> None:
    n_plot = min(signal_uv.shape[-1], int(round(fs_hz)))
    time_sec = np.arange(n_plot) / fs_hz
    trace = signal_uv[0, 0, 0, 0, :n_plot]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_sec, trace)
    ax.set_title("Example raw trace: A1 electrode 11")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (µV)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    metadata = load_json(args.metadata)
    well_shape = tuple(int(value) for value in metadata["well_shape"])
    plate_map = load_plate_map(args.plate_map, well_shape)

    print("正在读取短片段。此步骤不会读取完整 5.32 GiB RAW 到内存。")
    signal_uv, fs_hz, scale_v_per_sample = extract_segment(
        metadata=metadata,
        channel_array_path=args.channel_array,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal_path = args.output_dir / "segment_signal_uv.npz"
    np.savez_compressed(
        signal_path,
        signal_uv=signal_uv,
        fs_hz=np.asarray(fs_hz),
        start_sec=np.asarray(args.start_sec),
        duration_sec=np.asarray(args.duration_sec),
    )

    rms_uv, ptp_uv = calculate_activity(signal_uv)
    electrode_csv = args.output_dir / "electrode_activity.csv"
    electrode_heatmap = args.output_dir / "electrode_rms_heatmap.png"
    well_heatmap = args.output_dir / "well_rms_heatmap.png"
    example_trace = args.output_dir / "example_trace_A1_E11.png"

    write_electrode_csv(electrode_csv, rms_uv, ptp_uv, plate_map)
    plot_electrode_heatmap(electrode_heatmap, rms_uv, plate_map)
    plot_well_heatmap(well_heatmap, rms_uv, plate_map)
    plot_example_trace(example_trace, signal_uv, fs_hz)

    summary = {
        "raw_path": metadata["raw_path"],
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
        "sampling_rate_hz": fs_hz,
        "scale_v_per_sample": scale_v_per_sample,
        "signal_shape": list(signal_uv.shape),
        "signal_unit": "microvolt",
        "electrode_filtering_applied": False,
        "activity_metric_note": "RMS and peak-to-peak are diagnostic only; no electrode was removed.",
        "output_files": {
            "signal_npz": str(signal_path),
            "electrode_csv": str(electrode_csv),
            "electrode_heatmap": str(electrode_heatmap),
            "well_heatmap": str(well_heatmap),
            "example_trace": str(example_trace),
        },
    }
    summary_path = args.output_dir / "segment_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\n=== 短片段提取完成 ===")
    print(f"signal shape: {signal_uv.shape}")
    print(f"采样率: {fs_hz:.1f} Hz")
    print(f"开始时间: {args.start_sec:.3f} 秒")
    print(f"片段长度: {args.duration_sec:.3f} 秒")
    print("电极筛选: 未执行，全部 16 个电极均保留")
    print(f"结果目录: {args.output_dir.resolve()}")
    print(f"摘要: {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取短 RAW 片段并生成仅用于观察的电极活动 heatmap。"
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--channel-array", type=Path, default=DEFAULT_CHANNEL_ARRAY)
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument("--start-sec", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
