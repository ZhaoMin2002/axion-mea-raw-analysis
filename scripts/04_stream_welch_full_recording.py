"""Run memory-efficient Welch analysis directly from an Axion MEA RAW file.

The script follows the simplified first-pass workflow agreed for Plate 8 / 4039:

- keep all 16 electrodes in every well;
- do not perform electrode QC or channel removal;
- apply the existing 60 Hz notch filter;
- calculate electrode PSDs and average PSD across all 16 electrodes per well;
- calculate descriptive band powers per time window and per well;
- exclude only wells marked include=0 in group-level plots.

Aperiodic 1/f modelling and formal group statistics are deliberately deferred to
a later step. Band powers produced here are total powers, not 1/f-corrected powers.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.io import loadmat


DEFAULT_METADATA = Path("outputs/plate8_4039/raw_metadata.json")
DEFAULT_CHANNEL_ARRAY = Path("ad_organoids/channel_array.mat")
DEFAULT_PLATE_MAP = Path("resources/plate_maps/plate8_4039_platemap.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/plate8_4039/full_recording_welch")

BANDS_HZ: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
    "lfp_1_100": (1.0, 100.0),
}


def well_name(row_index: int, col_index: int) -> str:
    return f"{chr(ord('A') + row_index)}{col_index + 1}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_plate_map(path: Path) -> dict[str, dict[str, str]]:
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
            mapping[well] = {
                key: str(value or "").strip() for key, value in row.items()
            }
    return mapping


def load_raw_channel_indices(
    channel_array_path: Path,
    well_shape: tuple[int, int],
    electrode_shape: tuple[int, int],
) -> np.ndarray:
    """Map logical well/electrode positions to flat RAW channel indices."""
    if not channel_array_path.is_file():
        raise FileNotFoundError(f"找不到 channel_array.mat: {channel_array_path}")

    channel_array = loadmat(channel_array_path)
    if "out" not in channel_array:
        raise KeyError("channel_array.mat 中没有找到 'out' 结构体。")

    out = channel_array["out"]
    if out.dtype.names is None:
        raise ValueError("channel_array.mat 的 'out' 不是预期的 MATLAB struct。")

    channel_dict: dict[str, np.ndarray] = {}
    for name, values in zip(out.dtype.names, out[0][0]):
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

    locs = np.zeros(
        (n_rows_well, n_cols_well, n_rows_electrodes, n_cols_electrodes, 4),
        dtype=int,
    )
    seen_coordinates: set[tuple[int, int, int, int]] = set()

    for wr in range(n_rows_well):
        for wc in range(n_cols_well):
            for er in range(n_rows_electrodes):
                for ec in range(n_cols_electrodes):
                    found = np.where(
                        (wrow_inds == wr + 1)
                        & (wcol_inds == wc + 1)
                        & (erow_inds == er + 1)
                        & (ecol_inds == ec + 1)
                    )
                    coordinates = np.array(found).T
                    if coordinates.shape[0] != 1:
                        raise ValueError(
                            "channel_array 映射不是唯一的: "
                            f"well=({wr + 1},{wc + 1}), "
                            f"electrode=({er + 1},{ec + 1}), "
                            f"matches={coordinates.shape[0]}"
                        )
                    coordinate = tuple(int(value) for value in coordinates[0])
                    if coordinate in seen_coordinates:
                        raise ValueError(f"channel_array 映射包含重复位置: {coordinate}")
                    seen_coordinates.add(coordinate)
                    locs[wr, wc, er, ec] = coordinate

    flattened = locs.reshape(-1, 4).T
    raw_indices = np.ravel_multi_index(
        (flattened[3], flattened[2], flattened[1], flattened[0]),
        dims=shape_rev,
    )
    return raw_indices.reshape(
        n_rows_well,
        n_cols_well,
        n_rows_electrodes,
        n_cols_electrodes,
    )


def make_welch_settings(
    fs_hz: float,
    n_samples: int,
    segment_sec: float,
    overlap_fraction: float,
) -> tuple[int, int]:
    if segment_sec <= 0:
        raise ValueError("welch_segment_sec 必须大于 0。")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("welch_overlap_fraction 必须在 [0, 1) 范围内。")

    nperseg = int(round(segment_sec * fs_hz))
    nperseg = min(nperseg, n_samples)
    if nperseg < 8:
        raise ValueError("Welch segment 太短。")
    noverlap = int(round(nperseg * overlap_fraction))
    noverlap = min(noverlap, nperseg - 1)
    return nperseg, noverlap


def compute_welch(
    traces: np.ndarray,
    fs_hz: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(
        traces,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
    )


def integrate_band(
    freqs: np.ndarray,
    powers: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if np.count_nonzero(mask) < 2:
        return np.full(powers.shape[:-1], np.nan, dtype=float)
    return trapezoid(powers[..., mask], freqs[mask], axis=-1)


def mean_power_near(
    freqs: np.ndarray,
    powers: np.ndarray,
    center_hz: float,
    half_width_hz: float = 1.0,
) -> np.ndarray:
    mask = np.abs(freqs - center_hz) <= half_width_hz
    if not np.any(mask):
        raise ValueError(f"PSD 中没有 {center_hz} Hz 附近的频点。")
    return powers[..., mask].mean(axis=-1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_group_mean_psd(
    path: Path,
    freqs: np.ndarray,
    mean_well_psd: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> None:
    groups: dict[str, list[np.ndarray]] = {}
    for wr in range(mean_well_psd.shape[0]):
        for wc in range(mean_well_psd.shape[1]):
            well = well_name(wr, wc)
            row = plate_map.get(well, {})
            group = row.get("group", "")
            include = row.get("include", "")
            if include != "1" or not group or group.lower() == "empty":
                continue
            groups.setdefault(group, []).append(mean_well_psd[wr, wc])

    mask = (freqs >= 1.0) & (freqs <= 200.0)
    tiny = np.finfo(float).tiny
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, spectra in groups.items():
        group_array = np.stack(spectra, axis=0)
        mean = group_array.mean(axis=0)
        sem = group_array.std(axis=0, ddof=1) / np.sqrt(group_array.shape[0])
        x = freqs[mask]
        y = np.maximum(mean[mask], tiny)
        lower = np.maximum(mean[mask] - sem[mask], tiny)
        upper = np.maximum(mean[mask] + sem[mask], tiny)
        ax.loglog(x, y, label=f"{group} (n={group_array.shape[0]} wells)")
        ax.fill_between(x, lower, upper, alpha=0.15)

    ax.set_title("Group mean PSD after 60 Hz notch (all 16 electrodes per well)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_plate_heatmap(
    path: Path,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    plate_map: dict[str, dict[str, str]],
    log_display: bool,
) -> None:
    display = (
        np.log10(np.maximum(values, np.finfo(float).tiny))
        if log_display
        else values
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(display, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Well column")
    ax.set_ylabel("Well row")
    ax.set_xticks(np.arange(values.shape[1]), labels=np.arange(1, values.shape[1] + 1))
    ax.set_yticks(
        np.arange(values.shape[0]),
        labels=[chr(ord("A") + index) for index in range(values.shape[0])],
    )

    for wr in range(values.shape[0]):
        for wc in range(values.shape[1]):
            well = well_name(wr, wc)
            group = plate_map.get(well, {}).get("group", "")
            ax.text(
                wc,
                wr,
                f"{well}\n{group}\n{values[wr, wc]:.3g}",
                ha="center",
                va="center",
                fontsize=6,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_band_timecourse(
    path: Path,
    window_centers_sec: np.ndarray,
    band_windows: np.ndarray,
    band_name: str,
    plate_map: dict[str, dict[str, str]],
) -> None:
    groups: dict[str, list[tuple[int, int]]] = {}
    for wr in range(band_windows.shape[1]):
        for wc in range(band_windows.shape[2]):
            well = well_name(wr, wc)
            row = plate_map.get(well, {})
            group = row.get("group", "")
            include = row.get("include", "")
            if include != "1" or not group or group.lower() == "empty":
                continue
            groups.setdefault(group, []).append((wr, wc))

    fig, ax = plt.subplots(figsize=(11, 5))
    for group, positions in groups.items():
        values = np.stack(
            [band_windows[:, wr, wc] for wr, wc in positions],
            axis=1,
        )
        mean = values.mean(axis=1)
        sem = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
        ax.plot(window_centers_sec, mean, marker="o", label=group)
        ax.fill_between(window_centers_sec, mean - sem, mean + sem, alpha=0.15)

    low, high = BANDS_HZ[band_name]
    ax.set_title(f"{band_name} power over recording ({low:g}-{high:g} Hz)")
    ax.set_xlabel("Recording time (s)")
    ax.set_ylabel("Total band power (µV²; not 1/f corrected)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    metadata = load_json(args.metadata)
    plate_map = load_plate_map(args.plate_map)

    raw_path = Path(metadata["raw_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"找不到 RAW 文件: {raw_path}")

    fs_hz = float(metadata["sampling_rate_hz"])
    scale_uv_per_sample = float(metadata["scale_value"]) * 1_000_000.0
    signal_start_offset = int(metadata["signal_start_offset"])
    complete_timepoints = int(metadata["complete_timepoints_per_channel"])
    n_channels = int(metadata["n_channels"])
    well_shape = tuple(int(value) for value in metadata["well_shape"])
    electrode_shape = tuple(int(value) for value in metadata["electrode_shape"])

    available_duration_sec = complete_timepoints / fs_hz
    end_sec = available_duration_sec if args.end_sec is None else args.end_sec
    if args.start_sec < 0 or args.start_sec >= available_duration_sec:
        raise ValueError("start_sec 超出可用记录范围。")
    end_sec = min(end_sec, available_duration_sec)
    if end_sec <= args.start_sec:
        raise ValueError("end_sec 必须大于 start_sec。")
    if args.window_sec <= 0 or args.step_sec <= 0:
        raise ValueError("window_sec 和 step_sec 必须大于 0。")

    start_sample = int(round(args.start_sec * fs_hz))
    stop_sample = int(np.floor(end_sec * fs_hz))
    window_samples = int(round(args.window_sec * fs_hz))
    step_samples = int(round(args.step_sec * fs_hz))

    window_start_samples = np.arange(
        start_sample,
        stop_sample - window_samples + 1,
        step_samples,
        dtype=np.int64,
    )
    if window_start_samples.size == 0:
        raise ValueError("指定范围不足以形成一个完整分析窗口。")

    channel_indices = load_raw_channel_indices(
        args.channel_array,
        well_shape,
        electrode_shape,
    )

    nperseg, noverlap = make_welch_settings(
        fs_hz,
        window_samples,
        args.welch_segment_sec,
        args.welch_overlap_fraction,
    )
    b_notch, a_notch = signal.iirnotch(args.notch_hz, args.notch_q, fs=fs_hz)

    raw_memmap = np.memmap(
        raw_path,
        dtype="<i2",
        mode="r",
        offset=signal_start_offset,
        shape=(complete_timepoints, n_channels),
        order="C",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_window_rows: list[dict[str, Any]] = []
    psd_windows: list[np.ndarray] = []
    notch_windows: list[np.ndarray] = []
    band_window_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in BANDS_HZ
    }
    window_centers_sec: list[float] = []
    frequency_axis: np.ndarray | None = None

    total_start = time.perf_counter()
    n_rows_well, n_cols_well = well_shape

    print("=== Streaming Welch analysis ===")
    print(f"RAW: {raw_path}")
    print(f"范围: {args.start_sec:.3f}-{end_sec:.3f} 秒")
    print(f"分析窗口: {args.window_sec:.3f} 秒")
    print(f"窗口步长: {args.step_sec:.3f} 秒")
    print(f"窗口数量: {window_start_samples.size}")
    print(f"Welch segment: {nperseg / fs_hz:.3f} 秒")
    print("电极筛选: 不执行，全部 16 个电极均保留")
    print("1/f 拟合: 本步骤不执行")

    try:
        for window_index, window_start in enumerate(window_start_samples):
            window_end = int(window_start + window_samples)
            start_sec = float(window_start / fs_hz)
            end_window_sec = float(window_end / fs_hz)
            center_sec = (start_sec + end_window_sec) / 2.0
            window_centers_sec.append(center_sec)
            window_timer = time.perf_counter()

            print(
                f"\n窗口 {window_index + 1}/{window_start_samples.size}: "
                f"{start_sec:.3f}-{end_window_sec:.3f} 秒"
            )

            well_psd: np.ndarray | None = None
            notch_reduction = np.empty(well_shape, dtype=np.float64)

            for wr in range(n_rows_well):
                for wc in range(n_cols_well):
                    well = well_name(wr, wc)
                    raw_indices = channel_indices[wr, wc].reshape(-1)

                    counts = np.asarray(
                        raw_memmap[window_start:window_end][:, raw_indices],
                        dtype=np.float32,
                    ).T
                    traces_uv = counts * np.float32(scale_uv_per_sample)
                    del counts

                    filtered_uv = signal.filtfilt(
                        b_notch,
                        a_notch,
                        traces_uv,
                        axis=-1,
                    )

                    freqs_all, raw_electrode_psd = compute_welch(
                        traces_uv,
                        fs_hz,
                        nperseg,
                        noverlap,
                    )
                    freqs_filtered, filtered_electrode_psd = compute_welch(
                        filtered_uv,
                        fs_hz,
                        nperseg,
                        noverlap,
                    )
                    del traces_uv, filtered_uv

                    if not np.array_equal(freqs_all, freqs_filtered):
                        raise RuntimeError("滤波前后的 PSD 频率轴不一致。")

                    freq_mask = (freqs_all >= args.fmin) & (freqs_all <= args.fmax)
                    if np.count_nonzero(freq_mask) < 2:
                        raise ValueError("指定频率范围内没有足够的 PSD 频点。")
                    freqs = freqs_all[freq_mask]
                    raw_mean = raw_electrode_psd[:, freq_mask].mean(axis=0)
                    filtered_mean = filtered_electrode_psd[:, freq_mask].mean(axis=0)
                    del raw_electrode_psd, filtered_electrode_psd

                    if frequency_axis is None:
                        frequency_axis = freqs
                        well_psd = np.empty(
                            (n_rows_well, n_cols_well, freqs.size),
                            dtype=np.float64,
                        )
                    else:
                        if not np.array_equal(frequency_axis, freqs):
                            raise RuntimeError("不同窗口或 well 的频率轴不一致。")
                        if well_psd is None:
                            well_psd = np.empty(
                                (n_rows_well, n_cols_well, freqs.size),
                                dtype=np.float64,
                            )

                    well_psd[wr, wc] = filtered_mean
                    raw_60 = mean_power_near(
                        freqs,
                        raw_mean,
                        args.notch_hz,
                        half_width_hz=1.0,
                    )
                    filtered_60 = mean_power_near(
                        freqs,
                        filtered_mean,
                        args.notch_hz,
                        half_width_hz=1.0,
                    )
                    epsilon = np.finfo(float).tiny
                    notch_reduction[wr, wc] = 10.0 * np.log10(
                        max(float(raw_60), epsilon)
                        / max(float(filtered_60), epsilon)
                    )

            assert well_psd is not None
            assert frequency_axis is not None
            psd_windows.append(well_psd)
            notch_windows.append(notch_reduction)

            band_arrays: dict[str, np.ndarray] = {}
            for band_name, (low_hz, high_hz) in BANDS_HZ.items():
                values = integrate_band(
                    frequency_axis,
                    well_psd,
                    low_hz,
                    high_hz,
                )
                band_arrays[band_name] = values
                band_window_lists[band_name].append(values)

            for wr in range(n_rows_well):
                for wc in range(n_cols_well):
                    well = well_name(wr, wc)
                    map_row = plate_map.get(well, {})
                    row: dict[str, Any] = {
                        "window_index": window_index,
                        "window_start_sec": start_sec,
                        "window_end_sec": end_window_sec,
                        "window_center_sec": center_sec,
                        "well": well,
                        "group": map_row.get("group", ""),
                        "include": map_row.get("include", ""),
                        "notch_reduction_db_60hz": float(notch_reduction[wr, wc]),
                    }
                    for band_name, values in band_arrays.items():
                        row[f"{band_name}_power_uv2"] = float(values[wr, wc])
                    per_window_rows.append(row)

            elapsed = time.perf_counter() - window_timer
            print(f"窗口完成，用时 {elapsed:.1f} 秒")
    finally:
        del raw_memmap

    assert frequency_axis is not None
    psd_window_array = np.stack(psd_windows, axis=0)
    notch_window_array = np.stack(notch_windows, axis=0)
    band_window_arrays = {
        name: np.stack(values, axis=0)
        for name, values in band_window_lists.items()
    }
    window_centers_array = np.asarray(window_centers_sec, dtype=float)
    mean_well_psd = psd_window_array.mean(axis=0)

    per_window_csv = args.output_dir / "per_window_per_well_bandpower.csv"
    write_csv(per_window_csv, per_window_rows)

    per_well_rows: list[dict[str, Any]] = []
    for wr in range(n_rows_well):
        for wc in range(n_cols_well):
            well = well_name(wr, wc)
            map_row = plate_map.get(well, {})
            row = {
                "well": well,
                "group": map_row.get("group", ""),
                "include": map_row.get("include", ""),
                "n_windows": int(window_centers_array.size),
                "mean_notch_reduction_db_60hz": float(
                    notch_window_array[:, wr, wc].mean()
                ),
                "median_notch_reduction_db_60hz": float(
                    np.median(notch_window_array[:, wr, wc])
                ),
            }
            for band_name, values in band_window_arrays.items():
                well_values = values[:, wr, wc]
                row[f"mean_{band_name}_power_uv2"] = float(well_values.mean())
                row[f"median_{band_name}_power_uv2"] = float(
                    np.median(well_values)
                )
                row[f"std_{band_name}_power_uv2"] = float(
                    well_values.std(ddof=1) if well_values.size > 1 else 0.0
                )
            per_well_rows.append(row)

    per_well_csv = args.output_dir / "per_well_summary.csv"
    write_csv(per_well_csv, per_well_rows)

    npz_path = args.output_dir / "well_psd_windows.npz"
    np.savez_compressed(
        npz_path,
        freqs_hz=frequency_axis,
        window_centers_sec=window_centers_array,
        well_psd_windows_uv2_per_hz=psd_window_array,
        mean_well_psd_uv2_per_hz=mean_well_psd,
        notch_reduction_db_60hz=notch_window_array,
        **{
            f"{name}_power_uv2": values
            for name, values in band_window_arrays.items()
        },
    )

    group_psd_path = args.output_dir / "group_mean_psd.png"
    gamma_heatmap_path = args.output_dir / "gamma_mean_heatmap.png"
    notch_heatmap_path = args.output_dir / "notch_reduction_mean_heatmap.png"
    gamma_timecourse_path = args.output_dir / "gamma_group_timecourse.png"

    plot_group_mean_psd(
        group_psd_path,
        frequency_axis,
        mean_well_psd,
        plate_map,
    )
    gamma_mean = band_window_arrays["gamma"].mean(axis=0)
    plot_plate_heatmap(
        gamma_heatmap_path,
        gamma_mean,
        "Mean gamma band power across analysis windows (30-100 Hz)",
        "log10 total gamma power (µV²; not 1/f corrected)",
        plate_map,
        log_display=True,
    )
    plot_plate_heatmap(
        notch_heatmap_path,
        notch_window_array.mean(axis=0),
        "Mean 60 Hz power reduction across analysis windows",
        "Reduction (dB)",
        plate_map,
        log_display=False,
    )
    plot_group_band_timecourse(
        gamma_timecourse_path,
        window_centers_array,
        band_window_arrays["gamma"],
        "gamma",
        plate_map,
    )

    included_reductions = []
    for wr in range(n_rows_well):
        for wc in range(n_cols_well):
            well = well_name(wr, wc)
            if plate_map.get(well, {}).get("include", "") == "1":
                included_reductions.extend(
                    notch_window_array[:, wr, wc].astype(float).tolist()
                )

    processed_end_sec = float(
        (window_start_samples[-1] + window_samples) / fs_hz
    )
    elapsed_total = time.perf_counter() - total_start
    summary = {
        "raw_path": str(raw_path),
        "requested_start_sec": args.start_sec,
        "requested_end_sec": args.end_sec,
        "available_duration_sec": available_duration_sec,
        "processed_start_sec": float(window_start_samples[0] / fs_hz),
        "processed_end_sec": processed_end_sec,
        "analysis_window_sec": args.window_sec,
        "analysis_step_sec": args.step_sec,
        "n_analysis_windows": int(window_start_samples.size),
        "welch_segment_sec": nperseg / fs_hz,
        "welch_overlap_fraction": args.welch_overlap_fraction,
        "sampling_rate_hz": fs_hz,
        "notch_hz": args.notch_hz,
        "notch_q": args.notch_q,
        "frequency_range_hz": [float(frequency_axis[0]), float(frequency_axis[-1])],
        "electrode_filtering_applied": False,
        "all_16_electrodes_used_per_well": True,
        "empty_wells_excluded_from_group_plots": True,
        "aperiodic_1_over_f_model_applied": False,
        "formal_group_statistics_applied": False,
        "bandpower_warning": (
            "Band powers are total powers and still include the aperiodic 1/f "
            "component. They are not final aperiodic-corrected results."
        ),
        "median_notch_reduction_db_included_values": (
            float(np.median(included_reductions))
            if included_reductions
            else None
        ),
        "elapsed_seconds": elapsed_total,
        "output_files": {
            "per_window_csv": str(per_window_csv),
            "per_well_csv": str(per_well_csv),
            "well_psd_npz": str(npz_path),
            "group_mean_psd": str(group_psd_path),
            "gamma_heatmap": str(gamma_heatmap_path),
            "notch_heatmap": str(notch_heatmap_path),
            "gamma_timecourse": str(gamma_timecourse_path),
        },
    }
    summary_path = args.output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\n=== Streaming Welch analysis complete ===")
    print(f"处理窗口数量: {window_start_samples.size}")
    print(f"实际处理范围: {summary['processed_start_sec']:.3f}-{processed_end_sec:.3f} 秒")
    print(f"总用时: {elapsed_total:.1f} 秒")
    print("电极筛选: 未执行")
    print("正式统计: 未执行")
    print("1/f 拟合: 未执行")
    print(f"结果目录: {args.output_dir.resolve()}")
    print(f"摘要: {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "直接从 Axion RAW 流式计算较长时间范围的 Welch PSD，"
            "使用每孔全部 16 个电极，不执行 electrode QC。"
        )
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--channel-array", type=Path, default=DEFAULT_CHANNEL_ARRAY)
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-sec", type=float, default=10.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--step-sec", type=float, default=10.0)
    parser.add_argument("--welch-segment-sec", type=float, default=1.0)
    parser.add_argument("--welch-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--notch-hz", type=float, default=60.0)
    parser.add_argument("--notch-q", type=float, default=120.0)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=200.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
