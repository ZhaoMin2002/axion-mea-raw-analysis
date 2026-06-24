"""Validate 60 Hz notch filtering and Welch PSD on a short MEA segment.

This diagnostic step uses every electrode. It does not perform electrode QC,
does not remove channels, and does not provide final biological statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


DEFAULT_INPUT = Path("outputs/plate8_4039/short_segment/segment_signal_uv.npz")
DEFAULT_PLATE_MAP = Path("resources/plate_maps/plate8_4039_platemap.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/plate8_4039/short_segment_psd")

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


def electrode_name(erow: int, ecol: int) -> str:
    return f"{ecol + 1}{erow + 1}"


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
            if well:
                mapping[well] = {
                    key: str(value or "").strip() for key, value in row.items()
                }
    return mapping


def load_segment(path: Path) -> tuple[np.ndarray, float, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到短片段 NPZ: {path}")

    with np.load(path) as data:
        required = {"signal_uv", "fs_hz", "start_sec", "duration_sec"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"短片段 NPZ 缺少字段: {sorted(missing)}")
        signal_uv = np.asarray(data["signal_uv"], dtype=np.float32)
        fs_hz = float(np.asarray(data["fs_hz"]).item())
        start_sec = float(np.asarray(data["start_sec"]).item())
        duration_sec = float(np.asarray(data["duration_sec"]).item())

    if signal_uv.ndim != 5:
        raise ValueError(
            "signal_uv 应为 5 维数组: "
            "(well_row, well_col, electrode_row, electrode_col, time)"
        )
    return signal_uv, fs_hz, start_sec, duration_sec


def make_welch_settings(fs_hz: float, n_samples: int) -> tuple[int, int]:
    # Follow the existing workflow concept: one-second Welch segments.
    nperseg = min(int(round(fs_hz)), n_samples)
    if nperseg < 8:
        raise ValueError("信号片段太短，无法计算 Welch PSD。")
    noverlap = nperseg // 2
    return nperseg, noverlap


def compute_psd(
    traces: np.ndarray,
    fs_hz: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    freqs, powers = signal.welch(
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
    return freqs, powers


def bandpower(freqs: np.ndarray, powers: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 2:
        return np.full(powers.shape[:-1], np.nan, dtype=float)
    return np.trapezoid(powers[..., mask], freqs[mask], axis=-1)


def power_near_frequency(
    freqs: np.ndarray,
    powers: np.ndarray,
    center_hz: float,
    half_width_hz: float = 1.0,
) -> np.ndarray:
    mask = np.abs(freqs - center_hz) <= half_width_hz
    if not np.any(mask):
        raise ValueError(f"PSD 中没有 {center_hz} Hz 附近的频点。")
    return powers[..., mask].mean(axis=-1)


def calculate_psds(
    signal_uv: np.ndarray,
    fs_hz: float,
    notch_hz: float,
    notch_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols, n_erows, n_ecols, n_samples = signal_uv.shape
    nperseg, noverlap = make_welch_settings(fs_hz, n_samples)

    b_notch, a_notch = signal.iirnotch(notch_hz, notch_q, fs=fs_hz)

    freqs_full: np.ndarray | None = None
    raw_psd_full: np.ndarray | None = None
    filtered_psd_full: np.ndarray | None = None

    total_wells = n_rows * n_cols
    current = 0
    for wr in range(n_rows):
        for wc in range(n_cols):
            current += 1
            print(f"计算 PSD: well {well_name(wr, wc)} ({current}/{total_wells})")

            traces = signal_uv[wr, wc].reshape(n_erows * n_ecols, n_samples)
            filtered = signal.filtfilt(b_notch, a_notch, traces, axis=-1)

            freqs, raw_psd = compute_psd(traces, fs_hz, nperseg, noverlap)
            freqs_filtered, filtered_psd = compute_psd(
                filtered, fs_hz, nperseg, noverlap
            )
            if not np.array_equal(freqs, freqs_filtered):
                raise RuntimeError("滤波前后的 PSD 频率轴不一致。")

            if freqs_full is None:
                freqs_full = freqs
                output_shape = (
                    n_rows,
                    n_cols,
                    n_erows,
                    n_ecols,
                    freqs.size,
                )
                raw_psd_full = np.empty(output_shape, dtype=np.float64)
                filtered_psd_full = np.empty(output_shape, dtype=np.float64)

            assert raw_psd_full is not None
            assert filtered_psd_full is not None
            raw_psd_full[wr, wc] = raw_psd.reshape(n_erows, n_ecols, -1)
            filtered_psd_full[wr, wc] = filtered_psd.reshape(
                n_erows, n_ecols, -1
            )

    assert freqs_full is not None
    assert raw_psd_full is not None
    assert filtered_psd_full is not None
    return freqs_full, raw_psd_full, filtered_psd_full, np.asarray([nperseg, noverlap])


def write_well_features(
    path: Path,
    freqs: np.ndarray,
    well_psd: np.ndarray,
    notch_reduction_db: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wr in range(well_psd.shape[0]):
        for wc in range(well_psd.shape[1]):
            well = well_name(wr, wc)
            map_row = plate_map.get(well, {})
            row: dict[str, Any] = {
                "well": well,
                "group": map_row.get("group", ""),
                "include": map_row.get("include", ""),
                "notch_reduction_db_60hz": float(notch_reduction_db[wr, wc]),
            }
            for name, (low, high) in BANDS_HZ.items():
                value = bandpower(freqs, well_psd[wr, wc], low, high)
                row[f"{name}_power_uv2"] = float(value)
            rows.append(row)

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_notch_example(
    path: Path,
    freqs: np.ndarray,
    raw_psd: np.ndarray,
    filtered_psd: np.ndarray,
    notch_hz: float,
) -> None:
    mask = (freqs >= 1.0) & (freqs <= 200.0)
    tiny = np.finfo(float).tiny

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freqs[mask], np.maximum(raw_psd[0, 0, 0, 0, mask], tiny), label="Before notch")
    ax.loglog(
        freqs[mask],
        np.maximum(filtered_psd[0, 0, 0, 0, mask], tiny),
        label="After notch",
    )
    ax.axvline(notch_hz, linestyle="--", linewidth=1, label=f"{notch_hz:g} Hz")
    ax.set_title("A1 electrode 11: Welch PSD before and after notch")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_psd(
    path: Path,
    freqs: np.ndarray,
    well_psd: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> None:
    groups: dict[str, list[np.ndarray]] = {}
    for wr in range(well_psd.shape[0]):
        for wc in range(well_psd.shape[1]):
            well = well_name(wr, wc)
            row = plate_map.get(well, {})
            group = row.get("group", "")
            include = row.get("include", "")
            if include != "1" or not group or group.lower() == "empty":
                continue
            groups.setdefault(group, []).append(well_psd[wr, wc])

    mask = (freqs >= 1.0) & (freqs <= 200.0)
    tiny = np.finfo(float).tiny
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, spectra in groups.items():
        group_array = np.stack(spectra, axis=0)
        mean = group_array.mean(axis=0)
        ax.loglog(
            freqs[mask],
            np.maximum(mean[mask], tiny),
            label=f"{group} (n={group_array.shape[0]} wells)",
        )
    ax.set_title("Diagnostic group mean PSD after 60 Hz notch")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_notch_reduction_heatmap(path: Path, reduction_db: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(reduction_db, aspect="auto")
    ax.set_title("60 Hz power reduction after notch (dB)")
    ax.set_xlabel("Well column")
    ax.set_ylabel("Well row")
    ax.set_xticks(
        np.arange(reduction_db.shape[1]),
        labels=np.arange(1, reduction_db.shape[1] + 1),
    )
    ax.set_yticks(
        np.arange(reduction_db.shape[0]),
        labels=[chr(ord("A") + index) for index in range(reduction_db.shape[0])],
    )
    for wr in range(reduction_db.shape[0]):
        for wc in range(reduction_db.shape[1]):
            ax.text(
                wc,
                wr,
                f"{reduction_db[wr, wc]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Reduction (dB; positive means lower after notch)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_band_heatmap(
    path: Path,
    band_name: str,
    values: np.ndarray,
    plate_map: dict[str, dict[str, str]],
) -> None:
    display = np.log10(np.maximum(values, np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(display, aspect="auto")
    low, high = BANDS_HZ[band_name]
    ax.set_title(f"{band_name} band power ({low:g}-{high:g} Hz), diagnostic")
    ax.set_xlabel("Well column")
    ax.set_ylabel("Well row")
    ax.set_xticks(
        np.arange(values.shape[1]), labels=np.arange(1, values.shape[1] + 1)
    )
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
                f"{well}\n{group}\n{values[wr, wc]:.2g}",
                ha="center",
                va="center",
                fontsize=6,
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10 band power (µV²)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    plate_map = load_plate_map(args.plate_map)
    signal_uv, fs_hz, start_sec, duration_sec = load_segment(args.input)

    print("=== 短片段 notch / PSD 诊断 ===")
    print(f"输入 shape: {signal_uv.shape}")
    print(f"采样率: {fs_hz:.1f} Hz")
    print(f"片段: {start_sec:.3f}-{start_sec + duration_sec:.3f} 秒")
    print("电极筛选: 不执行，保留全部 16 个电极")

    freqs_full, raw_psd_full, filtered_psd_full, settings = calculate_psds(
        signal_uv=signal_uv,
        fs_hz=fs_hz,
        notch_hz=args.notch_hz,
        notch_q=args.notch_q,
    )
    del signal_uv

    freq_mask = (freqs_full >= args.fmin) & (freqs_full <= args.fmax)
    if np.count_nonzero(freq_mask) < 2:
        raise ValueError("指定频率范围内没有足够的 PSD 频点。")

    freqs = freqs_full[freq_mask]
    raw_psd = raw_psd_full[..., freq_mask]
    filtered_psd = filtered_psd_full[..., freq_mask]
    del raw_psd_full, filtered_psd_full

    raw_well_psd = raw_psd.mean(axis=(-3, -2))
    filtered_well_psd = filtered_psd.mean(axis=(-3, -2))

    raw_60 = power_near_frequency(freqs, raw_well_psd, args.notch_hz)
    filtered_60 = power_near_frequency(freqs, filtered_well_psd, args.notch_hz)
    epsilon = np.finfo(float).tiny
    notch_reduction_db = 10.0 * np.log10(
        np.maximum(raw_60, epsilon) / np.maximum(filtered_60, epsilon)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    psd_path = args.output_dir / "short_segment_psd.npz"
    np.savez_compressed(
        psd_path,
        freqs_hz=freqs,
        raw_electrode_psd_uv2_per_hz=raw_psd,
        filtered_electrode_psd_uv2_per_hz=filtered_psd,
        filtered_well_psd_uv2_per_hz=filtered_well_psd,
        notch_reduction_db_60hz=notch_reduction_db,
    )

    feature_path = args.output_dir / "per_well_bandpower_diagnostic.csv"
    feature_rows = write_well_features(
        feature_path,
        freqs,
        filtered_well_psd,
        notch_reduction_db,
        plate_map,
    )

    notch_plot = args.output_dir / "notch_before_after_A1_E11.png"
    group_psd_plot = args.output_dir / "group_mean_psd_diagnostic.png"
    notch_heatmap = args.output_dir / "notch_reduction_heatmap.png"
    gamma_heatmap = args.output_dir / "gamma_bandpower_heatmap_diagnostic.png"

    plot_notch_example(
        notch_plot, freqs, raw_psd, filtered_psd, args.notch_hz
    )
    plot_group_psd(group_psd_plot, freqs, filtered_well_psd, plate_map)
    plot_notch_reduction_heatmap(notch_heatmap, notch_reduction_db)
    gamma_values = bandpower(
        freqs,
        filtered_well_psd,
        *BANDS_HZ["gamma"],
    )
    plot_band_heatmap(gamma_heatmap, "gamma", gamma_values, plate_map)

    included_reductions = [
        float(row["notch_reduction_db_60hz"])
        for row in feature_rows
        if row.get("include") == "1"
    ]
    summary: dict[str, Any] = {
        "input": str(args.input),
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "sampling_rate_hz": fs_hz,
        "welch_nperseg_samples": int(settings[0]),
        "welch_noverlap_samples": int(settings[1]),
        "welch_segment_seconds": float(settings[0] / fs_hz),
        "notch_hz": args.notch_hz,
        "notch_q": args.notch_q,
        "electrode_filtering_applied": False,
        "all_16_electrodes_used_per_well": True,
        "aperiodic_1_over_f_model_applied": False,
        "diagnostic_only": True,
        "interpretation_warning": (
            "The two-second segment is sufficient for checking data flow and the "
            "60 Hz notch, but it is not sufficient for final low-frequency or "
            "group-level inference."
        ),
        "frequency_range_saved_hz": [float(freqs[0]), float(freqs[-1])],
        "median_notch_reduction_db_included_wells": (
            float(np.median(included_reductions)) if included_reductions else None
        ),
        "output_files": {
            "psd_npz": str(psd_path),
            "per_well_features_csv": str(feature_path),
            "notch_example": str(notch_plot),
            "group_mean_psd": str(group_psd_plot),
            "notch_reduction_heatmap": str(notch_heatmap),
            "gamma_heatmap": str(gamma_heatmap),
        },
    }
    summary_path = args.output_dir / "psd_diagnostic_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\n=== 诊断完成 ===")
    print(
        "纳入孔的 60 Hz 功率中位下降: "
        f"{summary['median_notch_reduction_db_included_wells']:.2f} dB"
    )
    print("正式统计: 未执行")
    print("1/f 拟合: 未执行")
    print(f"结果目录: {args.output_dir.resolve()}")
    print(f"摘要: {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用短片段检查 60 Hz notch 和 Welch PSD，不筛选电极。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notch-hz", type=float, default=60.0)
    parser.add_argument("--notch-q", type=float, default=120.0)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=200.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
