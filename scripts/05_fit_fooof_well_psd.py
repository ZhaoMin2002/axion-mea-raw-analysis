"""Fit FOOOF models to per-well PSDs from the full Plate 8 recording.

This step parameterizes each well's mean PSD into periodic and aperiodic
components. It keeps all 16 electrodes per well and performs no electrode QC.
The 60 Hz notch region is interpolated only for model fitting so that the notch
trough does not bias the aperiodic fit. Formal group statistics are not run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import fooof
    from fooof import FOOOF
except ImportError as exc:
    raise SystemExit(
        "缺少 FOOOF。请先运行: pip install fooof==1.1.1"
    ) from exc


DEFAULT_INPUT = Path(
    "outputs/plate8_4039/welch_full_10_290/well_psd_windows.npz"
)
DEFAULT_PLATE_MAP = Path(
    "resources/plate_maps/plate8_4039_platemap.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/plate8_4039/fooof_full_10_290"
)

BANDS_HZ: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def well_name(row_index: int, col_index: int) -> str:
    return f"{chr(ord('A') + row_index)}{col_index + 1}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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
                    key: str(value or "").strip()
                    for key, value in row.items()
                }
    return mapping


def load_psd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 PSD NPZ: {path}")

    with np.load(path) as data:
        required = {
            "freqs_hz",
            "window_centers_sec",
            "mean_well_psd_uv2_per_hz",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"PSD NPZ 缺少字段: {sorted(missing)}")

        freqs = np.asarray(data["freqs_hz"], dtype=float)
        window_centers = np.asarray(data["window_centers_sec"], dtype=float)
        mean_well_psd = np.asarray(
            data["mean_well_psd_uv2_per_hz"], dtype=float
        )

    if freqs.ndim != 1:
        raise ValueError("freqs_hz 必须是一维数组。")
    if mean_well_psd.ndim != 3:
        raise ValueError(
            "mean_well_psd_uv2_per_hz 应为 "
            "(well_row, well_col, frequency)。"
        )
    if mean_well_psd.shape[-1] != freqs.size:
        raise ValueError("PSD 最后一维与频率轴长度不一致。")
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("频率轴必须严格递增。")
    return freqs, window_centers, mean_well_psd


def interpolate_log_notch(
    freqs: np.ndarray,
    power: np.ndarray,
    notch_low: float,
    notch_high: float,
) -> np.ndarray:
    """Interpolate the notch interval in log10 power for FOOOF fitting only."""
    if notch_high <= notch_low:
        raise ValueError("notch_high 必须大于 notch_low。")

    tiny = np.finfo(float).tiny
    log_power = np.log10(np.maximum(power, tiny))
    inside = (freqs >= notch_low) & (freqs <= notch_high)
    if not np.any(inside):
        return np.power(10.0, log_power)

    left_candidates = np.flatnonzero(freqs < notch_low)
    right_candidates = np.flatnonzero(freqs > notch_high)
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise ValueError("notch 区间两侧没有足够频点用于插值。")

    left = int(left_candidates[-1])
    right = int(right_candidates[0])
    log_power[inside] = np.interp(
        freqs[inside],
        [freqs[left], freqs[right]],
        [log_power[left], log_power[right]],
    )
    return np.power(10.0, log_power)


def aperiodic_log10(
    freqs: np.ndarray,
    params: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "fixed":
        if params.size != 2:
            raise ValueError(f"fixed 模式预期 2 个参数，实际 {params.size}")
        offset, exponent = params
        return offset - exponent * np.log10(freqs)

    if mode == "knee":
        if params.size != 3:
            raise ValueError(f"knee 模式预期 3 个参数，实际 {params.size}")
        offset, knee, exponent = params
        denominator = knee + np.power(freqs, exponent)
        if np.any(denominator <= 0):
            raise ValueError("knee 模式产生非正数 denominator。")
        return offset - np.log10(denominator)

    raise ValueError(f"不支持的 aperiodic mode: {mode}")


def gaussian_log10(
    freqs: np.ndarray,
    gaussian_params: np.ndarray,
) -> np.ndarray:
    periodic = np.zeros_like(freqs, dtype=float)
    for center, height, std in np.atleast_2d(gaussian_params):
        if not np.all(np.isfinite([center, height, std])) or std <= 0:
            continue
        periodic += height * np.exp(
            -np.square(freqs - center) / (2.0 * std * std)
        )
    return periodic


def is_line_frequency(
    frequency_hz: float,
    notch_low: float,
    notch_high: float,
) -> bool:
    return notch_low <= frequency_hz <= notch_high


def select_band_peak(
    peaks: np.ndarray,
    low_hz: float,
    high_hz: float,
    notch_low: float,
    notch_high: float,
) -> tuple[float, float, float, int]:
    candidates = []
    for center, power, bandwidth in np.atleast_2d(peaks):
        if (
            low_hz <= center <= high_hz
            and not is_line_frequency(center, notch_low, notch_high)
        ):
            candidates.append((float(center), float(power), float(bandwidth)))

    if not candidates:
        return math.nan, math.nan, math.nan, 0

    best = max(candidates, key=lambda item: item[1])
    return best[0], best[1], best[2], len(candidates)


def mean_relative_db(
    freqs: np.ndarray,
    corrected_log10: np.ndarray,
    low_hz: float,
    high_hz: float,
    notch_low: float,
    notch_high: float,
) -> float:
    mask = (
        (freqs >= low_hz)
        & (freqs <= high_hz)
        & ~((freqs >= notch_low) & (freqs <= notch_high))
        & np.isfinite(corrected_log10)
    )
    if not np.any(mask):
        return math.nan
    return float(10.0 * np.mean(corrected_log10[mask]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_plate_heatmap(
    path: Path,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    plate_map: dict[str, dict[str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Well column")
    ax.set_ylabel("Well row")
    ax.set_xticks(
        np.arange(values.shape[1]),
        labels=np.arange(1, values.shape[1] + 1),
    )
    ax.set_yticks(
        np.arange(values.shape[0]),
        labels=[chr(ord("A") + index) for index in range(values.shape[0])],
    )

    for wr in range(values.shape[0]):
        for wc in range(values.shape[1]):
            well = well_name(wr, wc)
            group = plate_map.get(well, {}).get("group", "")
            value = values[wr, wc]
            text_value = "NA" if not np.isfinite(value) else f"{value:.3g}"
            ax.text(
                wc,
                wr,
                f"{well}\n{group}\n{text_value}",
                ha="center",
                va="center",
                fontsize=6,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_box(
    path: Path,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    groups = sorted(
        {
            str(row["group"])
            for row in rows
            if row.get("include") == "1"
            and row.get("fit_success") == 1
            and str(row.get("group", "")).lower() != "empty"
        }
    )
    data = []
    used_groups = []
    for group in groups:
        values = [
            float(row[field])
            for row in rows
            if row.get("group") == group
            and row.get("include") == "1"
            and row.get("fit_success") == 1
            and np.isfinite(float(row[field]))
        ]
        if values:
            used_groups.append(group)
            data.append(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=used_groups, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Group")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reconstruct_models(
    freqs: np.ndarray,
    ap_params: np.ndarray,
    gaussian_params: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    ap_log = aperiodic_log10(freqs, ap_params, mode)
    full_log = ap_log + gaussian_log10(freqs, gaussian_params)
    return ap_log, full_log


def plot_representative_fit(
    path: Path,
    freqs: np.ndarray,
    original_power: np.ndarray,
    fit_power: np.ndarray,
    ap_log: np.ndarray,
    full_log: np.ndarray,
    well: str,
    group: str,
    notch_low: float,
    notch_high: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(freqs, original_power, label="PSD after notch")
    ax.loglog(
        freqs,
        fit_power,
        linestyle=":",
        label="PSD used for FOOOF fit",
    )
    ax.loglog(freqs, np.power(10.0, ap_log), label="Aperiodic fit")
    ax.loglog(freqs, np.power(10.0, full_log), label="Full FOOOF model")
    ax.axvspan(
        notch_low,
        notch_high,
        alpha=0.15,
        label="Interpolated line-noise interval",
    )
    ax.set_title(f"FOOOF fit: {well} ({group})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_group_descriptive(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = sorted(
        {
            str(row["group"])
            for row in rows
            if row.get("include") == "1"
            and str(row.get("group", "")).lower() != "empty"
        }
    )
    output: list[dict[str, Any]] = []
    for group in groups:
        selected = [
            row
            for row in rows
            if row.get("group") == group
            and row.get("include") == "1"
        ]
        successful = [row for row in selected if row.get("fit_success") == 1]

        def summary(field: str) -> tuple[float, float, float]:
            values = np.asarray(
                [
                    float(row[field])
                    for row in successful
                    if np.isfinite(float(row[field]))
                ],
                dtype=float,
            )
            if values.size == 0:
                return math.nan, math.nan, math.nan
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            return float(values.mean()), std, float(np.median(values))

        exp_mean, exp_std, exp_median = summary("aperiodic_exponent")
        off_mean, off_std, off_median = summary("aperiodic_offset")
        gamma_mean, gamma_std, gamma_median = summary(
            "gamma_relative_db_mean"
        )
        gamma_peak_rows = [
            row
            for row in successful
            if int(row["gamma_peak_present"]) == 1
        ]
        gamma_peak_powers = np.asarray(
            [
                float(row["gamma_peak_power_log10"])
                for row in gamma_peak_rows
            ],
            dtype=float,
        )

        output.append(
            {
                "group": group,
                "n_wells": len(selected),
                "n_successful_fits": len(successful),
                "fit_success_fraction": (
                    len(successful) / len(selected) if selected else math.nan
                ),
                "aperiodic_exponent_mean": exp_mean,
                "aperiodic_exponent_std": exp_std,
                "aperiodic_exponent_median": exp_median,
                "aperiodic_offset_mean": off_mean,
                "aperiodic_offset_std": off_std,
                "aperiodic_offset_median": off_median,
                "gamma_relative_db_mean": gamma_mean,
                "gamma_relative_db_std": gamma_std,
                "gamma_relative_db_median": gamma_median,
                "gamma_peak_count": len(gamma_peak_rows),
                "gamma_peak_probability": (
                    len(gamma_peak_rows) / len(successful)
                    if successful
                    else math.nan
                ),
                "gamma_peak_power_log10_mean_detected": (
                    float(gamma_peak_powers.mean())
                    if gamma_peak_powers.size
                    else math.nan
                ),
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    plate_map = load_plate_map(args.plate_map)
    freqs_all, window_centers, mean_well_psd = load_psd(args.input)

    fit_mask = (freqs_all >= args.fmin) & (freqs_all <= args.fmax)
    freqs = freqs_all[fit_mask]
    if freqs.size < 10:
        raise ValueError("FOOOF 拟合频率范围内的频点过少。")
    if np.any(freqs <= 0):
        raise ValueError("FOOOF 拟合频率必须大于 0 Hz。")

    frequency_resolution = float(np.median(np.diff(freqs)))
    if args.peak_width_low < 2.0 * frequency_resolution:
        print(
            "[警告] peak width 下限小于两倍频率分辨率，"
            "可能拟合噪声峰。"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols, _ = mean_well_psd.shape
    corrected_log10 = np.full(
        (n_rows, n_cols, freqs.size), np.nan, dtype=np.float64
    )
    aperiodic_fits_log10 = np.full_like(corrected_log10, np.nan)
    full_models_log10 = np.full_like(corrected_log10, np.nan)
    fit_powers = np.full_like(corrected_log10, np.nan)

    rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    model_records: dict[str, dict[str, Any]] = {}

    print("=== FOOOF per-well fitting ===")
    print(f"输入: {args.input}")
    print(f"频率范围: {args.fmin:g}-{args.fmax:g} Hz")
    print(
        "peak width limits: "
        f"{args.peak_width_low:g}-{args.peak_width_high:g} Hz"
    )
    print(f"aperiodic mode: {args.aperiodic_mode}")
    print(
        "line-noise interpolation for fitting only: "
        f"{args.notch_low:g}-{args.notch_high:g} Hz"
    )
    print("电极筛选: 不执行，输入 PSD 已使用每孔全部 16 个电极")
    print("正式组间统计: 本步骤不执行")

    for wr in range(n_rows):
        for wc in range(n_cols):
            well = well_name(wr, wc)
            map_row = plate_map.get(well, {})
            group = map_row.get("group", "")
            include = map_row.get("include", "")
            original_power = np.asarray(
                mean_well_psd[wr, wc, fit_mask], dtype=float
            )
            fit_power = interpolate_log_notch(
                freqs,
                original_power,
                args.notch_low,
                args.notch_high,
            )
            fit_powers[wr, wc] = fit_power

            base_row: dict[str, Any] = {
                "well": well,
                "group": group,
                "include": include,
                "fit_success": 0,
                "fit_error_message": "",
                "fooof_r_squared": math.nan,
                "fooof_error": math.nan,
                "aperiodic_offset": math.nan,
                "aperiodic_knee": math.nan,
                "aperiodic_exponent": math.nan,
                "n_peaks": 0,
            }
            for band_name in BANDS_HZ:
                base_row[f"{band_name}_relative_db_mean"] = math.nan
                base_row[f"{band_name}_peak_present"] = 0
                base_row[f"{band_name}_peak_cf_hz"] = math.nan
                base_row[f"{band_name}_peak_power_log10"] = math.nan
                base_row[f"{band_name}_peak_bw_hz"] = math.nan
                base_row[f"{band_name}_peak_count"] = 0

            try:
                fm = FOOOF(
                    peak_width_limits=[
                        args.peak_width_low,
                        args.peak_width_high,
                    ],
                    max_n_peaks=args.max_n_peaks,
                    min_peak_height=args.min_peak_height,
                    peak_threshold=args.peak_threshold,
                    aperiodic_mode=args.aperiodic_mode,
                    verbose=False,
                )
                fm.fit(freqs, fit_power, [args.fmin, args.fmax])

                ap_params = np.asarray(fm.aperiodic_params_, dtype=float)
                peaks = np.asarray(fm.peak_params_, dtype=float).reshape(-1, 3)
                gaussians = np.asarray(
                    fm.gaussian_params_, dtype=float
                ).reshape(-1, 3)
                ap_log, full_log = reconstruct_models(
                    freqs,
                    ap_params,
                    gaussians,
                    args.aperiodic_mode,
                )
                original_log = np.log10(
                    np.maximum(original_power, np.finfo(float).tiny)
                )
                corrected = original_log - ap_log
                notch_mask = (
                    (freqs >= args.notch_low)
                    & (freqs <= args.notch_high)
                )
                corrected[notch_mask] = np.nan

                corrected_log10[wr, wc] = corrected
                aperiodic_fits_log10[wr, wc] = ap_log
                full_models_log10[wr, wc] = full_log

                base_row["fit_success"] = 1
                base_row["fooof_r_squared"] = float(fm.r_squared_)
                base_row["fooof_error"] = float(fm.error_)
                base_row["aperiodic_offset"] = float(ap_params[0])
                if args.aperiodic_mode == "fixed":
                    base_row["aperiodic_exponent"] = float(ap_params[1])
                else:
                    base_row["aperiodic_knee"] = float(ap_params[1])
                    base_row["aperiodic_exponent"] = float(ap_params[2])
                base_row["n_peaks"] = int(peaks.shape[0])

                for band_name, (low_hz, high_hz) in BANDS_HZ.items():
                    base_row[f"{band_name}_relative_db_mean"] = (
                        mean_relative_db(
                            freqs,
                            corrected,
                            low_hz,
                            high_hz,
                            args.notch_low,
                            args.notch_high,
                        )
                    )
                    center, power, bandwidth, count = select_band_peak(
                        peaks,
                        low_hz,
                        high_hz,
                        args.notch_low,
                        args.notch_high,
                    )
                    base_row[f"{band_name}_peak_present"] = int(count > 0)
                    base_row[f"{band_name}_peak_cf_hz"] = center
                    base_row[f"{band_name}_peak_power_log10"] = power
                    base_row[f"{band_name}_peak_bw_hz"] = bandwidth
                    base_row[f"{band_name}_peak_count"] = count

                for peak_index, (center, power, bandwidth) in enumerate(
                    peaks
                ):
                    peak_rows.append(
                        {
                            "well": well,
                            "group": group,
                            "include": include,
                            "peak_index": peak_index,
                            "center_frequency_hz": float(center),
                            "peak_power_log10": float(power),
                            "bandwidth_hz": float(bandwidth),
                            "line_noise_interval": int(
                                is_line_frequency(
                                    float(center),
                                    args.notch_low,
                                    args.notch_high,
                                )
                            ),
                        }
                    )

                model_records[well] = {
                    "group": group,
                    "original_power": original_power,
                    "fit_power": fit_power,
                    "ap_log": ap_log,
                    "full_log": full_log,
                    "r2": float(fm.r_squared_),
                }
                print(
                    f"{well:>3} {group:<6} "
                    f"R²={fm.r_squared_:.4f}, peaks={peaks.shape[0]}"
                )
            except Exception as exc:
                base_row["fit_error_message"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(
                    f"[失败] {well} {group}: "
                    f"{base_row['fit_error_message']}"
                )

            rows.append(base_row)

    per_well_path = args.output_dir / "fooof_per_well.csv"
    write_csv(per_well_path, rows)

    peaks_path = args.output_dir / "fooof_all_peaks.csv"
    if peak_rows:
        write_csv(peaks_path, peak_rows)
    else:
        with peaks_path.open("w", encoding="utf-8", newline="") as file_obj:
            file_obj.write(
                "well,group,include,peak_index,center_frequency_hz,"
                "peak_power_log10,bandwidth_hz,line_noise_interval\n"
            )

    descriptive_rows = build_group_descriptive(rows)
    descriptive_path = args.output_dir / "group_descriptive.csv"
    write_csv(descriptive_path, descriptive_rows)

    npz_path = args.output_dir / "fooof_models.npz"
    np.savez_compressed(
        npz_path,
        freqs_hz=freqs,
        window_centers_sec=window_centers,
        corrected_log10_power=corrected_log10,
        aperiodic_fit_log10_power=aperiodic_fits_log10,
        full_model_log10_power=full_models_log10,
        fit_input_power_uv2_per_hz=fit_powers,
    )

    r2_values = np.full((n_rows, n_cols), np.nan)
    exponent_values = np.full((n_rows, n_cols), np.nan)
    gamma_values = np.full((n_rows, n_cols), np.nan)
    for row in rows:
        well = str(row["well"])
        wr = ord(well[0]) - ord("A")
        wc = int(well[1:]) - 1
        if int(row["fit_success"]) == 1:
            r2_values[wr, wc] = float(row["fooof_r_squared"])
            exponent_values[wr, wc] = float(row["aperiodic_exponent"])
            gamma_values[wr, wc] = float(
                row["gamma_relative_db_mean"]
            )

    plot_plate_heatmap(
        args.output_dir / "fooof_r_squared_heatmap.png",
        r2_values,
        "FOOOF fit quality per well",
        "R²",
        plate_map,
    )
    plot_plate_heatmap(
        args.output_dir / "aperiodic_exponent_heatmap.png",
        exponent_values,
        "Aperiodic exponent per well",
        "Exponent",
        plate_map,
    )
    plot_plate_heatmap(
        args.output_dir / "gamma_relative_db_heatmap.png",
        gamma_values,
        "Gamma power relative to aperiodic fit (30-100 Hz)",
        "Mean relative power (dB; 58-62 Hz excluded)",
        plate_map,
    )
    plot_group_box(
        args.output_dir / "aperiodic_exponent_by_group.png",
        rows,
        "aperiodic_exponent",
        "Aperiodic exponent by group",
        "Exponent",
    )
    plot_group_box(
        args.output_dir / "gamma_relative_db_by_group.png",
        rows,
        "gamma_relative_db_mean",
        "Gamma power relative to aperiodic fit by group",
        "Mean relative power (dB)",
    )

    representative_paths: dict[str, str] = {}
    groups = sorted(
        {
            str(row["group"])
            for row in rows
            if row.get("include") == "1"
            and row.get("fit_success") == 1
            and str(row.get("group", "")).lower() != "empty"
        }
    )
    for group in groups:
        group_rows = [
            row
            for row in rows
            if row.get("group") == group
            and row.get("include") == "1"
            and row.get("fit_success") == 1
        ]
        r2s = np.asarray(
            [float(row["fooof_r_squared"]) for row in group_rows]
        )
        median_r2 = float(np.median(r2s))
        chosen = min(
            group_rows,
            key=lambda row: abs(
                float(row["fooof_r_squared"]) - median_r2
            ),
        )
        well = str(chosen["well"])
        record = model_records[well]
        plot_path = args.output_dir / (
            f"representative_fit_{safe_name(group)}_{well}.png"
        )
        plot_representative_fit(
            plot_path,
            freqs,
            record["original_power"],
            record["fit_power"],
            record["ap_log"],
            record["full_log"],
            well,
            group,
            args.notch_low,
            args.notch_high,
        )
        representative_paths[group] = str(plot_path)

    included_success = [
        row
        for row in rows
        if row.get("include") == "1"
        and row.get("fit_success") == 1
    ]
    included_failures = [
        row
        for row in rows
        if row.get("include") == "1"
        and row.get("fit_success") != 1
    ]
    r2_included = np.asarray(
        [float(row["fooof_r_squared"]) for row in included_success],
        dtype=float,
    )
    low_quality = [
        row["well"]
        for row in included_success
        if float(row["fooof_r_squared"]) < args.r2_review_threshold
    ]

    summary = {
        "input_psd": str(args.input),
        "plate_map": str(args.plate_map),
        "fooof_version": getattr(fooof, "__version__", "unknown"),
        "fit_frequency_range_hz": [args.fmin, args.fmax],
        "frequency_resolution_hz": frequency_resolution,
        "peak_width_limits_hz": [
            args.peak_width_low,
            args.peak_width_high,
        ],
        "max_n_peaks": args.max_n_peaks,
        "min_peak_height": args.min_peak_height,
        "peak_threshold": args.peak_threshold,
        "aperiodic_mode": args.aperiodic_mode,
        "line_noise_interpolation_hz_for_fit_only": [
            args.notch_low,
            args.notch_high,
        ],
        "electrode_filtering_applied": False,
        "all_16_electrodes_used_per_well": True,
        "formal_group_statistics_applied": False,
        "n_total_wells": len(rows),
        "n_included_wells": sum(
            row.get("include") == "1" for row in rows
        ),
        "n_successful_included_fits": len(included_success),
        "n_failed_included_fits": len(included_failures),
        "failed_included_wells": [
            row["well"] for row in included_failures
        ],
        "median_r_squared_included": (
            float(np.median(r2_included))
            if r2_included.size
            else None
        ),
        "minimum_r_squared_included": (
            float(np.min(r2_included))
            if r2_included.size
            else None
        ),
        "r2_review_threshold": args.r2_review_threshold,
        "included_wells_below_r2_review_threshold": low_quality,
        "interpretation": (
            "This step separates periodic and aperiodic spectral "
            "components and produces descriptive group plots. "
            "No formal group hypothesis tests are performed."
        ),
        "output_files": {
            "per_well_csv": str(per_well_path),
            "all_peaks_csv": str(peaks_path),
            "group_descriptive_csv": str(descriptive_path),
            "models_npz": str(npz_path),
            "r_squared_heatmap": str(
                args.output_dir / "fooof_r_squared_heatmap.png"
            ),
            "exponent_heatmap": str(
                args.output_dir / "aperiodic_exponent_heatmap.png"
            ),
            "gamma_relative_heatmap": str(
                args.output_dir / "gamma_relative_db_heatmap.png"
            ),
            "exponent_group_plot": str(
                args.output_dir / "aperiodic_exponent_by_group.png"
            ),
            "gamma_group_plot": str(
                args.output_dir / "gamma_relative_db_by_group.png"
            ),
            "representative_fits": representative_paths,
        },
    }
    summary_path = args.output_dir / "fooof_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\n=== FOOOF fitting complete ===")
    print(
        "纳入孔成功拟合: "
        f"{len(included_success)}/"
        f"{summary['n_included_wells']}"
    )
    print(
        "纳入孔 R² 中位数: "
        f"{summary['median_r_squared_included']}"
    )
    print(
        f"低于 R² {args.r2_review_threshold:g} 的纳入孔: "
        f"{low_quality}"
    )
    print("正式组间统计: 未执行")
    print(f"结果目录: {args.output_dir.resolve()}")
    print(f"摘要: {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "对完整记录的每孔平均 PSD 进行 FOOOF 参数化。"
            "保留全部 16 个电极，不执行 electrode QC。"
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plate-map", type=Path, default=DEFAULT_PLATE_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=100.0)
    parser.add_argument("--notch-low", type=float, default=58.0)
    parser.add_argument("--notch-high", type=float, default=62.0)
    parser.add_argument("--peak-width-low", type=float, default=2.0)
    parser.add_argument("--peak-width-high", type=float, default=12.0)
    parser.add_argument("--max-n-peaks", type=int, default=6)
    parser.add_argument("--min-peak-height", type=float, default=0.05)
    parser.add_argument("--peak-threshold", type=float, default=1.5)
    parser.add_argument(
        "--aperiodic-mode",
        choices=["fixed", "knee"],
        default="fixed",
    )
    parser.add_argument("--r2-review-threshold", type=float, default=0.90)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
