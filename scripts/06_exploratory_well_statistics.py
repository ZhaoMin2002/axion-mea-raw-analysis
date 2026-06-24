"""Run exploratory well-level group comparisons for Plate 8 / 4039.

The default input is the fixed-mode 1-100 Hz FOOOF result. The knee-mode run is
retained as a sensitivity analysis, but is not used as the primary model here.

Important limitation
--------------------
All wells come from one plate. The p-values therefore describe within-plate,
well-level differences and must not be presented as independent-batch biological
replication. No electrode is removed. Each well already represents the average
PSD across all 16 electrodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_INPUT = Path("outputs/plate8_4039/fooof_full_10_290/fooof_per_well.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/plate8_4039/statistics_fixed_full_10_290")

METRICS: dict[str, tuple[str, str]] = {
    "aperiodic_exponent": ("Aperiodic exponent", "Exponent"),
    "aperiodic_offset": ("Aperiodic offset", "Offset (log10 power)"),
    "delta_relative_db_mean": ("Delta power relative to aperiodic fit", "Mean relative power (dB)"),
    "theta_relative_db_mean": ("Theta power relative to aperiodic fit", "Mean relative power (dB)"),
    "alpha_relative_db_mean": ("Alpha power relative to aperiodic fit", "Mean relative power (dB)"),
    "beta_relative_db_mean": ("Beta power relative to aperiodic fit", "Mean relative power (dB)"),
    "gamma_relative_db_mean": ("Gamma power relative to aperiodic fit", "Mean relative power (dB)"),
}


def parse_numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if np.isfinite(result) else math.nan


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 FOOOF per-well CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError("CSV 没有表头。")
        required = {"well", "group", "include", "fit_success", *METRICS.keys()}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise KeyError(f"CSV 缺少字段: {sorted(missing)}")
        rows = list(reader)

    selected: list[dict[str, Any]] = []
    for row in rows:
        group = str(row.get("group", "")).strip()
        include = str(row.get("include", "")).strip()
        fit_success = str(row.get("fit_success", "")).strip()
        if include != "1" or fit_success != "1" or group.lower() == "empty":
            continue

        clean: dict[str, Any] = {
            "well": str(row["well"]).strip(),
            "group": group,
        }
        for metric in METRICS:
            clean[metric] = parse_numeric(row.get(metric))
        selected.append(clean)

    if not selected:
        raise ValueError("没有符合 include=1 且 fit_success=1 的 well。")
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_values(rows: list[dict[str, Any]], metric: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            groups.setdefault(str(row["group"]), []).append(value)
    return {
        group: np.asarray(values, dtype=float)
        for group, values in sorted(groups.items())
        if values
    }


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return output.tolist()

    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[finite] = restored
    return output.tolist()


def holm_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return output.tolist()

    finite_values = values[finite]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    m = ranked.size
    adjusted = (m - np.arange(m)) * ranked
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[finite] = restored
    return output.tolist()


def welch_anova(groups: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Return Welch F, p, numerator df and denominator df."""
    arrays = [values for values in groups.values() if values.size >= 2]
    k = len(arrays)
    if k < 2:
        return math.nan, math.nan, math.nan, math.nan

    ns = np.asarray([values.size for values in arrays], dtype=float)
    means = np.asarray([values.mean() for values in arrays], dtype=float)
    variances = np.asarray([values.var(ddof=1) for values in arrays], dtype=float)
    if np.any(variances <= 0):
        return math.nan, math.nan, math.nan, math.nan

    weights = ns / variances
    weight_sum = weights.sum()
    weighted_mean = np.sum(weights * means) / weight_sum
    numerator = np.sum(weights * np.square(means - weighted_mean)) / (k - 1)
    correction_term = np.sum(np.square(1.0 - weights / weight_sum) / (ns - 1.0))
    denominator = 1.0 + (2.0 * (k - 2.0) / (k * k - 1.0)) * correction_term
    f_value = numerator / denominator
    df1 = float(k - 1)
    df2 = float((k * k - 1.0) / (3.0 * correction_term))
    p_value = float(stats.f.sf(f_value, df1, df2))
    return float(f_value), p_value, df1, df2


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    n1, n2 = first.size, second.size
    if n1 < 2 or n2 < 2:
        return math.nan
    variance = (
        (n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)
    ) / (n1 + n2 - 2)
    if variance <= 0:
        return math.nan
    cohen_d = (first.mean() - second.mean()) / math.sqrt(variance)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(correction * cohen_d)


def build_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric, (title, _) in METRICS.items():
        for group, values in group_values(rows, metric).items():
            q1, q3 = np.quantile(values, [0.25, 0.75])
            output.append(
                {
                    "metric": metric,
                    "metric_label": title,
                    "group": group,
                    "n_wells": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "sem": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return output


def build_omnibus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric, (title, _) in METRICS.items():
        groups = group_values(rows, metric)
        f_value, welch_p, df1, df2 = welch_anova(groups)
        arrays = list(groups.values())
        if len(arrays) >= 2:
            kruskal = stats.kruskal(*arrays)
            levene = stats.levene(*arrays, center="median")
            kruskal_h = float(kruskal.statistic)
            kruskal_p = float(kruskal.pvalue)
            levene_w = float(levene.statistic)
            levene_p = float(levene.pvalue)
        else:
            kruskal_h = kruskal_p = levene_w = levene_p = math.nan

        output.append(
            {
                "metric": metric,
                "metric_label": title,
                "n_groups": len(groups),
                "welch_anova_f": f_value,
                "welch_anova_df1": df1,
                "welch_anova_df2": df2,
                "welch_anova_p": welch_p,
                "kruskal_h": kruskal_h,
                "kruskal_p": kruskal_p,
                "brown_forsythe_w": levene_w,
                "brown_forsythe_p": levene_p,
            }
        )

    welch_q = benjamini_hochberg([float(row["welch_anova_p"]) for row in output])
    kruskal_q = benjamini_hochberg([float(row["kruskal_p"]) for row in output])
    for row, q_welch, q_kruskal in zip(output, welch_q, kruskal_q):
        row["welch_anova_fdr_bh_q"] = q_welch
        row["kruskal_fdr_bh_q"] = q_kruskal
    return output


def ordered_pairs(groups: list[str], control_group: str) -> list[tuple[str, str]]:
    pairs = list(combinations(groups, 2))
    return sorted(
        pairs,
        key=lambda pair: (
            0 if control_group in pair else 1,
            pair[0],
            pair[1],
        ),
    )


def build_pairwise(
    rows: list[dict[str, Any]], control_group: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric, (title, _) in METRICS.items():
        groups = group_values(rows, metric)
        metric_rows: list[dict[str, Any]] = []
        for first_name, second_name in ordered_pairs(list(groups), control_group):
            first = groups[first_name]
            second = groups[second_name]
            test = stats.ttest_ind(first, second, equal_var=False)
            metric_rows.append(
                {
                    "metric": metric,
                    "metric_label": title,
                    "group_1": first_name,
                    "group_2": second_name,
                    "control_comparison": int(control_group in {first_name, second_name}),
                    "n_1": int(first.size),
                    "n_2": int(second.size),
                    "mean_1": float(first.mean()),
                    "mean_2": float(second.mean()),
                    "mean_difference_1_minus_2": float(first.mean() - second.mean()),
                    "welch_t": float(test.statistic),
                    "welch_p": float(test.pvalue),
                    "hedges_g_1_minus_2": hedges_g(first, second),
                }
            )

        adjusted = holm_adjust([float(row["welch_p"]) for row in metric_rows])
        for row, p_holm in zip(metric_rows, adjusted):
            row["welch_holm_p_within_metric"] = p_holm
        output.extend(metric_rows)

    global_q = benjamini_hochberg([float(row["welch_p"]) for row in output])
    for row, q_value in zip(output, global_q):
        row["welch_global_fdr_bh_q"] = q_value
    return output


def plot_metric(
    path: Path,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    control_group: str,
) -> None:
    groups = group_values(rows, metric)
    names = list(groups)
    if control_group in names:
        names = [control_group] + [name for name in names if name != control_group]
    arrays = [groups[name] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(arrays, tick_labels=names, showmeans=True)
    rng = np.random.default_rng(4039)
    for index, values in enumerate(arrays, start=1):
        jitter = rng.normal(0.0, 0.045, size=values.size)
        ax.scatter(np.full(values.size, index) + jitter, values, s=24, alpha=0.75)
    ax.set_title(f"{title} (single-plate exploratory analysis)")
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    rows = load_rows(args.input)
    groups = sorted({str(row["group"]) for row in rows})
    if args.control_group not in groups:
        raise ValueError(
            f"control group {args.control_group!r} 不在数据组别 {groups} 中。"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_summary = build_group_summary(rows)
    omnibus = build_omnibus(rows)
    pairwise = build_pairwise(rows, args.control_group)

    group_summary_path = args.output_dir / "group_summary.csv"
    omnibus_path = args.output_dir / "omnibus_tests.csv"
    pairwise_path = args.output_dir / "pairwise_welch_tests.csv"
    write_csv(group_summary_path, group_summary)
    write_csv(omnibus_path, omnibus)
    write_csv(pairwise_path, pairwise)

    plot_paths: dict[str, str] = {}
    for metric, (title, ylabel) in METRICS.items():
        path = plot_dir / f"{metric}_by_group.png"
        plot_metric(path, rows, metric, title, ylabel, args.control_group)
        plot_paths[metric] = str(path)

    significant_welch = [
        row["metric"]
        for row in omnibus
        if np.isfinite(float(row["welch_anova_fdr_bh_q"]))
        and float(row["welch_anova_fdr_bh_q"]) < args.alpha
    ]
    significant_pairs = [
        {
            "metric": row["metric"],
            "group_1": row["group_1"],
            "group_2": row["group_2"],
            "holm_p": row["welch_holm_p_within_metric"],
            "hedges_g": row["hedges_g_1_minus_2"],
        }
        for row in pairwise
        if np.isfinite(float(row["welch_holm_p_within_metric"]))
        and float(row["welch_holm_p_within_metric"]) < args.alpha
    ]

    summary = {
        "input": str(args.input),
        "model_label": args.model_label,
        "control_group": args.control_group,
        "groups": groups,
        "n_included_wells": len(rows),
        "metrics": list(METRICS),
        "alpha": args.alpha,
        "electrode_filtering_applied": False,
        "all_16_electrodes_used_per_well": True,
        "statistical_unit": "well",
        "single_plate_exploratory_analysis": True,
        "independent_batch_replication": False,
        "primary_omnibus_test": "Welch one-way ANOVA",
        "omnibus_multiple_testing": "Benjamini-Hochberg FDR across metrics",
        "pairwise_test": "Welch independent-samples t-test",
        "pairwise_multiple_testing": "Holm within each metric; global BH FDR also reported",
        "effect_size": "Hedges g; sign is group_1 minus group_2",
        "significant_omnibus_metrics_after_fdr": significant_welch,
        "significant_pairwise_comparisons_after_holm": significant_pairs,
        "interpretation_warning": (
            "All wells are from one plate. Results are exploratory within-plate "
            "comparisons and cannot establish independent-batch biological "
            "replication. Frequency-band values depend on the selected spectral "
            "parameterization model."
        ),
        "output_files": {
            "group_summary": str(group_summary_path),
            "omnibus_tests": str(omnibus_path),
            "pairwise_tests": str(pairwise_path),
            "plots": plot_paths,
        },
    }
    summary_path = args.output_dir / "statistics_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("=== Exploratory well-level statistics complete ===")
    print(f"模型标签: {args.model_label}")
    print(f"组别: {groups}")
    print(f"纳入 wells: {len(rows)}")
    print(f"FDR 后显著的 omnibus metrics: {significant_welch}")
    print(f"Holm 后显著的 pairwise comparisons: {significant_pairs}")
    print("限制: 单块 plate 的 well-level 探索性结果，不代表独立批次复现。")
    print(f"结果目录: {args.output_dir.resolve()}")
    print(f"摘要: {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "对 FOOOF per-well 结果进行单板、well-level 探索性组间统计。"
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control-group", default="7889")
    parser.add_argument("--model-label", default="fixed_1_100_primary")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
