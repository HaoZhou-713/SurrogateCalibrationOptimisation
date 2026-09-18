from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
REVISION = Path(__file__).resolve().parents[2]
RESULTS = REVISION / "results"
EXPORT_WARNINGS: list[str] = []

FON_SUMMARY = RESULTS / "FON" / "FON_5methods_20seeds_summary_final.csv"
DTLZ2_SUMMARY = RESULTS / "DTLZ2" / "DTLZ2_5methods_20seeds_summary_final.csv"
DTLZ2_SENSITIVITY = RESULTS / "dtlz2_samples_sensitivity"
DTLZ2_SAMPLE_SIZE_XLSX = (
    DTLZ2_SENSITIVITY / "DTLZ2_sample_size_sensitivity_full_data_ECE95.xlsx"
)

METHOD_ORDER = [
    "Vanilla GP",
    "Selected GP/ICM",
    "Selected + Bagging",
    "Selected + Bagging + Split CP",
    "Selected + Bagging + CV",
]

METHOD_LABELS = {
    "Vanilla GP": "V-GP",
    "Selected GP/ICM": "Sel-GP",
    "Selected + Bagging": "Bag",
    "Selected + Bagging + Split CP": "Split-CP",
    "Selected + Bagging + CV": "CFSC",
}

METHOD_DISPLAY_NAMES = {
    "Vanilla GP": "Vanilla GP",
    "Selected GP/ICM": "Selected GP/ICM",
    "Selected + Bagging": "Selected + Bagging",
    "Selected + Bagging + Split CP": "Selected + Bagging + Split CP",
    "Selected + Bagging + CV": "Selected + Bagging + CFSC",
}

METHOD_COLORS = {
    "Vanilla GP": "#3B3B3B",
    "Selected GP/ICM": "#1F77B4",
    "Selected + Bagging": "#2CA02C",
    "Selected + Bagging + Split CP": "#FF7F0E",
    "Selected + Bagging + CV": "#9467BD",
}


def direction_label(label: str, higher_is_better: bool | None) -> str:
    if higher_is_better is True:
        return rf"{label} $\uparrow$"
    if higher_is_better is False:
        return rf"{label} $\downarrow$"
    return label


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_final_summaries() -> pd.DataFrame:
    frames = []
    for benchmark, path in [("FON", FON_SUMMARY), ("DTLZ2", DTLZ2_SUMMARY)]:
        df = pd.read_csv(path)
        df["Benchmark"] = benchmark
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Method"] = pd.Categorical(
        combined["Method"], categories=METHOD_ORDER, ordered=True
    )
    return combined.sort_values(["Benchmark", "Method"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = OUT / f"{stem}.png"
    pdf_path = OUT / f"{stem}.pdf"
    try:
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
    except PermissionError:
        alt_png = OUT / f"{stem}_new.png"
        fig.savefig(alt_png, dpi=600, bbox_inches="tight")
        EXPORT_WARNINGS.append(f"Could not overwrite {png_path.name}; wrote {alt_png.name}.")
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
    except PermissionError:
        alt_pdf = OUT / f"{stem}_new.pdf"
        fig.savefig(alt_pdf, bbox_inches="tight")
        EXPORT_WARNINGS.append(f"Could not overwrite {pdf_path.name}; wrote {alt_pdf.name}.")
    plt.close(fig)


def fmt_interval(row: pd.Series, metric: str, decimals: int = 3) -> str:
    med = row[f"{metric}_median"]
    q25 = row.get(f"{metric}_q25", np.nan)
    q75 = row.get(f"{metric}_q75", np.nan)
    if pd.isna(q25) or pd.isna(q75):
        return f"{med:.{decimals}f}"
    lower = min(med, q25, q75)
    upper = max(med, q25, q75)
    return f"{med:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def interval_bounds(median: float, q25: float, q75: float) -> tuple[float, float]:
    return min(median, q25, q75), max(median, q25, q75)


def make_summary_tables(summary: pd.DataFrame) -> None:
    metrics = [
        ("nRMSE_mean", "nRMSE"),
        ("R2_mean", "R2"),
        ("Coverage_mean", "Coverage"),
        ("ECE", "ECE"),
        ("IGD", "IGD"),
        ("HV_ratio", "HV ratio"),
        ("Recall", "Recall"),
    ]

    numeric_cols = ["Benchmark", "Method"]
    for metric, label in metrics:
        for suffix in ["median", "q25", "q75"]:
            col = f"{metric}_{suffix}"
            if col in summary.columns:
                numeric_cols.append(col)
    numeric = summary[numeric_cols].copy()
    numeric.to_csv(OUT / "table_5_controlled_benchmark_summary_numeric.csv", index=False)

    formatted_rows = []
    for _, row in summary.iterrows():
        out = {
            "Benchmark": row["Benchmark"],
            "Method": row["Method"],
        }
        for metric, label in metrics:
            if f"{metric}_median" in summary.columns:
                out[label] = fmt_interval(row, metric)
        formatted_rows.append(out)
    formatted = pd.DataFrame(formatted_rows)
    formatted.to_csv(OUT / "table_5_controlled_benchmark_summary_formatted.csv", index=False)
    formatted.to_latex(
        OUT / "table_5_controlled_benchmark_summary.tex",
        index=False,
        escape=True,
        column_format="llrrrrrrr",
    )

    try:
        with pd.ExcelWriter(OUT / "table_5_controlled_benchmark_summary.xlsx") as writer:
            formatted.to_excel(writer, sheet_name="formatted", index=False)
            numeric.to_excel(writer, sheet_name="numeric", index=False)
    except Exception as exc:
        (OUT / "table_5_excel_export_error.txt").write_text(str(exc), encoding="utf-8")

    # Reader-facing table figure for quick manuscript inspection. Full method
    # names stay in the editable CSV/XLSX/TeX files; codes keep the image legible.
    plot_formatted = formatted.copy()
    plot_formatted["Method"] = plot_formatted["Method"].map(METHOD_LABELS)
    fig, ax = plt.subplots(figsize=(18.5, 5.2))
    ax.axis("off")
    table = ax.table(
        cellText=plot_formatted.values,
        colLabels=plot_formatted.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 1.65)
    for (row_i, _), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.7)
        if row_i == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")
    ax.set_title(
        "Controlled benchmark validation summary (median [q25, q75])",
        pad=18,
        fontweight="bold",
    )
    save_figure(fig, "table_5_controlled_benchmark_summary")


def metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    higher_is_better: bool | None = None,
    target: float | None = None,
) -> None:
    rows = df.set_index("Method").loc[METHOD_ORDER].reset_index()
    y = np.arange(len(rows))[::-1]
    med = rows[f"{metric}_median"].to_numpy(dtype=float)
    q25 = rows[f"{metric}_q25"].to_numpy(dtype=float)
    q75 = rows[f"{metric}_q75"].to_numpy(dtype=float)
    lower = np.minimum.reduce([med, q25, q75])
    upper = np.maximum.reduce([med, q25, q75])
    xerr = np.vstack([med - lower, upper - med])
    colors = [METHOD_COLORS[m] for m in rows["Method"]]
    ax.errorbar(
        med,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.3,
        capsize=3.5,
        zorder=1,
    )
    ax.scatter(
        med,
        y,
        s=72,
        c=colors,
        edgecolor="#111111",
        linewidth=0.75,
        zorder=2,
    )
    if target is not None:
        ax.axvline(target, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title(direction_label(title, higher_is_better), fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in rows["Method"]])
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if higher_is_better is False:
        best = np.nanmin(med)
    elif higher_is_better is True:
        best = np.nanmax(med)
    else:
        best = None
    if best is not None:
        best_idx = int(np.where(med == best)[0][0])
        ax.scatter(
            med[best_idx],
            y[best_idx],
            s=142,
            facecolors="none",
            edgecolors="#000000",
            linewidth=1.6,
            zorder=3,
        )


def make_metric_grid(
    summary: pd.DataFrame,
    metrics: list[tuple[str, str, str, bool | None, float | None]],
    stem: str,
    suptitle: str,
) -> None:
    benchmarks = ["FON", "DTLZ2"]
    fig, axes = plt.subplots(
        len(metrics),
        len(benchmarks),
        figsize=(10.8, 2.85 * len(metrics) + 1.5),
        sharey="row",
    )
    for row_i, (metric, title, xlabel, higher, target) in enumerate(metrics):
        for col_i, benchmark in enumerate(benchmarks):
            bdf = summary[summary["Benchmark"] == benchmark]
            ax = axes[row_i, col_i]
            metric_panel(ax, bdf, metric, title, xlabel, higher, target)
            if row_i == 0:
                ax.text(
                    0.5,
                    1.24,
                    benchmark,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    fontweight="bold",
                )
            if col_i != 0:
                ax.tick_params(labelleft=False)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="#111111",
            label=f"{METHOD_LABELS[m]}: {METHOD_DISPLAY_NAMES[m]}",
            markersize=8,
        )
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.98), h_pad=2.0, w_pad=2.2)
    save_figure(fig, stem)


def make_sample_size_figure() -> None:
    records = []
    sample_summary = pd.read_excel(DTLZ2_SAMPLE_SIZE_XLSX, sheet_name="Key_Metrics_ECE95")
    metric_sources = ["nRMSE_mean", "ECE95", "HV_ratio", "Recall"]
    required_cols = ["Training samples", "Method"]
    required_cols += [
        f"{metric}_{suffix}"
        for metric in metric_sources
        for suffix in ["median", "q25", "q75"]
    ]
    missing = sorted(set(required_cols) - set(sample_summary.columns))
    if missing:
        raise ValueError(f"Missing sample-size columns: {missing}")

    for _, row in sample_summary.iterrows():
        n_train = int(row["Training samples"])
        method = str(row["Method"])
        for metric in metric_sources:
            median = float(row[f"{metric}_median"])
            q25 = float(row[f"{metric}_q25"])
            q75 = float(row[f"{metric}_q75"])
            lower, upper = interval_bounds(median, q25, q75)
            records.append(
                {
                    "n_train": n_train,
                    "method": method,
                    "metric": metric,
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "source": (
                        f"{DTLZ2_SAMPLE_SIZE_XLSX.relative_to(REVISION)}"
                        "::Key_Metrics_ECE95"
                    ),
                }
            )
    sens = pd.DataFrame(records)
    sens.to_csv(OUT / "table_dtlz2_sample_size_sensitivity.csv", index=False)

    panels = [
        ("nRMSE_mean", "nRMSE", False),
        ("ECE95", "ECE", False),
        ("HV_ratio", "HV ratio", True),
        ("Recall", "Recall", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
    for ax, (metric, title, higher) in zip(axes.ravel(), panels):
        sdf = sens[sens["metric"] == metric].sort_values("n_train")
        x = sdf["n_train"].to_numpy(dtype=float)
        med = sdf["median"].to_numpy(dtype=float)
        lower = sdf["interval_lower"].to_numpy(dtype=float)
        upper = sdf["interval_upper"].to_numpy(dtype=float)
        ax.plot(x, med, "-o", color="#1F77B4", linewidth=2.1, markersize=7)
        ax.fill_between(x, lower, upper, color="#1F77B4", alpha=0.16, linewidth=0)
        ax.set_title(direction_label(title, higher), fontweight="bold")
        ax.set_xlabel("Training samples")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.grid(color="#D8D8D8", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig5_3_dtlz2_sample_size_sensitivity")


def make_reliability_diagonal_figure(summary: pd.DataFrame) -> None:
    reliability = summary[
        [
            "Benchmark",
            "Method",
            "calibration_target",
            "Coverage_mean_median",
            "Coverage_mean_q25",
            "Coverage_mean_q75",
            "ECE_median",
        ]
    ].copy()
    reliability.to_csv(OUT / "table_calibration_reliability_diagonal.csv", index=False)

    benchmarks = ["FON", "DTLZ2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), sharex=True, sharey=True)
    offsets = np.linspace(-0.014, 0.014, len(METHOD_ORDER))
    for ax, benchmark in zip(axes, benchmarks):
        bdf = reliability[reliability["Benchmark"] == benchmark].set_index("Method").loc[METHOD_ORDER].reset_index()
        for offset, (_, row) in zip(offsets, bdf.iterrows()):
            method = row["Method"]
            nominal = float(row["calibration_target"])
            x = nominal + float(offset)
            median = float(row["Coverage_mean_median"])
            lower = min(median, float(row["Coverage_mean_q25"]), float(row["Coverage_mean_q75"]))
            upper = max(median, float(row["Coverage_mean_q25"]), float(row["Coverage_mean_q75"]))
            ax.errorbar(
                x,
                median,
                yerr=np.array([[median - lower], [upper - median]]),
                fmt="o",
                color=METHOD_COLORS[method],
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                markersize=8,
                elinewidth=1.25,
                capsize=3.5,
                zorder=3,
            )
        ax.plot([0.75, 1.0], [0.75, 1.0], "--", color="#333333", linewidth=1.4, label="Ideal")
        ax.axvline(0.95, color="#777777", linestyle=":", linewidth=1.1)
        ax.set_title(benchmark, fontweight="bold")
        ax.set_xlabel("Nominal coverage")
        ax.grid(color="#D8D8D8", linewidth=0.8, alpha=0.8)
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.75, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Empirical coverage")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor="#111111",
            label=f"{METHOD_LABELS[m]}",
            markersize=8,
        )
        for m in METHOD_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], linestyle="--", color="#333333", label="Ideal y=x", linewidth=1.4)
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    save_figure(fig, "fig5_4_reliability_vs_pareto")


def write_readme() -> None:
    text = """# For paper artifacts

Generated by `generate_paper_artifacts.py`.

Primary data sources:
- `revision/results/FON/FON_5methods_20seeds_summary_final.csv`
- `revision/results/DTLZ2/DTLZ2_5methods_20seeds_summary_final.csv`
- `revision/results/dtlz2_samples_sensitivity/DTLZ2_sample_size_sensitivity_full_data_ECE95.xlsx` for DTLZ2 sample-size sensitivity

Figure style:
- 600 dpi PNG exports plus vector PDF exports
- Times New Roman / serif typography with large axis labels
- Figure-level titles are omitted for manuscript use; captions should provide the figure title.
- Clean black axes, restrained colors, minimal grid
- Method medians shown with q25-q75 intervals where available
- Formatted intervals are made monotonic if a source q25/q75 entry is inconsistent; the numeric CSV preserves the source columns.

Suggested manuscript use:
- `table_5_controlled_benchmark_summary.*` for the main controlled-benchmark table.
- `fig5_1_surrogate_accuracy_calibration.*` for Sections 5.1-5.2.
- `fig5_2_pareto_performance.*` for Section 5.3, including GD, IGD, Hausdorff, HV ratio, and Recall.
- `fig5_3_dtlz2_sample_size_sensitivity.*` for Section 5.5, using the ECE95 sample-size workbook.
- `fig5_4_reliability_vs_pareto.*` for Section 5.2/5.6 as a nominal-vs-empirical calibration reliability diagram.
"""
    if EXPORT_WARNINGS:
        text += "\nExport warnings:\n"
        for warning in EXPORT_WARNINGS:
            text += f"- {warning}\n"
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    summary = read_final_summaries()
    make_summary_tables(summary)
    make_metric_grid(
        summary,
        [
            ("nRMSE_mean", "nRMSE", "nRMSE", False, None),
            ("R2_mean", "R2", "R2", True, None),
            ("Coverage_mean", "Coverage", "Empirical coverage", None, 0.95),
            ("ECE", "ECE", "ECE", False, None),
        ],
        "fig5_1_surrogate_accuracy_calibration",
        "Surrogate accuracy and calibration on controlled benchmarks",
    )
    make_metric_grid(
        summary,
        [
            ("HV_ratio", "HV ratio", "HV ratio", True, None),
            ("Recall", "Recall", "Pareto recall", True, None),
            ("GD", "GD", "GD", False, None),
            ("IGD", "IGD", "IGD", False, None),
            ("Hausdorff", "Hausdorff", "Hausdorff distance", False, None),
        ],
        "fig5_2_pareto_performance",
        "Downstream Pareto optimisation performance",
    )
    make_sample_size_figure()
    make_reliability_diagonal_figure(summary)
    write_readme()


if __name__ == "__main__":
    main()
