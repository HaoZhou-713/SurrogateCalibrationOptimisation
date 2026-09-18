from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
REVISION = Path(__file__).resolve().parents[2]

SOURCES = {
    "FON": REVISION
    / "results"
    / "benchmark_FON_5methods_20seeds"
    / "fon_5methods_20seeds_with_surrogate_calibration_metrics.csv",
    "DTLZ2": REVISION
    / "results"
    / "benchmark_DTLZ2_5methods_10seeds"
    / "dtlz2_5methods_10seeds_with_surrogate_calibration_metrics.csv",
}

METHOD_ORDER = [
    "Vanilla GP",
    "Selected GP/ICM",
    "Selected + Bagging",
    "Selected + Bagging + Split CP",
    "Selected + Bagging + CV+",
]

METHOD_LABELS = {
    "Vanilla GP": "V-GP",
    "Selected GP/ICM": "Sel-GP",
    "Selected + Bagging": "Bag",
    "Selected + Bagging + Split CP": "Split-CP",
    "Selected + Bagging + CV+": "CV",
}

TIME_COLUMNS = {
    "fit_time_sec": r"$T_{\mathrm{fit}}$",
    "calibration_time_sec": r"$T_{\mathrm{cal}}$",
    "optimisation_time_sec": r"$T_{\mathrm{opt}}$",
    "total_time_sec": r"$T_{\mathrm{total}}$",
}


def summarize(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return {
            "count": 0,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }
    return {
        "count": int(clean.size),
        "median": float(np.nanmedian(clean)),
        "q25": float(np.nanquantile(clean, 0.25)),
        "q75": float(np.nanquantile(clean, 0.75)),
        "mean": float(np.nanmean(clean)),
        "std": float(np.nanstd(clean, ddof=1)) if clean.size > 1 else np.nan,
    }


def format_seconds(median: float, q25: float, q75: float) -> str:
    if pd.isna(median):
        return r"--"
    lower = min(median, q25, q75)
    upper = max(median, q25, q75)
    if upper < 0.01:
        return r"$<0.01$"
    if upper < 1:
        return f"{median:.3f} [{lower:.3f}, {upper:.3f}]"
    if upper < 10:
        return f"{median:.2f} [{lower:.2f}, {upper:.2f}]"
    return f"{median:.1f} [{lower:.1f}, {upper:.1f}]"


def aggregate_costs() -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_records: list[dict[str, object]] = []
    formatted_records: list[dict[str, object]] = []

    for benchmark, path in SOURCES.items():
        df = pd.read_csv(path)
        required = {"Method", "seed", *TIME_COLUMNS.keys()}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        for method in METHOD_ORDER:
            mdf = df[df["Method"] == method]
            if mdf.empty:
                continue
            valid_runs = int(mdf["total_time_sec"].notna().sum())

            formatted_row: dict[str, object] = {
                "Benchmark": benchmark,
                "Method": METHOD_LABELS[method],
                "Runs": valid_runs,
            }
            for column, latex_label in TIME_COLUMNS.items():
                stats = summarize(mdf[column])
                numeric_records.append(
                    {
                        "Benchmark": benchmark,
                        "Method": method,
                        "Method_label": METHOD_LABELS[method],
                        "Metric": column,
                        "Metric_label": latex_label,
                        "count": stats["count"],
                        "median_sec": stats["median"],
                        "q25_sec": stats["q25"],
                        "q75_sec": stats["q75"],
                        "mean_sec": stats["mean"],
                        "std_sec": stats["std"],
                        "source": str(path.relative_to(REVISION)),
                    }
                )
                formatted_row[latex_label] = format_seconds(
                    float(stats["median"]),
                    float(stats["q25"]),
                    float(stats["q75"]),
                )
            formatted_records.append(formatted_row)

    return pd.DataFrame(numeric_records), pd.DataFrame(formatted_records)


def write_latex_table(formatted: pd.DataFrame) -> str:
    col_labels = [
        "Benchmark",
        "Method",
        "Runs",
        r"$T_{\mathrm{fit}}$",
        r"$T_{\mathrm{cal}}$",
        r"$T_{\mathrm{opt}}$",
        r"$T_{\mathrm{total}}$",
    ]
    rows = formatted[col_labels].copy()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational cost of the controlled benchmark validation. Times are reported in seconds as median [q25, q75] across valid seed runs. Runs denotes the number of non-missing successful timing records used for aggregation. $T_{\mathrm{fit}}$ is surrogate fitting time, $T_{\mathrm{cal}}$ is uncertainty-calibration time, $T_{\mathrm{opt}}$ is downstream surrogate-assisted optimisation time, and $T_{\mathrm{total}} = T_{\mathrm{fit}} + T_{\mathrm{cal}} + T_{\mathrm{opt}}$.}",
        r"\label{tab:benchmark_cost_evaluation}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Benchmark & Method & Runs & $T_{\mathrm{fit}}$ & $T_{\mathrm{cal}}$ & $T_{\mathrm{opt}}$ & $T_{\mathrm{total}}$ \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row["Benchmark"]),
                    str(row["Method"]),
                    str(row["Runs"]),
                    str(row[r"$T_{\mathrm{fit}}$"]),
                    str(row[r"$T_{\mathrm{cal}}$"]),
                    str(row[r"$T_{\mathrm{opt}}$"]),
                    str(row[r"$T_{\mathrm{total}}$"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numeric, formatted = aggregate_costs()

    numeric.to_csv(OUT / "table_benchmark_cost_evaluation_numeric.csv", index=False)
    formatted.to_csv(OUT / "table_benchmark_cost_evaluation_formatted.csv", index=False)
    latex = write_latex_table(formatted)
    (OUT / "table_benchmark_cost_evaluation.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
