from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_three_pf_with_validation(
    Y_raw,
    Y_raw_uncertainty,
    Y_cfsc,
    validation_csv,
    Y_train=None,
    save_dir=None,
):
    # ---------- 读取各批次已保存的预测值与仿真真值 ----------
    df = pd.read_csv(validation_csv).copy()
    df.columns = df.columns.str.strip()

    required = [
        "framework",
        "predicted_Twall_avg",
        "predicted_ΔP(Pa)",
        "Real T",
        "Real P",
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}")

    df["framework"] = (
        df["framework"].astype(str).str.strip().str.lower()
        .replace({
            "cv": "cfsc",
            "cv+": "cfsc",
            "cal": "cfsc",
            "calibrated": "cfsc",
            "raw_unc": "raw_uncertainty",
        })
    )

    styles = {
        "raw": {
            "label": "Raw",
            "color": "#27AE60",
            "marker": "o",
        },
        "raw_uncertainty": {
            "label": "Raw + uncertainty",
            "color": "#2980B9",
            "marker": "^",
        },
        "cfsc": {
            "label": "CFSC",
            "color": "#E67E22",
            "marker": "s",
        },
    }

    unknown = set(df["framework"]) - set(styles)
    if unknown:
        raise ValueError(f"未知 framework：{unknown}")

    value_cols = required[1:]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    predicted = df[
        ["predicted_Twall_avg", "predicted_ΔP(Pa)"]
    ].to_numpy(float)
    observed = df[["Real T", "Real P"]].to_numpy(float)

    if not np.isfinite(predicted).all():
        raise ValueError("CSV 中存在缺失或无效的预测值。")

    has_truth = np.isfinite(observed).all(axis=1)
    df["truth_available"] = has_truth
    df["error_T_K"] = observed[:, 0] - predicted[:, 0]
    df["error_P_Pa"] = observed[:, 1] - predicted[:, 1]

    fronts = {
        "raw": np.asarray(Y_raw, dtype=float),
        "raw_uncertainty": np.asarray(Y_raw_uncertainty, dtype=float),
        "cfsc": np.asarray(Y_cfsc, dtype=float),
    }

    for method, front in fronts.items():
        if (
            front.ndim != 2
            or front.shape[1] != 2
            or len(front) == 0
            or not np.isfinite(front).all()
        ):
            raise ValueError(f"{method} 的 PF 必须是非空、有限的 N×2 数组。")

    # ---------- 绘图 ----------
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(11, 8.2))

        # 可选：训练数据背景
        training = None
        if Y_train is not None:
            training = np.asarray(Y_train, dtype=float)
            if (
                training.ndim != 2
                or training.shape[1] != 2
                or not np.isfinite(training).all()
            ):
                raise ValueError("Y_train 必须是有限的 N×2 数组。")

            ax.scatter(
                training[:, 0], training[:, 1],
                color="0.5",
                s=20,
                alpha=0.25,
                linewidths=0,
                zorder=1,
            )

        # 三组 PF：保留全部原始点，不重新筛选或连接
        for method, front in fronts.items():
            style = styles[method]
            ax.scatter(
                front[:, 0], front[:, 1],
                color=style["color"],
                marker=style["marker"],
                s=32,
                alpha=0.40,
                linewidths=0,
                zorder=2,
            )

        # CSV 中每条验证记录独立绘制，保留不同批次和重复仿真
        for method, style in styles.items():
            mask = df["framework"].eq(method).to_numpy()
            pred_method = predicted[mask]

            # 保存于 CSV 的原批次预测值：空心标记
            ax.scatter(
                pred_method[:, 0], pred_method[:, 1],
                marker=style["marker"],
                facecolors="none",
                edgecolors=style["color"],
                s=150,
                linewidths=2.3,
                zorder=5,
            )

            valid = mask & has_truth
            pred_valid = predicted[valid]
            true_valid = observed[valid]

            # 仿真真值：同色叉号
            ax.scatter(
                true_valid[:, 0], true_valid[:, 1],
                marker="x",
                color=style["color"],
                s=150,
                linewidths=2.8,
                zorder=6,
            )

            # 逐条连接该记录的预测值与仿真真值
            for y_pred, y_true in zip(pred_valid, true_valid):
                ax.plot(
                    [y_pred[0], y_true[0]],
                    [y_pred[1], y_true[1]],
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.70,
                    zorder=3,
                )

        # ---------- 图例 ----------
        handles = []
        for style in styles.values():
            handles.append(
                Line2D(
                    [], [],
                    linestyle="none",
                    marker=style["marker"],
                    markerfacecolor=style["color"],
                    markeredgecolor="none",
                    markersize=8,
                    alpha=0.45,
                    label=f'{style["label"]} Pareto front',
                )
            )

        for method, style in styles.items():
            if df["framework"].eq(method).any():
                handles.append(
                    Line2D(
                        [], [],
                        linestyle="none",
                        marker=style["marker"],
                        markerfacecolor="none",
                        markeredgecolor=style["color"],
                        markeredgewidth=2,
                        markersize=9,
                        label=f'Prediction ({style["label"]})',
                    )
                )

        if has_truth.any():
            handles.append(
                Line2D(
                    [], [],
                    linestyle="none",
                    marker="x",
                    color="black",
                    markeredgewidth=2,
                    markersize=9,
                    label="Simulation truth",
                )
            )

        if training is not None:
            handles.append(
                Line2D(
                    [], [],
                    linestyle="none",
                    marker="o",
                    color="0.5",
                    alpha=0.4,
                    markersize=6,
                    label="Training data",
                )
            )

        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=10,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
        )

        ax.set_xlabel(r"Wall Average Temperature $T_{\mathrm{avg}}$ (K)")
        ax.set_ylabel(r"Pressure Drop $\Delta P$ (Pa)")
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(
            True,
            linestyle="--",
            color="0.85",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.set_axisbelow(True)
        ax.margins(x=0.05, y=0.07)
        fig.tight_layout()

        # ---------- 保存新图和对应数据 ----------
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = "simulation_three_pf_validation_from_csv"

            fig.savefig(
                save_dir / f"{stem}.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                save_dir / f"{stem}.png",
                dpi=600,
                bbox_inches="tight",
            )

            # 每条记录独立保存，不按 framework/index 去重
            df.to_csv(
                save_dir / f"{stem}_validation.csv",
                index=False,
            )

            pf_tables = []
            for method, front in fronts.items():
                table = pd.DataFrame(
                    front,
                    columns=["predicted_Twall_avg", "predicted_ΔP(Pa)"],
                )
                table.insert(0, "point_index", np.arange(len(front)))
                table.insert(0, "framework", method)
                pf_tables.append(table)

            pd.concat(pf_tables, ignore_index=True).to_csv(
                save_dir / f"{stem}_pf.csv",
                index=False,
            )

            if training is not None:
                pd.DataFrame(
                    training,
                    columns=["Twall_avg", "ΔP(Pa)"],
                ).to_csv(
                    save_dir / f"{stem}_training.csv",
                    index=False,
                )

            print(f"保存目录：{save_dir.resolve()}")

        plt.show()

    print(
        f"验证记录：{len(df)} 条；"
        f"完整仿真真值：{has_truth.sum()} 条；"
        f"缺少完整真值：{(~has_truth).sum()} 条。"
    )
    return fig, ax, df

