import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull


def local_support_union_3d(Y_train_3d, pf_store_side, k=5, max_seeds=20):
    """
    在 3D 目标空间里做 kNN，返回聚合后的 local support 点 (3D)
    """
    Y_train_3d = np.asarray(Y_train_3d, float)
    nbrs = NearestNeighbors(n_neighbors=k).fit(Y_train_3d)

    idx_all = []
    for seed, Xp, Yp in pf_store_side[:max_seeds]:
        Yp = np.asarray(Yp, float)
        idx = nbrs.kneighbors(Yp, return_distance=False)
        idx_all.append(idx.ravel())

    idx_all = np.unique(np.concatenate(idx_all))
    return Y_train_3d[idx_all]  # (Ns,3)


def draw_convex_hull_2d(ax, Y2, *, color="0.3", alpha=0.18, lw=1.0, label=None):
    """
    在 2D 投影里画 hull
    """
    Y2 = np.unique(np.asarray(Y2, float), axis=0)
    if Y2.shape[0] < 3:
        return

    hull = ConvexHull(Y2)
    poly = Y2[hull.vertices]

    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=0, label=label)
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, alpha=0.6, zorder=1)


def plot_pf_overlay_with_support_3d_pairwise(
    pf_store,
    Y_train_3d,
    *,
    k_support=5,
    max_seeds=20,
    pairs=((0,1), (0,2), (1,2)),
    axis_labels=("Obj1 (mean)", "Obj2 (mean)", "Obj3 (mean)"),
    s_pf=10,
    alpha_pf=0.25,
    s_sup=18,
    alpha_sup=0.45,
    dpi=600,
    save_path=None,
    save_pdf_path=None,
):
    """
    画 3D PF 的两两投影：
      - 每行一个 pair (i,j)
      - 每行两列：左 raw / 右 cal
    背景：local support hull + local support points（来自 3D kNN，投影到 2D）
    """

    Y_sup_raw_3d = local_support_union_3d(
            Y_train_3d, pf_store["raw"], k=k_support, max_seeds=max_seeds
        )

    Y_sup_raw_unc_3d = local_support_union_3d(
        Y_train_3d, pf_store["raw_uncertainty"], k=k_support, max_seeds=max_seeds
    )
    Y_sup_cal_3d = local_support_union_3d(
        Y_train_3d, pf_store["cfsc"], k=k_support, max_seeds=max_seeds
    )

    nrows = len(pairs)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(14, 4.2 * nrows),
        dpi=dpi
    )

    if nrows == 1:
        axes = np.array([axes])

    for r, (i, j) in enumerate(pairs):

        # ---------------- raw ----------------
        ax = axes[r, 0]

        Y_sup_2d = Y_sup_raw_3d[:, [i, j]]
        draw_convex_hull_2d(
            ax,
            Y_sup_2d,
            color="0.3",
            alpha=0.18,
            label=f"Local support hull (k={k_support})"
        )

        ax.scatter(
            Y_sup_2d[:, 0],
            Y_sup_2d[:, 1],
            c="k",
            s=s_sup,
            alpha=alpha_sup,
            zorder=2,
            label="Local training points"
        )

        for seed, Xp, Yp in pf_store["raw"][:max_seeds]:
            Yp = np.asarray(Yp, float)
            ax.scatter(
                Yp[:, i],
                Yp[:, j],
                s=s_pf,
                alpha=alpha_pf,
                zorder=3
            )

        ax.set_title(f"Raw | ({i+1},{j+1})")
        ax.set_xlabel(axis_labels[i])
        ax.set_ylabel(axis_labels[j])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best")

        # ---------------- Raw+Uncertainty ----------------
        ax = axes[r, 1]

        Y_sup_2d = Y_sup_raw_unc_3d[:, [i, j]]
        draw_convex_hull_2d(
            ax,
            Y_sup_2d,
            color="0.3",
            alpha=0.18,
            label=f"Local support hull (k={k_support})"
        )

        ax.scatter(
            Y_sup_2d[:, 0],
            Y_sup_2d[:, 1],
            c="k",
            s=s_sup,
            alpha=alpha_sup,
            zorder=2,
            label="Local training points"
        )

        for seed, Xp, Yp in pf_store["raw_uncertainty"][:max_seeds]:
            Yp = np.asarray(Yp, float)
            ax.scatter(
                Yp[:, i],
                Yp[:, j],
                s=s_pf,
                alpha=alpha_pf,
                zorder=3
            )

        ax.set_title(f"Raw+Uncertainty | ({i+1},{j+1})")
        ax.set_xlabel(axis_labels[i])
        ax.set_ylabel(axis_labels[j])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best")

        # ---------------- calibrated ----------------
        ax = axes[r, 2]

        Y_sup_2d = Y_sup_cal_3d[:, [i, j]]
        draw_convex_hull_2d(
            ax,
            Y_sup_2d,
            color="0.3",
            alpha=0.18,
            label=f"Local support hull (k={k_support})"
        )

        ax.scatter(
            Y_sup_2d[:, 0],
            Y_sup_2d[:, 1],
            c="k",
            s=s_sup,
            alpha=alpha_sup,
            zorder=2,
            label="Local training points"
        )

        for seed, Xp, Yp in pf_store["cfsc"][:max_seeds]:
            Yp = np.asarray(Yp, float)
            ax.scatter(
                Yp[:, i],
                Yp[:, j],
                s=s_pf,
                alpha=alpha_pf,
                zorder=3
            )

        ax.set_title(f"Calibrated (CFSC) | ({i+1},{j+1})")
        ax.set_xlabel(axis_labels[i])
        ax.set_ylabel(axis_labels[j])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if save_pdf_path is not None:
        fig.savefig(save_pdf_path, bbox_inches="tight")

    plt.show()
