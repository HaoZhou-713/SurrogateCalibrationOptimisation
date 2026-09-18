import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from surrogate_pareto_botorch import surrogate_pareto_front_nsga2  # 你已有
# calibrate_from_cv_plus_up 是你写的 CV+ calibrator
# choose_ref_point_max / hypervolume_max / to_max_space 如果你已有就用；没有我下面给一个最简 HV

# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_bounds_from_auto(auto):
    x_min  = auto.x_min.detach().cpu().numpy()
    x_span = auto.x_span.detach().cpu().numpy()
    return np.column_stack([x_min, x_min + x_span])  # (d,2)

def pareto_front_min(Y):
    """Return boolean mask of Pareto-optimal points for minimization (O(N^2), PF size typically small)."""
    Y = np.asarray(Y, float)
    N = Y.shape[0]
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_efficient[i]:
            continue
        # any point that dominates i -> i not efficient
        dominates_i = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)
        if np.any(dominates_i):
            is_efficient[i] = False
            continue
        # remove points dominated by i
        dominated_by_i = np.all(Y[i] <= Y, axis=1) & np.any(Y[i] < Y, axis=1)
        is_efficient[dominated_by_i] = False
        is_efficient[i] = True
    return is_efficient

def hv_2d_min(Y, ref):
    """
    Hypervolume for 2D minimization.
    Y: (N,2) PF points in minimization sense
    ref: (2,) reference point (worse than all PF points)
    """
    Y = np.asarray(Y, float)
    ref = np.asarray(ref, float)
    if Y.shape[0] == 0:
        return 0.0
    # keep Pareto front
    m = pareto_front_min(Y)
    P = Y[m]
    # sort by f1 ascending
    P = P[np.argsort(P[:,0])]
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in P:
        # rectangle: [f1, ref1] x [f2, prev_f2]
        w = max(0.0, ref[0] - f1)
        h = max(0.0, prev_f2 - f2)
        hv += w * h
        prev_f2 = min(prev_f2, f2)
    return float(hv)

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

import torch
import numpy as np
from botorch.utils.multi_objective.hypervolume import Hypervolume

def hypervolume_min_botorch_normalized(
    Y_min,
    ref_min,
    Y_pool=None,
    eps=1e-12,
):
    """
    Normalized hypervolume for minimization problems (multi-D).

    Parameters
    ----------
    Y_min : (N, M)
        Pareto front points (minimization).
    ref_min : (M,)
        Reference point in ORIGINAL space (worse than PF).
    Y_pool : (Nref, M), optional
        Pool used to compute normalization (recommended: all training Y).
        If None, will normalize using PF itself (NOT recommended).
    eps : float
        Numerical stability.

    Returns
    -------
    hv : float
        Normalized hypervolume.
    """

    Y_min = np.asarray(Y_min, float)
    ref_min = np.asarray(ref_min, float)

    if Y_pool is None:
        Y_pool = np.vstack([Y_min, ref_min[None, :]])
    else:
        Y_pool = np.asarray(Y_pool, float)

    # ---------- min–max normalization ----------
    lo = Y_pool.min(axis=0)
    hi = Y_pool.max(axis=0)

    Yn = (Y_min - lo) / (hi - lo + eps)
    refn = (ref_min - lo) / (hi - lo + eps)

    # ---------- minimization → maximization ----------
    Y_max = -torch.as_tensor(Yn, dtype=torch.double)
    ref_max = -torch.as_tensor(refn, dtype=torch.double)

    hv = Hypervolume(ref_point=ref_max)
    return float(hv.compute(Y_max))



def pf_spread(Y):
    """Bounding-box diagonal length in objective space (min space)."""
    Y = np.asarray(Y, float)
    if Y.shape[0] == 0:
        return np.nan
    mins = Y.min(axis=0)
    maxs = Y.max(axis=0)
    return float(np.linalg.norm(maxs - mins))

def run_pf_once(auto, use_calibration: bool, *, transform='mean', beta=1., pop_size=256, n_gen=150, seed=0):
    bounds = get_bounds_from_auto(auto)
    res = surrogate_pareto_front_nsga2(
        auto,
        bounds=bounds,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        model_key="final_models",
        sense=["min", "min", "min"],
        use_hull=False,
        hull_ref="all",
        transform=transform,
        beta=beta,
        use_calibration=use_calibration,
        calibrator=calibrate_for_auto if use_calibration else None,  # 你已有的 hook
    )
    # 你函数里 Y_pareto 是 mean（mu），这是我们要的“可解释 PF”
    Xp = res.X_pareto
    Yp = res.Y_pareto
    return Xp, Yp

def normalize_Y_minmax(Y, Y_ref):
    """
    Y: (N,M) 需要归一化的数据
    Y_ref: (Nref,M) 用来确定缩放尺度，比如训练集/全数据 Y_pool
    """
    Y = np.asarray(Y, float)
    Y_ref = np.asarray(Y_ref, float)
    lo = Y_ref.min(axis=0)
    hi = Y_ref.max(axis=0)
    return (Y - lo) / (hi - lo + 1e-12)

def pf_spread_normalized(Y_pf, Y_pool):
    Yn = normalize_Y_minmax(Y_pf, Y_pool)
    mins = Yn.min(axis=0)
    maxs = Yn.max(axis=0)
    return float(np.linalg.norm(maxs - mins))

# ----------------------------
# Main experiment
# ----------------------------
def multi_seed_pf_compare(
    make_auto,
    X, Y,
    *,
    seeds=range(20),
    alpha=0.10,
    K=5,
    pop_size=256,
    n_gen=150,
    hv_margin=0.05,  # ref point margin
):
    records = []
    pf_store = {"raw": [], "raw_uncertainty": [], "cfsc": []}

    # reference point for HV in surrogate mean space:
    # use pool of Y (all data) in min sense, take "worse than max" with margin
    Y_pool = np.asarray(Y, float)
    ref = Y_pool.max(axis=0) + hv_margin * (Y_pool.max(axis=0) - Y_pool.min(axis=0) + 1e-12)

    for seed in seeds:
        set_global_seed(int(seed))

        auto = make_auto(seed)  # 你自己的 factory，确保 seed 生效
        auto.fit(X, Y)
        if hasattr(auto, "finalize"):
            auto.finalize(X, Y, set_as_current=True)

        # ---------- PF: no calibration ----------
        X_raw, Y_raw = run_pf_once(auto, use_calibration=False, pop_size=pop_size, n_gen=n_gen, seed=seed)
        pf_store["raw"].append((seed, X_raw, Y_raw))

        hv_raw = hypervolume_min_botorch_normalized(
            Y_raw,
            ref,
            Y_pool=Y_pool
        )
        spread_raw = pf_spread_normalized(Y_raw, Y_pool)

        records.append({
            "seed": int(seed),
            "method": "raw",
            "HV_mu": hv_raw,
            "spread": spread_raw,
            "n_pf": int(len(Y_raw)),
        })

        # ---------- PF: Raw uncertainty ----------
        X_raw_unc, Y_raw_unc = run_pf_once(
            auto,
            use_calibration=False,
            transform="ucb",
            beta=1.,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )

        pf_store["raw_uncertainty"].append(
            (seed, X_raw_unc, Y_raw_unc)
        )


        hv_raw_unc = hypervolume_min_botorch_normalized(
            Y_raw_unc,
            ref,
            Y_pool=Y_pool
        )
        spread_raw_unc = pf_spread_normalized(Y_raw_unc, Y_pool)

        records.append({
            "seed": int(seed),
            "method": "raw_uncertainty",
            "HV_mu": hv_raw_unc,
            "spread": spread_raw_unc,
            "n_pf": int(len(Y_raw_unc)),
        })

        # ---------- CV+ calibration ----------
        auto.calibration = calibrate_from_cv_plus_up(
            auto=auto,
            X=auto.X_all,
            Y=auto.Y_all,
            K=K,
            alpha=alpha,
            method="per_target",
            model_key="final_models",  # 强烈建议对齐 final_models
            seed=int(seed),
        )

        # ⚠️ 不要在 calibration 后 refit model（否则 q 和模型不匹配）
        # 你如果 finalize 会 refit，就别调 finalize；只需要确保 predict 会读取 auto.calibration
        # 如果你的 finalize 只是 set_as_current，那可以保留；否则注释掉
        # auto.finalize(X, Y, set_as_current=True)

        # ---------- PF: with calibration ----------
        X_cal, Y_cal = run_pf_once(auto, use_calibration=True, pop_size=pop_size, n_gen=n_gen, seed=seed)
        pf_store["cfsc"].append((seed, X_cal, Y_cal))

        hv_cal = hypervolume_min_botorch_normalized(
            Y_cal,
            ref,
            Y_pool=Y_pool
        )
        spread_cal = pf_spread_normalized(Y_cal, Y_pool)

        records.append({
            "seed": int(seed),
            "method": "cfsc",
            "HV_mu": hv_cal,
            "spread": spread_cal,
            "n_pf": int(len(Y_cal)),
        })

        print(f"[seed {seed}] raw HV={hv_raw:.4g}, raw uncertainty HV={hv_raw_unc:.4g}, cal HV={hv_cal:.4g} | spread raw={spread_raw:.4g}, raw uncertainty spread={spread_raw_unc:.4g}, cal spread={spread_cal:.4g}")

    return pd.DataFrame(records), pf_store
