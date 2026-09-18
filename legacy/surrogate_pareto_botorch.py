"""
surrogate_pareto_botorch.py

Sampling-based Pareto front extraction from a trained surrogate model (e.g., AutoMultiOutputGP),
using BoTorch's built-in Pareto utilities.

What you get
------------
- Candidate sampling in a box (Sobol / Uniform)
- Optional "no extrapolation" filtering: keep only points inside the convex hull of reference X
- Optional calibration hook: calibrator(mu, sd) -> (mu_cal, sd_cal)
- Model selection interface: choose which model set stored in `auto` to use
  ("models" | "eval_models" | "final_models")
- Pareto front computed via BoTorch: `is_non_dominated`
- (Optional) also returns a `NondominatedPartitioning` object for HV / (q)EHVI workflows

Assumptions about `auto`
------------------------
- auto.predict(X) -> (mu, sd) in ORIGINAL units, both shape (N, M)
- auto may store different model sets:
    auto.models, auto.eval_models, auto.final_models
- auto stores split & data if you want hull filtering:
    auto.split["tr"/"va"/"te"], auto.X_all
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
Sense = Union[Literal["min"], Literal["max"]]
ModelKey = Union[Literal["models"], Literal["eval_models"], Literal["final_models"]]

Calibrator = Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]


# =========================
# Helpers
# =========================

def _as_2d_np(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    return x


def _sense_to_minimization(Y: np.ndarray, sense: Sequence[Sense]) -> np.ndarray:
    """
    Convert mixed min/max objectives into a pure minimization representation
    by flipping sign for "max" objectives.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be (N,M)")
    if len(sense) != Y.shape[1]:
        raise ValueError(f"sense length must equal #objectives (got {len(sense)} vs {Y.shape[1]})")
    Ymin = Y.copy()
    for j, s in enumerate(sense):
        sj = s.lower()
        if sj == "max":
            Ymin[:, j] = -Ymin[:, j]
        elif sj != "min":
            raise ValueError("sense entries must be 'min' or 'max'")
    return Ymin


# =========================
# Sampling
# =========================

def sample_uniform(bounds: ArrayLike, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    bounds = _as_2d_np(bounds)
    if bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (d,2) as [[lb,ub], ...]")
    d = bounds.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random((n, d))
    lb = bounds[:, 0][None, :]
    ub = bounds[:, 1][None, :]
    return lb + u * (ub - lb)


def sample_sobol(bounds: ArrayLike, n: int, scramble: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """
    Sobol sampling in a box. Tries SciPy; falls back to uniform if unavailable.
    """
    bounds = _as_2d_np(bounds)
    if bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (d,2) as [[lb,ub], ...]")
    d = bounds.shape[0]
    try:
        from scipy.stats import qmc  # type: ignore
        sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
        u = sampler.random(n=n)
    except Exception:
        rng = np.random.default_rng(seed)
        u = rng.random((n, d))
    lb = bounds[:, 0][None, :]
    ub = bounds[:, 1][None, :]
    return lb + u * (ub - lb)


# =========================
# In-hull filtering (avoid extrapolation)
# =========================

def in_convex_hull_mask(
    X: np.ndarray,
    X_ref: np.ndarray,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Mask of points in X that lie inside the convex hull of X_ref.

    Implementation:
    - Prefer SciPy Delaunay.
    - Fall back to bounding-box filter if SciPy unavailable or hull is degenerate.

    Caveat:
    - High-dimensional hull checks can be costly/fragile. Consider ellipsoid / kNN
      domain filters if d is large.
    """
    X = _as_2d_np(X)
    X_ref = _as_2d_np(X_ref)
    d = X_ref.shape[1]

    if X_ref.shape[0] < d + 1:
        lb = X_ref.min(axis=0)
        ub = X_ref.max(axis=0)
        return np.all((X >= lb[None, :] - tol) & (X <= ub[None, :] + tol), axis=1)

    try:
        from scipy.spatial import Delaunay  # type: ignore
        tri = Delaunay(X_ref)
        return tri.find_simplex(X) >= 0
    except Exception:
        lb = X_ref.min(axis=0)
        ub = X_ref.max(axis=0)
        return np.all((X >= lb[None, :] - tol) & (X <= ub[None, :] + tol), axis=1)

def in_distribution_mask_knn_md(
    X_query_raw,
    X_ref_raw,
    *,
    x_min,
    x_span,
    q_knn=0.95,
    q_md=0.95,
    ridge=1e-9,
):
    """
    In-distribution filter in SCALED space using:
      - kNN min-distance gate
      - Mahalanobis distance gate
    """
    # scale
    def scale(X):
        return (X - x_min) / x_span

    X_ref = scale(X_ref_raw)
    Xq    = scale(X_query_raw)

    # ---------- kNN gate ----------
    q2 = np.sum(Xq**2, axis=1, keepdims=True)
    r2 = np.sum(X_ref**2, axis=1, keepdims=True).T
    dist2 = q2 + r2 - 2.0 * (Xq @ X_ref.T)
    np.maximum(dist2, 0.0, out=dist2)
    knn_q = np.sqrt(np.min(dist2, axis=1))

    # ref NN distances
    D2 = np.sum(X_ref**2, axis=1, keepdims=True) \
       + np.sum(X_ref**2, axis=1, keepdims=True).T \
       - 2.0 * (X_ref @ X_ref.T)
    np.fill_diagonal(D2, np.inf)
    tr_knn = np.sqrt(np.min(np.maximum(D2, 0.0), axis=1))
    thr_knn = np.quantile(tr_knn, q_knn)

    mask_knn = knn_q <= thr_knn

    # ---------- Mahalanobis gate ----------
    mu = X_ref.mean(axis=0)
    S  = np.cov(X_ref.T, bias=False) + ridge * np.eye(X_ref.shape[1])
    iS = np.linalg.inv(S)

    dif = Xq - mu
    md_q = np.sqrt(np.einsum("ni,ij,nj->n", dif, iS, dif))

    md_tr = np.sqrt(np.einsum("ni,ij,nj->n", X_ref - mu, iS, X_ref - mu))
    thr_md = np.quantile(md_tr, q_md)

    mask_md = md_q <= thr_md

    return mask_knn & mask_md

# =========================
# Uncertainty transform
# =========================

def apply_uncertainty_transform(
    mu: np.ndarray,
    sd: Optional[np.ndarray],
    *,
    transform: Literal["mean", "lcb", "ucb"] = "mean",
    beta: float = 2.0,
) -> np.ndarray:
    """
    Deterministic objective values used for Pareto filtering.
    - mean: mu
    - lcb:  mu - beta*sd
    - ucb:  mu + beta*sd
    """
    mu = np.asarray(mu, dtype=float)
    if sd is None or transform == "mean":
        return mu
    sd = np.asarray(sd, dtype=float)
    if mu.shape != sd.shape:
        raise ValueError(f"mu and sd must have same shape, got {mu.shape} vs {sd.shape}")
    if transform == "lcb":
        return mu - beta * sd
    if transform == "ucb":
        return mu + beta * sd
    raise ValueError("transform must be 'mean'|'lcb'|'ucb'")


# =========================
# Model selection (auto has multiple model sets)
# =========================

def _get_models_from_auto(auto, model_key: ModelKey):
    if model_key == "models":
        models = getattr(auto, "models", None)
    elif model_key == "eval_models":
        models = getattr(auto, "eval_models", None)
    elif model_key == "final_models":
        models = getattr(auto, "final_models", None)
    else:
        raise ValueError("model_key must be 'models'|'eval_models'|'final_models'")
    if models is None:
        raise RuntimeError(f"auto.{model_key} is None. Train/evaluate/finalize first, or pick another model_key.")
    return models


def predict_with_model_key(
    auto,
    X: np.ndarray,
    *,
    model_key: ModelKey = "current",
    batch_size: int = 1000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Thin wrapper around auto.predict with explicit model_key.
    No state mutation.
    """
    mu, sd = auto.predict(
        X,
        batch_size=batch_size,
        model_key=model_key,
    )
    return (
        np.asarray(mu, dtype=float),
        None if sd is None else np.asarray(sd, dtype=float),
    )

# =========================
# BoTorch Pareto front
# =========================

def pareto_front_botorch(
    Y: np.ndarray,
    *,
    sense: Sequence[Sense],
    return_sorted: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute non-dominated set mask / indices using BoTorch.
    BoTorch assumes minimization, so we flip sign for max objectives via `sense`.
    """
    Ymin = _sense_to_minimization(Y, sense=sense)
    Yt = torch.as_tensor(Ymin, dtype=torch.double)
    mask_t = is_non_dominated(-Yt)  # shape (N,)
    idx = torch.where(mask_t)[0].cpu().numpy()
    mask = mask_t.cpu().numpy()

    if return_sorted and idx.size > 0:
        order = np.argsort(Ymin[idx, 0])
        idx = idx[order]
    return idx, mask


# =========================
# Main API
# =========================

@dataclass
class ParetoResult:
    X_candidates: np.ndarray
    Y_mu: np.ndarray
    Y_sd: Optional[np.ndarray]

    # deterministic objective values used for Pareto filtering
    Y_for_pareto: np.ndarray

    idx_pareto: np.ndarray
    X_pareto: np.ndarray
    Y_pareto: np.ndarray  # mu at Pareto indices (original units)
    mask_pareto: np.ndarray

    mask_in_hull: Optional[np.ndarray]
    partitioning: Optional[NondominatedPartitioning]
    meta: Dict[str, object]


def surrogate_pareto_front(
    auto,
    bounds: ArrayLike,
    *,
    # Candidate generation
    n_candidates: int = 4096,
    sampler: Literal["sobol", "uniform"] = "sobol",
    seed: Optional[int] = 0,

    # Which model set to use
    model_key: ModelKey = "current",

    # Objective settings
    sense: Optional[Sequence[Sense]] = None,
    transform: Literal["mean", "lcb", "ucb"] = "mean",
    beta: float = 2.0,

    # Convex hull filter (avoid extrapolation)
    use_hull: bool = False,
    hull_ref: Literal["train", "train_val", "all"] = "train_val",
    hull_tol: float = 1e-12,

    # Calibration hook
    use_calibration: bool = False,
    calibrator: Optional[Calibrator] = None,


    # Optional: build partitioning object (for HV / EHVI workflows)
    build_partitioning: bool = False,
    ref_point: Optional[Sequence[float]] = None,
) -> ParetoResult:
    """
    Sample candidates within `bounds`, optionally filter extrapolation points via convex hull,
    predict using the selected surrogate model set, optionally calibrate, then compute a Pareto front
    using BoTorch `is_non_dominated`.

    Parameters
    ----------
    sense:
      list like ["min","min"] or ["max","min"] (length = M). If None => all "min".
    transform:
      which deterministic vector to use for Pareto filtering: "mean"|"lcb"|"ucb".
      NOTE: we apply transform in ORIGINAL objective direction; then convert to minimization via `sense`.
    build_partitioning/ref_point:
      if build_partitioning=True, we create NondominatedPartitioning on the Pareto set.
      BoTorch expects minimization and uses ref_point in the same minimization space.
      If you provide ref_point in ORIGINAL objectives, we will convert it to minimization using `sense`.
    """
    bounds_np = _as_2d_np(bounds)
    d = bounds_np.shape[0]
    rng = np.random.default_rng(seed)

    # 1) Sample candidates
    if sampler == "sobol":
        Xc = sample_sobol(bounds_np, n=n_candidates, seed=seed)
    elif sampler == "uniform":
        Xc = sample_uniform(bounds_np, n=n_candidates, rng=rng)
    else:
        raise ValueError("sampler must be 'sobol' or 'uniform'")

    mask_in_hull = None

    # 2) Optional: filter by convex hull (avoid extrapolation)
    if use_hull:
        if getattr(auto, "X_all", None) is None or getattr(auto, "split", None) is None:
            raise RuntimeError("use_hull=True requires auto.X_all and auto.split to exist (call fit() first).")

        tr = auto.split.get("tr", None)
        va = auto.split.get("va", None)
        te = auto.split.get("te", None)
        if tr is None or va is None:
            raise RuntimeError("auto.split must contain 'tr' and 'va' indices for hull_ref='train_val'.")

        if hull_ref == "train":
            ref_idx = tr
        elif hull_ref == "train_val":
            ref_idx = np.concatenate([tr, va])
        elif hull_ref == "all":
            if te is None:
                raise RuntimeError("auto.split missing 'te' for hull_ref='all'.")
            ref_idx = np.concatenate([tr, va, te])
        else:
            raise ValueError("hull_ref must be 'train'|'train_val'|'all'")

        X_ref = np.asarray(auto.X_all)[ref_idx]
        mask_in_hull = in_distribution_mask_knn_md(Xc, X_ref, x_min=auto.x_min.cpu().numpy(), x_span=auto.x_span.cpu().numpy())
        Xc = Xc[mask_in_hull]
        if Xc.shape[0] == 0:
            raise RuntimeError(
                "All candidates were filtered out by convex hull. "
                "Try larger bounds, hull_ref='all', or disable use_hull."
            )

    # 3) Predict with selected model set
    mu, sd = predict_with_model_key(auto, Xc, model_key=model_key)

    # 4) Optional calibration
    if use_calibration:

        if calibrator is None:
            raise ValueError("use_calibration=True requires calibrator(mu, sd) callable.")
        Y_for_pareto = calibrator(auto, mu, sd, sense)
    else:
        # 5) Deterministic objective for Pareto filtering
        if sense is None:
            sense = ["min"] * mu.shape[1]
        Y_for_pareto = apply_uncertainty_transform(mu, sd, transform=transform, beta=beta)

    # 6) Pareto via BoTorch
    idx, mask = pareto_front_botorch(Y_for_pareto, sense=sense, return_sorted=True)
    Xp = Xc[idx]
    Yp = mu[idx]  # return mu coordinates for the Pareto front

    # 7) Optional: build partitioning object (for HV / EHVI)
    partitioning = None
    if build_partitioning:
        if ref_point is None:
            raise ValueError("build_partitioning=True requires ref_point.")
        ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
        if ref_point.shape[0] != mu.shape[1]:
            raise ValueError(f"ref_point length must equal M={mu.shape[1]}")
        # Convert ref_point to minimization space
        ref_min = _sense_to_minimization(ref_point[None, :], sense=sense).reshape(-1)
        Ymin_pareto = _sense_to_minimization(Y_for_pareto[idx], sense=sense)
        partitioning = NondominatedPartitioning(
            ref_point=torch.as_tensor(ref_min, dtype=torch.double),
            Y=torch.as_tensor(Ymin_pareto, dtype=torch.double),
        )

    meta = {
        "n_candidates_requested": int(n_candidates),
        "n_candidates_used": int(Xc.shape[0]),
        "sampler": sampler,
        "seed": seed,
        "model_key": model_key,
        "sense": list(sense),
        "transform": transform,
        "beta": float(beta),
        "use_hull": bool(use_hull),
        "hull_ref": hull_ref,
        "hull_tol": float(hull_tol),
        "use_calibration": bool(use_calibration),
        "build_partitioning": bool(build_partitioning),
        "d": int(d),
        "M": int(mu.shape[1]),
        "target_names": getattr(auto, "target_names", None),
    }

    return ParetoResult(
        X_candidates=Xc,
        Y_mu=mu,
        Y_sd=sd,
        Y_for_pareto=Y_for_pareto,
        idx_pareto=idx,
        X_pareto=Xp,
        Y_pareto=Yp,
        mask_pareto=mask,
        mask_in_hull=mask_in_hull,
        partitioning=partitioning,
        meta=meta,
    )


# 你已有这些：
# - _as_2d_np(bounds)
# - sample_sobol / sample_uniform
# - predict_with_model_key(auto, X, model_key=...)
# - apply_uncertainty_transform(mu, sd, transform, beta)
# - pareto_front_botorch(Y, sense, return_sorted=True)
# - in_distribution_mask_knn_md(...)  (你 hull 过滤用的)
# - ParetoResult dataclass

def surrogate_pareto_front_nsga2(
    auto,
    bounds,
    *,
    # NSGA-II settings
    pop_size: int = 128,
    n_gen: int = 150,
    init_sampler: Literal["sobol", "uniform"] = "sobol",
    seed: Optional[int] = 0,

    # Variation operators
    cx_prob: float = 0.9,
    mut_prob: Optional[float] = None,    # default = 1/d
    eta_c: float = 15.0,                 # SBX distribution index
    eta_m: float = 20.0,                 # polynomial mutation distribution index

    # Which model set to use
    model_key: str = "current",

    # Objective settings
    sense: Optional[Sequence[str]] = None,
    transform: Literal["mean", "lcb", "ucb"] = "mean",
    beta: float = 2.0,

    # Convex hull filter (avoid extrapolation)
    use_hull: bool = False,
    hull_ref: Literal["train", "train_val", "all"] = "train_val",
    hull_tol: float = 1e-12,  # 这里只保留接口，实际你用的是 knn mask

    # Calibration hook
    use_calibration: bool = False,
    calibrator=None,   # callable(auto, mu, sd, sense) -> Y_for_pareto

    # Predict batch size
    pred_batch_size: int = 8192,
):
    """
    NSGA-II on surrogate objectives (mean/lcb/ucb or calibrated robust objective).
    Returns ParetoResult compatible with your existing pipeline.
    """

    bounds_np = _as_2d_np(bounds)  # (d,2)
    d = bounds_np.shape[0]
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]

    rng = np.random.default_rng(seed)

    if sense is None:
        # default all min
        # NOTE: your pareto_front_botorch expects sense list
        # and internally flips max -> min. We'll keep consistent.
        # We'll use mu.shape[1] later, but need sense now only if calibrator called.
        pass

    if mut_prob is None:
        mut_prob = 1.0 / d

    # ---------- helpers: SBX crossover & polynomial mutation ----------
    def _sbx_crossover(p1, p2):
        # p1,p2: (d,)
        c1 = p1.copy()
        c2 = p2.copy()
        if rng.random() > cx_prob:
            return c1, c2
        for j in range(d):
            if rng.random() > 0.5:
                continue
            if abs(p1[j] - p2[j]) < 1e-14:
                continue
            x1 = min(p1[j], p2[j])
            x2 = max(p1[j], p2[j])
            rand = rng.random()

            beta = 1.0 + (2.0 * (x1 - lb[j]) / (x2 - x1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + (2.0 * (ub[j] - x2) / (x2 - x1))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

            # assign with random swap
            if rng.random() <= 0.5:
                c1[j] = child1
                c2[j] = child2
            else:
                c1[j] = child2
                c2[j] = child1

        # clip bounds
        c1 = np.clip(c1, lb, ub)
        c2 = np.clip(c2, lb, ub)
        return c1, c2

    def _poly_mutation(x):
        # x: (d,)
        y = x.copy()
        for j in range(d):
            if rng.random() > mut_prob:
                continue
            xj = y[j]
            if ub[j] - lb[j] < 1e-14:
                continue
            delta1 = (xj - lb[j]) / (ub[j] - lb[j])
            delta2 = (ub[j] - xj) / (ub[j] - lb[j])
            rand = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                deltaq = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - (val ** mut_pow)

            y[j] = xj + deltaq * (ub[j] - lb[j])

        y = np.clip(y, lb, ub)
        return y

    # ---------- hull mask helper ----------
    def _mask_in_hull(Xcand):
        if not use_hull:
            return np.ones(Xcand.shape[0], dtype=bool)

        if getattr(auto, "X_all", None) is None or getattr(auto, "split", None) is None:
            raise RuntimeError("use_hull=True requires auto.X_all and auto.split (call fit() first).")

        tr = auto.split.get("tr", None)
        va = auto.split.get("va", None)
        te = auto.split.get("te", None)
        if tr is None or va is None:
            raise RuntimeError("auto.split must contain 'tr' and 'va' for hull_ref='train_val'.")

        if hull_ref == "train":
            ref_idx = tr
        elif hull_ref == "train_val":
            ref_idx = np.concatenate([tr, va])
        elif hull_ref == "all":
            if te is None:
                raise RuntimeError("auto.split missing 'te' for hull_ref='all'.")
            ref_idx = np.concatenate([tr, va, te])
        else:
            raise ValueError("hull_ref must be 'train'|'train_val'|'all'")

        X_ref = np.asarray(auto.X_all)[ref_idx]
        # 你原来是这个 knn-md mask（比真正 convex hull 稳定）
        mask = in_distribution_mask_knn_md(
            Xcand, X_ref,
            x_min=auto.x_min.detach().cpu().numpy(),
            x_span=auto.x_span.detach().cpu().numpy(),
        )
        return mask

    # ---------- evaluation: X -> mu, sd, Y_for_pareto ----------
    def _evaluate(Xcand):
        mu, sd = auto.predict(Xcand, model_key=model_key, batch_size=pred_batch_size)

        if sense is None:
            local_sense = ["min"] * mu.shape[1]
        else:
            local_sense = list(sense)

        if use_calibration:
            if calibrator is None:
                raise ValueError("use_calibration=True requires calibrator(auto, mu, sd, sense).")
            Y_for = calibrator(auto, mu, sd, local_sense)
            Y_for = np.asarray(Y_for, dtype=float)
        else:
            Y_for = apply_uncertainty_transform(mu, sd, transform=transform, beta=beta)
            Y_for = np.asarray(Y_for, dtype=float)

        return np.asarray(mu, float), np.asarray(sd, float), Y_for, local_sense

    # ---------- non-dominated sorting + crowding distance (for selection) ----------
    def _fast_non_dominated_sort(Fmin):
        # Fmin: (N,M) in ORIGINAL objective direction BUT with sense mixed handled by pareto_front_botorch
        # For sorting, easiest: convert to minimization space and do O(N^2) (pop sizes small => OK).
        Ymin = _sense_to_minimization(Fmin, sense=local_sense)  # (N,M)
        N = Ymin.shape[0]
        S = [[] for _ in range(N)]
        n = np.zeros(N, dtype=int)
        rank = np.zeros(N, dtype=int)
        fronts = [[]]

        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                # p dominates q (minimization)
                if np.all(Ymin[p] <= Ymin[q]) and np.any(Ymin[p] < Ymin[q]):
                    S[p].append(q)
                elif np.all(Ymin[q] <= Ymin[p]) and np.any(Ymin[q] < Ymin[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)
        fronts.pop()  # last empty
        return fronts, rank

    def _crowding_distance(Fmin, front):
        # Fmin: (N,M) (we'll use minimization space here)
        if len(front) == 0:
            return np.array([])
        M = Fmin.shape[1]
        dist = np.zeros(len(front), dtype=float)
        idx = np.array(front, dtype=int)

        for m in range(M):
            vals = Fmin[idx, m]
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            vmin = vals[order[0]]
            vmax = vals[order[-1]]
            if vmax - vmin < 1e-14:
                continue
            for i in range(1, len(front) - 1):
                dist[order[i]] += (vals[order[i+1]] - vals[order[i-1]]) / (vmax - vmin)
        return dist

    # ---------- init population ----------
    if init_sampler == "sobol":
        Xpop = sample_sobol(bounds_np, n=pop_size, seed=seed)
    elif init_sampler == "uniform":
        Xpop = sample_uniform(bounds_np, n=pop_size, rng=rng)
    else:
        raise ValueError("init_sampler must be 'sobol' or 'uniform'")

    # apply hull mask (re-sample until enough or give up)
    if use_hull:
        Xkeep = []
        tries = 0
        while len(Xkeep) < pop_size and tries < 20:
            mask = _mask_in_hull(Xpop)
            Xkeep.append(Xpop[mask])
            Xkeep_cat = np.vstack(Xkeep) if len(Xkeep) else np.empty((0, d))
            if Xkeep_cat.shape[0] >= pop_size:
                Xpop = Xkeep_cat[:pop_size]
                break
            # resample more
            need = pop_size - Xkeep_cat.shape[0]
            Xmore = sample_sobol(bounds_np, n=need, seed=(seed + 1000 + tries)) if init_sampler == "sobol" else sample_uniform(bounds_np, n=need, rng=rng)
            Xpop = Xmore
            tries += 1
        if Xpop.shape[0] < pop_size:
            raise RuntimeError("Could not sample enough in-hull initial points. Try use_hull=False or hull_ref='all'.")

    # evaluate initial
    mu_pop, sd_pop, F_pop, local_sense = _evaluate(Xpop)

    # ---------- evolution loop ----------
    for gen in range(n_gen):
        # create offspring
        Xoff = []
        while len(Xoff) < pop_size:
            i, j = rng.integers(0, pop_size, size=2)
            c1, c2 = _sbx_crossover(Xpop[i], Xpop[j])
            c1 = _poly_mutation(c1)
            c2 = _poly_mutation(c2)
            Xoff.append(c1)
            if len(Xoff) < pop_size:
                Xoff.append(c2)
        Xoff = np.asarray(Xoff, dtype=float)

        # hull filter for offspring (reject-and-resample light)
        if use_hull:
            mask = _mask_in_hull(Xoff)
            Xoff = Xoff[mask]
            # if too few survived, refill by random sampling
            while Xoff.shape[0] < pop_size:
                need = pop_size - Xoff.shape[0]
                Xmore = sample_sobol(bounds_np, n=need, seed=(seed + 2000 + gen)) if init_sampler == "sobol" else sample_uniform(bounds_np, n=need, rng=rng)
                m2 = _mask_in_hull(Xmore)
                Xoff = np.vstack([Xoff, Xmore[m2]])
            Xoff = Xoff[:pop_size]

        mu_off, sd_off, F_off, _ = _evaluate(Xoff)

        # combine parent+offspring
        Xmix = np.vstack([Xpop, Xoff])
        mu_mix = np.vstack([mu_pop, mu_off])
        sd_mix = np.vstack([sd_pop, sd_off])
        F_mix  = np.vstack([F_pop,  F_off])

        # non-dominated sorting on F_mix (in original direction; helper uses local_sense)
        fronts, rank = _fast_non_dominated_sort(F_mix)

        # select next generation using crowding distance
        next_idx = []
        # precompute minimization space for crowding
        Fmin_mix = _sense_to_minimization(F_mix, sense=local_sense)

        for front in fronts:
            if len(next_idx) + len(front) <= pop_size:
                next_idx.extend(front)
            else:
                cd = _crowding_distance(Fmin_mix, front)
                order = np.argsort(-cd)  # descending
                remain = pop_size - len(next_idx)
                next_idx.extend(list(np.array(front)[order[:remain]]))
                break

        next_idx = np.asarray(next_idx, dtype=int)

        Xpop = Xmix[next_idx]
        mu_pop = mu_mix[next_idx]
        sd_pop = sd_mix[next_idx]
        F_pop  = F_mix[next_idx]

    # ---------- final Pareto extraction ----------
    idx_pf, mask_pf = pareto_front_botorch(F_pop, sense=local_sense, return_sorted=True)

    Xp = Xpop[idx_pf]
    Yp_mu = mu_pop[idx_pf]  # return mu coords (consistent with your original)
    # NOTE: return also the deterministic vector used for Pareto filtering (F_pop)
    Y_for_pf = F_pop[idx_pf]

    meta = {
        "optimizer": "nsga2",
        "pop_size": int(pop_size),
        "n_gen": int(n_gen),
        "seed": seed,
        "model_key": model_key,
        "sense": list(local_sense),
        "transform": transform,
        "beta": float(beta),
        "use_hull": bool(use_hull),
        "hull_ref": hull_ref,
        "use_calibration": bool(use_calibration),
        "d": int(d),
        "M": int(mu_pop.shape[1]),
    }

    return ParetoResult(
        X_candidates=Xpop,
        Y_mu=mu_pop,
        Y_sd=sd_pop,
        Y_for_pareto=F_pop,
        idx_pareto=idx_pf,
        X_pareto=Xp,
        Y_pareto=Yp_mu,
        mask_pareto=mask_pf,
        mask_in_hull=None,   # 这里不再返回 candidate mask（NSGA-II内部在用）
        partitioning=None,
        meta=meta,
    )



def calibrate_from_auto_test(
    auto,
    alpha: float = 0.10,
    method: str = "per_target",   # "per_target" or "joint"
    eps: float = 1e-12,
):
    """
    Use auto's internal TEST split to perform conformal calibration.
    Calibration results are stored in auto.calibration.
    """

    # ---- 1. 取 test split（根据你 auto 的实现，选一种即可） ----
    
    if hasattr(auto, "split"):
        idx = auto.split["te"]
        X_cal = auto.X_all[idx]
        Y_cal = auto.Y_all[idx]
    else:
        raise RuntimeError("auto does not expose test split.")

    # ---- 2. 确保模型在 eval 状态 ----
    if hasattr(auto, "eval_models"):
        old_models = auto.models
        auto.models = auto.eval_models
        # auto.models.eval()

    # ---- 3. 预测 ----
    with torch.inference_mode():
        MU, SIG = auto.predict(X_cal)

    auto.models = old_models  

    MU = torch.as_tensor(MU, dtype=torch.double)
    SIG = torch.as_tensor(SIG, dtype=torch.double)
    Y   = torch.as_tensor(Y_cal, dtype=torch.double)

    N, M = Y.shape
    Z = torch.abs(Y - MU) / (SIG + eps)

    # ---- 4. 算 conformal q ----
    method = method.lower()
    if method == "per_target":
        alpha_j = alpha / M
        q_vec = torch.quantile(Z, 1.0 - alpha_j, dim=0).cpu().numpy()
    elif method == "joint":
        alpha_j = alpha
        s = torch.max(Z, dim=1).values
        q = float(torch.quantile(s, 1.0 - alpha).cpu().numpy())
        q_vec = np.full(M, q)
    else:
        raise ValueError("method must be 'per_target' or 'joint'")

    # ---- 5. 存回 auto ----
    auto.calibration = {
        "enabled": True,
        "method": method,
        "alpha": alpha,
        "alpha_j": alpha_j,
        "q_vec": q_vec,
        "source": "auto.test",
    }

    return auto.calibration


def calibrate_for_auto(auto, mu, std, sense):
    """
    Apply conformal calibration to compute robust objective values.

    Parameters
    ----------
    auto : object
        Must have auto.calibration['q_vec']
    mu : array-like or torch.Tensor, shape [N, M]
        Predictive mean
    std : array-like or torch.Tensor, shape [N, M]
        Predictive std (uncalibrated)
    sense : list[str] of length M
        Each entry is 'min' or 'max'

    Returns
    -------
    ROB : torch.Tensor, shape [N, M]
        Robust objective values after calibration
    """

    mu  = torch.as_tensor(mu,  dtype=torch.double)
    std = torch.as_tensor(std, dtype=torch.double)

    calib = getattr(auto, "calibration", None)
    if calib is None or not calib.get("enabled", False):
        raise RuntimeError("Calibration not found in auto.calibration")

    q = torch.as_tensor(calib["q_vec"], dtype=torch.double)  # [M]

    if mu.shape != std.shape:
        raise ValueError("mu and std must have the same shape")

    if mu.shape[1] != len(sense):
        raise ValueError("Length of sense must match number of objectives")

    # broadcast q: [M] -> [N, M]
    q_std = std * q.unsqueeze(0)

    ROB = torch.empty_like(mu)

    for j, s in enumerate(sense):
        s = s.lower()
        if s == "min":
            # worst-case (upper confidence bound)
            ROB[:, j] = mu[:, j] + q_std[:, j]
        elif s == "max":
            # worst-case (lower confidence bound)
            ROB[:, j] = mu[:, j] - q_std[:, j]
        else:
            raise ValueError(f"sense[{j}] must be 'min' or 'max', got '{sense[j]}'")

    return ROB

def calibrate_from_cv_plus(
    auto,                   # factory: make_auto() -> fresh auto
    X: np.ndarray,
    Y: np.ndarray,
    *,
    K: int = 5,
    alpha: float = 0.10,
    method: str = "per_target",   # "per_target" | "joint"
    eps: float = 1e-12,
    model_key: str = "final_models",   # which models inside each fold-auto to use
    seed: int = 0,
    batch_size: int = 4096,
):
    """
    CV+ / cross-fitted conformal calibration.

    Returns a calibration dict compatible with auto.calibration.
    """

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X must be (N,d), Y must be (N,M)")

    N, M = Y.shape
    if N < K:
        raise ValueError(f"N={N} must be >= K={K}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    folds = np.array_split(perm, K)

    Z_all = []   # standardized residuals from all folds

    for k in range(K):
        idx_te = folds[k]
        idx_tr = np.concatenate([folds[i] for i in range(K) if i != k])

        auto_k = auto
        # auto_k.fit(X[idx_tr], Y[idx_tr])
        # auto_k.eval()
        # auto_k.finalize(
        #     X[idx_tr], Y[idx_tr], refit_scope="full", scaler_mode = "refit"
        # )
        auto_k.final_models = auto_k._train_models_on(X[idx_tr], Y[idx_tr], scaler_mode="frozen")

        with torch.inference_mode():
            MU, SIG = auto_k.predict(
                X[idx_te],
                model_key=model_key,
                batch_size=batch_size,
            )

        MU = np.asarray(MU, float)
        SIG = np.asarray(SIG, float)
        Yk  = Y[idx_te]

        Zk = np.abs(Yk - MU) / (SIG + eps)   # (n_k, M)
        Z_all.append(Zk)

    Z = np.vstack(Z_all)   # (N, M)  ← out-of-fold residuals

    # ---- conformal quantile ----
    method = method.lower()
    if method == "per_target":
        alpha_j = alpha / M
        q_vec = np.quantile(Z, 1.0 - alpha_j, axis=0)
    elif method == "joint":
        alpha_j = alpha
        s = np.max(Z, axis=1)   # (N,)
        q = float(np.quantile(s, 1.0 - alpha))
        q_vec = np.full(M, q, dtype=float)
    else:
        raise ValueError("method must be 'per_target' or 'joint'")

    return {
        "enabled": True,
        "method": method,
        "alpha": float(alpha),
        "alpha_j": float(alpha_j),
        "q_vec": np.asarray(q_vec, float),
        "source": "cv+",
        "K": int(K),
        "n_cal": int(N),
        "seed": int(seed),
        "model_key": model_key,
    }
