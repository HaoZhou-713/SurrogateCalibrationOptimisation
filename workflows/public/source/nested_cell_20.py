import gc
import json
import random
import time
import traceback
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


# ============================================================
# General helpers
# ============================================================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _finite_sample_conformal_quantile(
    values: np.ndarray,
    alpha: float,
    axis=None,
):
    """
    Finite-sample conformal quantile:

        ceil((n + 1)(1 - alpha)) / n

    using the 'higher' quantile rule.
    """

    values = np.asarray(values, dtype=float)

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if axis is None:
        n = values.size
    else:
        n = values.shape[axis]

    if n < 1:
        raise ValueError("Cannot calculate a quantile from empty values.")

    quantile_level = min(
        np.ceil((n + 1) * (1.0 - alpha)) / n,
        1.0,
    )

    try:
        return np.quantile(
            values,
            quantile_level,
            axis=axis,
            method="higher",
        )

    except TypeError:
        # Compatibility with older NumPy
        return np.quantile(
            values,
            quantile_level,
            axis=axis,
            interpolation="higher",
        )


def _gaussian_multiplier(alpha: float) -> float:
    """
    Two-sided Gaussian interval multiplier.

    For alpha=0.10, q approximately equals 1.645.
    """

    return float(
        NormalDist().inv_cdf(
            1.0 - alpha / 2.0
        )
    )


def _serialise_kernel(kernel: Any) -> str:
    try:
        return json.dumps(kernel)
    except TypeError:
        return str(kernel)


# ============================================================
# Model Selection + Bagging model construction
# ============================================================

def _make_model_selection_bagging_auto(
    *,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    seed: int,
    bags: int,
    kernel_grid: Sequence[
        Tuple[str, Optional[float]]
    ],
    icm_rank: int,
    noise_floor: float,
    init_noise: float,
    val_frac: float,
    obj_weights: Optional[np.ndarray],
    device: Any,
    dtype: Any,
) -> Any:
    """
    Create a fresh AutoMultiOutputGP configured for:

        Model Selection + Bagging
    """

    config = AutoConfig(
        kernel_grid=list(kernel_grid),
        icm_rank=icm_rank,
        bags=bags,
        noise_floor=noise_floor,
        init_noise=init_noise,
        seed=seed,
        val_frac=val_frac,
        test_frac=0.10,
        use_ks_for_test=False,
        device=device,
        dtype=dtype,
        obj_weights=obj_weights,
        use_model_selection=True,

        # These values are not used when model selection is enabled,
        # but are still supplied for config completeness.
        fixed_family="IND",
        fixed_kernel=("Matern", 2.5),
    )

    return AutoMultiOutputGP(
        feature_names=list(feature_names),
        target_names=list(target_names),
        config=config,
    )


def _fit_model_selection_bagging(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    seed: int,
    bags: int,
    kernel_grid: Sequence[
        Tuple[str, Optional[float]]
    ],
    icm_rank: int,
    noise_floor: float,
    init_noise: float,
    val_frac: float,
    obj_weights: Optional[np.ndarray],
    device: Any,
    dtype: Any,
    verbose: bool = False,
):
    """
    Fit one fresh Model Selection + Bagging surrogate.

    AutoMultiOutputGP.fit() internally divides X_train into train and
    validation subsets, performs model selection, and retrains the selected
    family/kernel on train + validation.
    """

    _set_seed(seed)

    auto = _make_model_selection_bagging_auto(
        feature_names=feature_names,
        target_names=target_names,
        seed=seed,
        bags=bags,
        kernel_grid=kernel_grid,
        icm_rank=icm_rank,
        noise_floor=noise_floor,
        init_noise=init_noise,
        val_frac=val_frac,
        obj_weights=obj_weights,
        device=device,
        dtype=dtype,
    )

    # test_frac must remain positive for the current AutoMultiOutputGP.fit()
    # implementation. The internal test subset is not used for final outer
    # evaluation. The final model is trained on its internal train+validation.
    auto.fit(
        X_train,
        Y_train,
        test_frac=0.10,
        val_frac=val_frac,
        verbose=verbose,
    )

    if not getattr(auto, "_is_fitted", False):
        raise RuntimeError(
            "Model training completed, but auto._is_fitted is False."
        )

    return auto


# ============================================================
# Calibration score construction
# ============================================================

def _calibration_from_standardized_residuals(
    standardized_residuals: np.ndarray,
    *,
    alpha: float,
    conformal_mode: str,
) -> Dict[str, Any]:
    """
    Construct q_vec from standardized absolute residuals.

    conformal_mode
    ----------------
    per_target:
        Each objective separately targets marginal 1-alpha coverage.

    bonferroni:
        Each objective uses alpha / M to approximate simultaneous coverage.

    joint:
        Uses max standardized residual across objectives and one common q.
    """

    z = np.asarray(
        standardized_residuals,
        dtype=float,
    )

    if z.ndim != 2:
        raise ValueError(
            "standardized_residuals must have shape (N, M)."
        )

    n_cal, n_targets = z.shape

    if n_cal < 1:
        raise ValueError("Calibration residual array is empty.")

    conformal_mode = conformal_mode.lower()

    if conformal_mode == "per_target":
        alpha_j = float(alpha)

        q_vec = _finite_sample_conformal_quantile(
            z,
            alpha=alpha_j,
            axis=0,
        )

    elif conformal_mode == "bonferroni":
        alpha_j = float(alpha / n_targets)

        q_vec = _finite_sample_conformal_quantile(
            z,
            alpha=alpha_j,
            axis=0,
        )

    elif conformal_mode == "joint":
        alpha_j = float(alpha)

        max_scores = np.max(
            z,
            axis=1,
        )

        q = float(
            _finite_sample_conformal_quantile(
                max_scores,
                alpha=alpha,
                axis=None,
            )
        )

        q_vec = np.full(
            n_targets,
            q,
            dtype=float,
        )

    else:
        raise ValueError(
            "conformal_mode must be "
            "'per_target', 'bonferroni', or 'joint'."
        )

    return {
        "enabled": True,
        "method": conformal_mode,
        "alpha": float(alpha),
        "alpha_j": float(alpha_j),
        "q_vec": np.asarray(q_vec, dtype=float),
        "n_cal": int(n_cal),
    }


def _uncalibrated_gaussian_calibration(
    n_targets: int,
    alpha: float,
) -> Dict[str, Any]:
    """
    Nominal Gaussian interval without conformal recalibration.
    """

    q = _gaussian_multiplier(alpha)

    return {
        "enabled": False,
        "method": "uncalibrated",
        "alpha": float(alpha),
        "alpha_j": float(alpha),
        "q_vec": np.full(
            n_targets,
            q,
            dtype=float,
        ),
        "n_cal": 0,
    }


# ============================================================
# Holdout calibration
# ============================================================

def _fit_holdout_calibrated_model(
    X_outer_train: np.ndarray,
    Y_outer_train: np.ndarray,
    *,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    seed: int,
    alpha: float,
    calibration_frac: float,
    conformal_mode: str,
    bags: int,
    kernel_grid: Sequence[
        Tuple[str, Optional[float]]
    ],
    icm_rank: int,
    noise_floor: float,
    init_noise: float,
    val_frac: float,
    obj_weights: Optional[np.ndarray],
    device: Any,
    dtype: Any,
    batch_size: int,
    verbose: bool = False,
):
    """
    Single train/calibration split.

    The model is trained on the fit subset.
    Calibration scores are calculated on the held-out calibration subset.
    """

    n_samples = X_outer_train.shape[0]
    all_indices = np.arange(n_samples)

    fit_indices, calibration_indices = train_test_split(
        all_indices,
        test_size=calibration_frac,
        random_state=seed,
        shuffle=True,
    )

    auto = _fit_model_selection_bagging(
        X_outer_train[fit_indices],
        Y_outer_train[fit_indices],
        feature_names=feature_names,
        target_names=target_names,
        seed=seed,
        bags=bags,
        kernel_grid=kernel_grid,
        icm_rank=icm_rank,
        noise_floor=noise_floor,
        init_noise=init_noise,
        val_frac=val_frac,
        obj_weights=obj_weights,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    with torch.inference_mode():
        mu_cal, std_cal = auto.predict(
            X_outer_train[calibration_indices],
            batch_size=batch_size,
        )

    mu_cal = _to_numpy(mu_cal).astype(float)
    std_cal = _to_numpy(std_cal).astype(float)

    standardized_residuals = (
        np.abs(
            Y_outer_train[calibration_indices]
            - mu_cal
        )
        / np.maximum(std_cal, 1e-12)
    )

    calibration = (
        _calibration_from_standardized_residuals(
            standardized_residuals,
            alpha=alpha,
            conformal_mode=conformal_mode,
        )
    )

    calibration.update(
        {
            "source": "holdout",
            "seed": int(seed),
            "n_fit": int(len(fit_indices)),
            "n_cal": int(len(calibration_indices)),
            "calibration_frac": float(calibration_frac),
        }
    )

    return auto, calibration


# ============================================================
# Cross-fitted calibration
# ============================================================

def _cross_fitted_calibration(
    X_outer_train: np.ndarray,
    Y_outer_train: np.ndarray,
    *,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    seed: int,
    alpha: float,
    K: int,
    conformal_mode: str,
    bags: int,
    kernel_grid: Sequence[
        Tuple[str, Optional[float]]
    ],
    icm_rank: int,
    noise_floor: float,
    init_noise: float,
    val_frac: float,
    obj_weights: Optional[np.ndarray],
    device: Any,
    dtype: Any,
    batch_size: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    K-fold cross-fitted conformal calibration.

    Every outer-training observation receives one out-of-fold prediction.
    Every fold independently performs Model Selection + Bagging.
    """

    X_outer_train = np.asarray(
        X_outer_train,
        dtype=float,
    )

    Y_outer_train = np.asarray(
        Y_outer_train,
        dtype=float,
    )

    n_samples, n_targets = Y_outer_train.shape

    if K < 2:
        raise ValueError("K must be at least 2.")

    if n_samples < K:
        raise ValueError(
            f"Number of training samples {n_samples} is smaller than K={K}."
        )

    rng = np.random.default_rng(seed)

    shuffled_indices = rng.permutation(
        n_samples
    )

    folds = np.array_split(
        shuffled_indices,
        K,
    )

    standardized_residuals = []

    fold_models = []

    for fold_id in range(K):
        fold_validation_indices = folds[fold_id]

        fold_training_indices = np.concatenate(
            [
                folds[j]
                for j in range(K)
                if j != fold_id
            ]
        )

        fold_seed = (
            seed * 1000
            + fold_id
            + 1
        )

        if verbose:
            print(
                f"  Cross-fitting fold {fold_id + 1}/{K}: "
                f"train={len(fold_training_indices)}, "
                f"validation={len(fold_validation_indices)}"
            )

        auto_fold = _fit_model_selection_bagging(
            X_outer_train[fold_training_indices],
            Y_outer_train[fold_training_indices],
            feature_names=feature_names,
            target_names=target_names,
            seed=fold_seed,
            bags=bags,
            kernel_grid=kernel_grid,
            icm_rank=icm_rank,
            noise_floor=noise_floor,
            init_noise=init_noise,
            val_frac=val_frac,
            obj_weights=obj_weights,
            device=device,
            dtype=dtype,
            verbose=False,
        )

        with torch.inference_mode():
            mu_fold, std_fold = auto_fold.predict(
                X_outer_train[
                    fold_validation_indices
                ],
                batch_size=batch_size,
            )

        mu_fold = _to_numpy(mu_fold).astype(float)
        std_fold = _to_numpy(std_fold).astype(float)

        y_fold = Y_outer_train[
            fold_validation_indices
        ]

        z_fold = (
            np.abs(y_fold - mu_fold)
            / np.maximum(std_fold, 1e-12)
        )

        standardized_residuals.append(z_fold)

        model_info = auto_fold.info()

        fold_models.append(
            {
                "fold": int(fold_id),
                "family": model_info["family"],
                "kernel": _serialise_kernel(
                    model_info["kernel"]
                ),
            }
        )

        del auto_fold
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    z_all = np.vstack(
        standardized_residuals
    )

    calibration = (
        _calibration_from_standardized_residuals(
            z_all,
            alpha=alpha,
            conformal_mode=conformal_mode,
        )
    )

    calibration.update(
        {
            "source": "cross_fitted",
            "seed": int(seed),
            "K": int(K),
            "n_cal": int(n_samples),
            "fold_models": fold_models,
        }
    )

    return calibration


# ============================================================
# Prediction intervals
# ============================================================

def _predict_interval(
    auto,
    X_test: np.ndarray,
    calibration: Dict[str, Any],
    *,
    batch_size: int,
):
    with torch.inference_mode():
        y_mean, y_std = auto.predict(
            X_test,
            batch_size=batch_size,
        )

    y_mean = _to_numpy(y_mean).astype(float)
    y_std = _to_numpy(y_std).astype(float)

    y_std = np.maximum(
        y_std,
        1e-12,
    )

    q_vec = np.asarray(
        calibration["q_vec"],
        dtype=float,
    ).reshape(1, -1)

    half_width = q_vec * y_std

    lower = y_mean - half_width
    upper = y_mean + half_width

    return y_mean, y_std, lower, upper


# ============================================================
# Interval-quality metrics
# ============================================================

def _evaluate_prediction_intervals(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_names: Sequence[str],
    alpha: float,
    normalization_scale: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate coverage, MPIW and interval score.

    Raw and normalized interval widths/scores are both reported.
    """

    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if (
        y_true.shape != lower.shape
        or y_true.shape != upper.shape
    ):
        raise ValueError(
            "y_true, lower and upper must have identical shapes."
        )

    n_samples, n_targets = y_true.shape

    if len(target_names) != n_targets:
        raise ValueError(
            "target_names does not match the number of objectives."
        )

    normalization_scale = np.asarray(
        normalization_scale,
        dtype=float,
    ).reshape(1, -1)

    normalization_scale = np.maximum(
        normalization_scale,
        1e-12,
    )

    covered = (
        (y_true >= lower)
        & (y_true <= upper)
    )

    width = upper - lower

    lower_miss = (
        y_true < lower
    )

    upper_miss = (
        y_true > upper
    )

    interval_score = (
        width
        + (2.0 / alpha)
        * (lower - y_true)
        * lower_miss
        + (2.0 / alpha)
        * (y_true - upper)
        * upper_miss
    )

    normalized_width = (
        width / normalization_scale
    )

    normalized_interval_score = (
        interval_score
        / normalization_scale
    )

    metrics: Dict[str, float] = {}

    target_coverages = []
    target_coverage_errors = []
    target_mpiw = []
    target_normalized_mpiw = []
    target_interval_scores = []
    target_normalized_interval_scores = []

    nominal_coverage = 1.0 - alpha

    for j, target_name in enumerate(
        target_names
    ):
        coverage_j = float(
            np.mean(covered[:, j])
        )

        coverage_error_j = float(
            abs(
                coverage_j
                - nominal_coverage
            )
        )

        mpiw_j = float(
            np.mean(width[:, j])
        )

        normalized_mpiw_j = float(
            np.mean(
                normalized_width[:, j]
            )
        )

        interval_score_j = float(
            np.mean(
                interval_score[:, j]
            )
        )

        normalized_interval_score_j = float(
            np.mean(
                normalized_interval_score[:, j]
            )
        )

        metrics[
            f"{target_name}_coverage"
        ] = coverage_j

        metrics[
            f"{target_name}_coverage_error"
        ] = coverage_error_j

        metrics[
            f"{target_name}_MPIW"
        ] = mpiw_j

        metrics[
            f"{target_name}_normalized_MPIW"
        ] = normalized_mpiw_j

        metrics[
            f"{target_name}_interval_score"
        ] = interval_score_j

        metrics[
            f"{target_name}_normalized_interval_score"
        ] = normalized_interval_score_j

        target_coverages.append(
            coverage_j
        )

        target_coverage_errors.append(
            coverage_error_j
        )

        target_mpiw.append(
            mpiw_j
        )

        target_normalized_mpiw.append(
            normalized_mpiw_j
        )

        target_interval_scores.append(
            interval_score_j
        )

        target_normalized_interval_scores.append(
            normalized_interval_score_j
        )

    metrics["macro_coverage"] = float(
        np.mean(target_coverages)
    )

    metrics["macro_coverage_error"] = float(
        np.mean(target_coverage_errors)
    )

    metrics["macro_MPIW"] = float(
        np.mean(target_mpiw)
    )

    metrics["macro_normalized_MPIW"] = float(
        np.mean(
            target_normalized_mpiw
        )
    )

    metrics["macro_interval_score"] = float(
        np.mean(
            target_interval_scores
        )
    )

    metrics[
        "macro_normalized_interval_score"
    ] = float(
        np.mean(
            target_normalized_interval_scores
        )
    )

    metrics[
        "simultaneous_joint_coverage"
    ] = float(
        np.mean(
            np.all(covered, axis=1)
        )
    )

    metrics["nominal_coverage"] = float(
        nominal_coverage
    )

    metrics["n_test"] = int(
        n_samples
    )

    return metrics


# ============================================================
# One calibration method at multiple nominal levels
# ============================================================

def _evaluate_reliability_for_method(
    *,
    calibration_method: str,
    final_auto,
    X_outer_train: np.ndarray,
    Y_outer_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    seed: int,
    nominal_coverages: Sequence[float],
    K: int,
    calibration_frac: float,
    conformal_mode: str,
    bags: int,
    kernel_grid: Sequence[
        Tuple[str, Optional[float]]
    ],
    icm_rank: int,
    noise_floor: float,
    init_noise: float,
    val_frac: float,
    obj_weights: Optional[np.ndarray],
    device: Any,
    dtype: Any,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generate reliability-curve data for one seed and one method.
    """

    records = []

    for nominal_coverage in nominal_coverages:
        alpha = (
            1.0
            - float(nominal_coverage)
        )

        temporary_auto = None

        if calibration_method == "uncalibrated":
            calibration = (
                _uncalibrated_gaussian_calibration(
                    n_targets=Y_test.shape[1],
                    alpha=alpha,
                )
            )

            prediction_auto = final_auto

        elif calibration_method == "holdout":
            (
                temporary_auto,
                calibration,
            ) = _fit_holdout_calibrated_model(
                X_outer_train,
                Y_outer_train,
                feature_names=feature_names,
                target_names=target_names,
                seed=seed,
                alpha=alpha,
                calibration_frac=calibration_frac,
                conformal_mode=conformal_mode,
                bags=bags,
                kernel_grid=kernel_grid,
                icm_rank=icm_rank,
                noise_floor=noise_floor,
                init_noise=init_noise,
                val_frac=val_frac,
                obj_weights=obj_weights,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                verbose=False,
            )

            prediction_auto = temporary_auto

        elif calibration_method == "cross_fitted":
            calibration = _cross_fitted_calibration(
                X_outer_train,
                Y_outer_train,
                feature_names=feature_names,
                target_names=target_names,
                seed=seed,
                alpha=alpha,
                K=K,
                conformal_mode=conformal_mode,
                bags=bags,
                kernel_grid=kernel_grid,
                icm_rank=icm_rank,
                noise_floor=noise_floor,
                init_noise=init_noise,
                val_frac=val_frac,
                obj_weights=obj_weights,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                verbose=False,
            )

            prediction_auto = final_auto

        else:
            raise ValueError(
                f"Unknown calibration method: "
                f"{calibration_method}"
            )

        _, _, lower, upper = _predict_interval(
            prediction_auto,
            X_test,
            calibration,
            batch_size=batch_size,
        )

        covered = (
            (Y_test >= lower)
            & (Y_test <= upper)
        )

        for target_id, target_name in enumerate(
            target_names
        ):
            empirical_coverage = float(
                np.mean(
                    covered[:, target_id]
                )
            )

            records.append(
                {
                    "seed": int(seed),
                    "calibration_method": (
                        calibration_method
                    ),
                    "target": target_name,
                    "nominal_coverage": float(
                        nominal_coverage
                    ),
                    "empirical_coverage": (
                        empirical_coverage
                    ),
                    "absolute_calibration_error": (
                        abs(
                            empirical_coverage
                            - nominal_coverage
                        )
                    ),
                }
            )

        if temporary_auto is not None:
            del temporary_auto
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    reliability_df = pd.DataFrame(
        records
    )

    target_ece = (
        reliability_df
        .groupby("target")[
            "absolute_calibration_error"
        ]
        .mean()
    )

    ece_metrics = {
        f"{target}_ECE": float(value)
        for target, value
        in target_ece.items()
    }

    ece_metrics["macro_ECE"] = float(
        target_ece.mean()
    )

    return reliability_df, ece_metrics


# ============================================================
# Main function
# ============================================================

def run_model_selection_bagging_calibration(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    *,
    output_dir: str = (
        "model_selection_bagging_calibration"
    ),
    seeds: Sequence[int] = tuple(range(20)),
    outer_test_frac: float = 0.20,
    alpha: float = 0.10,
    K: int = 5,
    calibration_frac: float = 0.20,
    conformal_mode: str = "per_target",
    bags: int = 5,
    kernel_grid: Optional[
        Sequence[
            Tuple[str, Optional[float]]
        ]
    ] = None,
    icm_rank: int = 2,
    noise_floor: float = 1e-4,
    init_noise: float = 1e-3,
    model_selection_val_frac: float = 0.12,
    obj_weights: Optional[np.ndarray] = None,
    nominal_coverages: Sequence[float] = (
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ),
    batch_size: int = 4096,
    device: Any = torch.device("cpu"),
    dtype: Any = torch.double,
    save_predictions: bool = True,
    verbose: bool = True,
):
    """
    Run a complete 20-seed calibration benchmark using only:

        Model Selection + Bagging

    Calibration methods compared:
        1. uncalibrated Gaussian intervals
        2. holdout conformal calibration
        3. K-fold cross-fitted conformal calibration

    Parameters
    ----------
    X
        Input data with shape (N, D).

    Y
        Multi-objective outputs with shape (N, M).

    feature_names
        Names of the D input variables.

    target_names
        Names of the M objectives.

    outer_test_frac
        Fraction of the complete dataset reserved for final evaluation.

    alpha
        Miscoverage level for the main interval.
        alpha=0.10 means nominal 90% coverage.

    K
        Number of folds for cross-fitted calibration.

    calibration_frac
        Fraction of outer-training data used for holdout calibration.

    conformal_mode
        "per_target", "bonferroni", or "joint".

    bags
        Number of bootstrap models in addition to the full-data base model.
        Therefore, total ensemble size is 1 + bags.

    Returns
    -------
    metrics_by_seed
        Main calibration metrics for each seed and method.

    calibration_summary
        Aggregate statistics over seeds.

    reliability_data
        Reliability-diagram data for every nominal coverage level.

    reliability_summary
        Mean and standard deviation of empirical coverage over seeds.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (N,D), received {X.shape}."
        )

    if Y.ndim == 1:
        Y = Y[:, None]

    if Y.ndim != 2:
        raise ValueError(
            f"Y must have shape (N,M), received {Y.shape}."
        )

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of samples."
        )

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "feature_names does not match X.shape[1]."
        )

    if len(target_names) != Y.shape[1]:
        raise ValueError(
            "target_names does not match Y.shape[1]."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Y contains NaN or infinite values."
        )

    if bags <= 0:
        raise ValueError(
            "bags must be greater than zero."
        )

    if kernel_grid is None:
        kernel_grid = [
            ("RBF", None),
            ("Matern", 0.5),
            ("Matern", 1.5),
            ("Matern", 2.5),
        ]

    kernel_grid = list(kernel_grid)
    feature_names = list(feature_names)
    target_names = list(target_names)
    seeds = [int(seed) for seed in seeds]

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        output_path / "predictions"
    )

    reliability_figure_path = (
        output_path / "reliability_diagrams"
    )

    if save_predictions:
        prediction_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    reliability_figure_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    methods = [
        "uncalibrated",
        "holdout",
        "cross_fitted",
    ]

    metric_records = []
    reliability_records = []
    selected_model_records = []
    failure_records = []

    for seed in seeds:
        if verbose:
            print("\n" + "=" * 80)
            print(f"Outer seed {seed}")
            print("=" * 80)

        _set_seed(seed)

        all_indices = np.arange(
            X.shape[0]
        )

        (
            outer_train_indices,
            outer_test_indices,
        ) = train_test_split(
            all_indices,
            test_size=outer_test_frac,
            random_state=seed,
            shuffle=True,
        )

        X_outer_train = X[
            outer_train_indices
        ]

        Y_outer_train = Y[
            outer_train_indices
        ]

        X_test = X[
            outer_test_indices
        ]

        Y_test = Y[
            outer_test_indices
        ]

        # Used to normalize widths and interval scores.
        normalization_scale = np.std(
            Y_outer_train,
            axis=0,
            ddof=0,
        )

        normalization_scale = np.maximum(
            normalization_scale,
            1e-12,
        )

        final_auto = None

        try:
            # Final model used by uncalibrated and cross-fitted methods.
            final_auto = _fit_model_selection_bagging(
                X_outer_train,
                Y_outer_train,
                feature_names=feature_names,
                target_names=target_names,
                seed=seed,
                bags=bags,
                kernel_grid=kernel_grid,
                icm_rank=icm_rank,
                noise_floor=noise_floor,
                init_noise=init_noise,
                val_frac=model_selection_val_frac,
                obj_weights=obj_weights,
                device=device,
                dtype=dtype,
                verbose=False,
            )

            final_info = final_auto.info()

            selected_model_records.append(
                {
                    "seed": seed,
                    "model_role": "final_outer_model",
                    "selected_family": (
                        final_info["family"]
                    ),
                    "selected_kernel": (
                        _serialise_kernel(
                            final_info["kernel"]
                        )
                    ),
                    "bags": bags,
                    "total_ensemble_members": (
                        1 + bags
                    ),
                }
            )

        except Exception as exc:
            failure_records.append(
                {
                    "seed": seed,
                    "calibration_method": (
                        "final_outer_model"
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_message": str(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )

            print(
                f"FAILED final model: seed={seed}, "
                f"{type(exc).__name__}: {exc}"
            )

            continue

        for calibration_method in methods:
            prediction_auto = None

            try:
                start_time = time.perf_counter()

                if calibration_method == "uncalibrated":
                    calibration = (
                        _uncalibrated_gaussian_calibration(
                            n_targets=Y.shape[1],
                            alpha=alpha,
                        )
                    )

                    prediction_auto = final_auto

                elif calibration_method == "holdout":
                    (
                        prediction_auto,
                        calibration,
                    ) = _fit_holdout_calibrated_model(
                        X_outer_train,
                        Y_outer_train,
                        feature_names=feature_names,
                        target_names=target_names,
                        seed=seed,
                        alpha=alpha,
                        calibration_frac=(
                            calibration_frac
                        ),
                        conformal_mode=(
                            conformal_mode
                        ),
                        bags=bags,
                        kernel_grid=kernel_grid,
                        icm_rank=icm_rank,
                        noise_floor=noise_floor,
                        init_noise=init_noise,
                        val_frac=(
                            model_selection_val_frac
                        ),
                        obj_weights=obj_weights,
                        device=device,
                        dtype=dtype,
                        batch_size=batch_size,
                        verbose=False,
                    )

                elif calibration_method == "cross_fitted":
                    calibration = (
                        _cross_fitted_calibration(
                            X_outer_train,
                            Y_outer_train,
                            feature_names=feature_names,
                            target_names=target_names,
                            seed=seed,
                            alpha=alpha,
                            K=K,
                            conformal_mode=(
                                conformal_mode
                            ),
                            bags=bags,
                            kernel_grid=kernel_grid,
                            icm_rank=icm_rank,
                            noise_floor=noise_floor,
                            init_noise=init_noise,
                            val_frac=(
                                model_selection_val_frac
                            ),
                            obj_weights=obj_weights,
                            device=device,
                            dtype=dtype,
                            batch_size=batch_size,
                            verbose=False,
                        )
                    )

                    prediction_auto = final_auto

                else:
                    raise RuntimeError(
                        f"Unknown method "
                        f"{calibration_method}."
                    )

                (
                    y_mean,
                    y_std,
                    lower,
                    upper,
                ) = _predict_interval(
                    prediction_auto,
                    X_test,
                    calibration,
                    batch_size=batch_size,
                )

                interval_metrics = (
                    _evaluate_prediction_intervals(
                        y_true=Y_test,
                        lower=lower,
                        upper=upper,
                        target_names=target_names,
                        alpha=alpha,
                        normalization_scale=(
                            normalization_scale
                        ),
                    )
                )

                reliability_df, ece_metrics = (
                    _evaluate_reliability_for_method(
                        calibration_method=(
                            calibration_method
                        ),
                        final_auto=final_auto,
                        X_outer_train=(
                            X_outer_train
                        ),
                        Y_outer_train=(
                            Y_outer_train
                        ),
                        X_test=X_test,
                        Y_test=Y_test,
                        feature_names=feature_names,
                        target_names=target_names,
                        seed=seed,
                        nominal_coverages=(
                            nominal_coverages
                        ),
                        K=K,
                        calibration_frac=(
                            calibration_frac
                        ),
                        conformal_mode=(
                            conformal_mode
                        ),
                        bags=bags,
                        kernel_grid=kernel_grid,
                        icm_rank=icm_rank,
                        noise_floor=noise_floor,
                        init_noise=init_noise,
                        val_frac=(
                            model_selection_val_frac
                        ),
                        obj_weights=obj_weights,
                        device=device,
                        dtype=dtype,
                        batch_size=batch_size,
                    )
                )

                reliability_records.extend(
                    reliability_df.to_dict(
                        orient="records"
                    )
                )

                runtime_seconds = (
                    time.perf_counter()
                    - start_time
                )

                prediction_info = (
                    prediction_auto.info()
                )

                record = {
                    "seed": seed,
                    "calibration_method": (
                        calibration_method
                    ),
                    "selected_family": (
                        prediction_info["family"]
                    ),
                    "selected_kernel": (
                        _serialise_kernel(
                            prediction_info["kernel"]
                        )
                    ),
                    "bags": bags,
                    "total_ensemble_members": (
                        1 + bags
                    ),
                    "alpha": float(alpha),
                    "q_vec": json.dumps(
                        np.asarray(
                            calibration["q_vec"]
                        ).tolist()
                    ),
                    "n_outer_train": int(
                        len(outer_train_indices)
                    ),
                    "n_outer_test": int(
                        len(outer_test_indices)
                    ),
                    "runtime_seconds": float(
                        runtime_seconds
                    ),
                    **interval_metrics,
                    **ece_metrics,
                }

                metric_records.append(record)

                if save_predictions:
                    np.savez_compressed(
                        prediction_path
                        / (
                            f"seed_{seed:03d}_"
                            f"{calibration_method}.npz"
                        ),
                        outer_train_indices=(
                            outer_train_indices
                        ),
                        outer_test_indices=(
                            outer_test_indices
                        ),
                        y_true=Y_test,
                        y_mean=y_mean,
                        y_std=y_std,
                        lower=lower,
                        upper=upper,
                        q_vec=np.asarray(
                            calibration["q_vec"]
                        ),
                        normalization_scale=(
                            normalization_scale
                        ),
                    )

                if verbose:
                    print(
                        f"{calibration_method:<14} | "
                        f"coverage="
                        f"{interval_metrics['macro_coverage']:.4f} | "
                        f"coverage error="
                        f"{interval_metrics['macro_coverage_error']:.4f} | "
                        f"ECE="
                        f"{ece_metrics['macro_ECE']:.4f} | "
                        f"normalized MPIW="
                        f"{interval_metrics['macro_normalized_MPIW']:.4f} | "
                        f"normalized interval score="
                        f"{interval_metrics['macro_normalized_interval_score']:.4f}"
                    )

            except Exception as exc:
                failure_records.append(
                    {
                        "seed": seed,
                        "calibration_method": (
                            calibration_method
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error_message": str(exc),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

                print(
                    f"FAILED: seed={seed}, "
                    f"method={calibration_method}, "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if (
                    calibration_method == "holdout"
                    and prediction_auto is not None
                ):
                    del prediction_auto

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if final_auto is not None:
            del final_auto

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Incremental saving
        if metric_records:
            pd.DataFrame(
                metric_records
            ).to_csv(
                output_path
                / "calibration_metrics_by_seed.csv",
                index=False,
            )

        if reliability_records:
            pd.DataFrame(
                reliability_records
            ).to_csv(
                output_path
                / "reliability_data_by_seed.csv",
                index=False,
            )

        if selected_model_records:
            pd.DataFrame(
                selected_model_records
            ).to_csv(
                output_path
                / "selected_models.csv",
                index=False,
            )

        if failure_records:
            pd.DataFrame(
                failure_records
            ).to_csv(
                output_path
                / "failed_runs.csv",
                index=False,
            )

    if not metric_records:
        raise RuntimeError(
            "All calibration runs failed. "
            "Check failed_runs.csv."
        )

    metrics_by_seed = pd.DataFrame(
        metric_records
    )

    reliability_data = pd.DataFrame(
        reliability_records
    )

    # --------------------------------------------------------
    # Calibration summary
    # --------------------------------------------------------
    excluded_numeric_columns = {
        "seed",
        "alpha",
        "n_outer_train",
        "n_outer_test",
        "n_test",
        "nominal_coverage",
    }

    metric_columns = [
        column
        for column in metrics_by_seed.columns
        if (
            pd.api.types.is_numeric_dtype(
                metrics_by_seed[column]
            )
            and column
            not in excluded_numeric_columns
        )
    ]

    calibration_summary = (
        metrics_by_seed
        .groupby("calibration_method")[
            metric_columns
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    calibration_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in calibration_summary.columns
    ]

    calibration_summary = (
        calibration_summary
        .reset_index()
    )

    # --------------------------------------------------------
    # Reliability summary
    # --------------------------------------------------------
    reliability_summary = (
        reliability_data
        .groupby(
            [
                "calibration_method",
                "target",
                "nominal_coverage",
            ]
        )[
            "empirical_coverage"
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "count",
            ]
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Save final tables
    # --------------------------------------------------------
    metrics_by_seed.to_csv(
        output_path
        / "calibration_metrics_by_seed.csv",
        index=False,
    )

    calibration_summary.to_csv(
        output_path
        / "calibration_summary.csv",
        index=False,
    )

    reliability_data.to_csv(
        output_path
        / "reliability_data_by_seed.csv",
        index=False,
    )

    reliability_summary.to_csv(
        output_path
        / "reliability_summary.csv",
        index=False,
    )

    if selected_model_records:
        pd.DataFrame(
            selected_model_records
        ).to_csv(
            output_path
            / "selected_models.csv",
            index=False,
        )

    if failure_records:
        pd.DataFrame(
            failure_records
        ).to_csv(
            output_path
            / "failed_runs.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Reliability diagrams
    # --------------------------------------------------------
    _save_calibration_reliability_diagrams(
        reliability_summary=(
            reliability_summary
        ),
        target_names=target_names,
        output_dir=(
            reliability_figure_path
        ),
    )

    # --------------------------------------------------------
    # Save settings
    # --------------------------------------------------------
    settings = {
        "surrogate_method": (
            "Model Selection + Bagging"
        ),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "outer_test_frac": (
            outer_test_frac
        ),
        "alpha": alpha,
        "nominal_coverage": (
            1.0 - alpha
        ),
        "K": K,
        "calibration_frac": (
            calibration_frac
        ),
        "conformal_mode": (
            conformal_mode
        ),
        "bags": bags,
        "total_ensemble_members": (
            1 + bags
        ),
        "kernel_grid": [
            _serialise_kernel(kernel)
            for kernel in kernel_grid
        ],
        "icm_rank": icm_rank,
        "noise_floor": noise_floor,
        "init_noise": init_noise,
        "model_selection_val_frac": (
            model_selection_val_frac
        ),
        "nominal_coverages": list(
            nominal_coverages
        ),
        "feature_names": feature_names,
        "target_names": target_names,
        "device": str(device),
        "dtype": str(dtype),
    }

    with open(
        output_path
        / "calibration_settings.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            settings,
            file,
            indent=2,
        )

    if verbose:
        main_columns = [
            column
            for column in [
                "calibration_method",
                "macro_coverage_mean",
                "macro_coverage_std",
                "macro_coverage_error_mean",
                "macro_ECE_mean",
                "macro_normalized_MPIW_mean",
                "macro_normalized_MPIW_std",
                "macro_normalized_interval_score_mean",
                "macro_normalized_interval_score_std",
                "simultaneous_joint_coverage_mean",
                "runtime_seconds_mean",
            ]
            if column
            in calibration_summary.columns
        ]

        print("\n" + "=" * 80)
        print("Calibration benchmark completed")
        print("=" * 80)

        print(
            calibration_summary[
                main_columns
            ].to_string(index=False)
        )

        print(
            f"\nResults saved to: "
            f"{output_path.resolve()}"
        )

    return (
        metrics_by_seed,
        calibration_summary,
        reliability_data,
        reliability_summary,
    )


# ============================================================
# Reliability plotting
# ============================================================

def _save_calibration_reliability_diagrams(
    reliability_summary: pd.DataFrame,
    target_names: Sequence[str],
    output_dir: Path,
) -> None:
    """
    Save one reliability diagram for each objective.
    """

    methods = (
        reliability_summary[
            "calibration_method"
        ]
        .drop_duplicates()
        .tolist()
    )

    for target_name in target_names:
        target_data = reliability_summary[
            reliability_summary["target"]
            == target_name
        ]

        plt.figure(
            figsize=(6.5, 5.5)
        )

        plt.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            label="Ideal",
        )

        for method in methods:
            method_data = (
                target_data[
                    target_data[
                        "calibration_method"
                    ]
                    == method
                ]
                .sort_values(
                    "nominal_coverage"
                )
            )

            plt.plot(
                method_data[
                    "nominal_coverage"
                ],
                method_data[
                    "mean"
                ],
                marker="o",
                label=method,
            )

        plt.xlabel(
            "Nominal coverage"
        )

        plt.ylabel(
            "Empirical coverage"
        )

        plt.title(
            f"Reliability diagram: "
            f"{target_name}"
        )

        plt.xlim(
            0.45,
            1.0,
        )

        plt.ylim(
            0.45,
            1.0,
        )

        plt.grid(
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        safe_target_name = (
            str(target_name)
            .replace(" ", "_")
            .replace("/", "_")
        )

        plt.savefig(
            output_dir
            / (
                f"reliability_"
                f"{safe_target_name}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.savefig(
            output_dir
            / (
                f"reliability_"
                f"{safe_target_name}.pdf"
            ),
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# Nested repeated outer CV calibration benchmark
# ============================================================
#
# The calibration score itself is unchanged:
#
#     z_ij = |y_ij - mu_ij| / max(sigma_ij, eps)
#
# and q is still obtained from the same finite-sample conformal quantile.
#
# What changes is the evaluation structure:
#   1) repeated outer K-fold CV gives every observation one OOF interval/repeat;
#   2) inner K-fold CV selects the surrogate configuration without outer-test data;
#   3) cross-fitted calibration is performed only inside the outer-training set;
#   4) the selected outer-fold configuration is held fixed during calibration
#      cross-fitting, while GP numerical hyperparameters/preprocessing are
#      re-estimated on each calibration-training complement;
#   5) metrics are computed on all N OOF observations per repeat;
#   6) CIs bootstrap original observation IDs, never seeds/repeats.


def _cross_fitted_calibration_fixed_configuration(
    X_outer_train,
    Y_outer_train,
    *,
    selected_family,
    selected_kernel,
    feature_names,
    target_names,
    seed,
    K,
    bags,
    kernel_grid,
    icm_rank,
    noise_floor,
    init_noise,
    obj_weights,
    device,
    dtype,
    batch_size,
    verbose=False,
):
    """
    Cross-fitted standardized residuals using a FIXED outer-selected
    surrogate configuration.

    Each calibration model is nevertheless freshly fitted on its own
    calibration-training complement, so its GP hyperparameters, preprocessing
    and bagging are learned without the held-out calibration fold.
    """
    X_outer_train = np.asarray(
        X_outer_train,
        dtype=float,
    )
    Y_outer_train = np.asarray(
        Y_outer_train,
        dtype=float,
    )
    if Y_outer_train.ndim == 1:
        Y_outer_train = Y_outer_train[:, None]

    n_samples, n_targets = Y_outer_train.shape

    splitter = KFold(
        n_splits=int(K),
        shuffle=True,
        random_state=int(seed),
    )

    z_oof = np.full(
        (n_samples, n_targets),
        np.nan,
        dtype=float,
    )
    fold_records = []

    for cal_fold, (
        fit_idx,
        cal_idx,
    ) in enumerate(
        splitter.split(np.arange(n_samples))
    ):
        fold_seed = (
            int(seed) * 1000
            + cal_fold
            + 1
        )

        if verbose:
            print(
                f"  Calibration fold {cal_fold + 1}/{K}: "
                f"fit={len(fit_idx)}, cal={len(cal_idx)}"
            )

        auto = _fit_fixed_surrogate_configuration(
            X_outer_train[fit_idx],
            Y_outer_train[fit_idx],
            selected_family=selected_family,
            selected_kernel=selected_kernel,
            feature_names=feature_names,
            target_names=target_names,
            seed=fold_seed,
            bags=bags,
            kernel_grid=kernel_grid,
            icm_rank=icm_rank,
            noise_floor=noise_floor,
            init_noise=init_noise,
            obj_weights=obj_weights,
            device=device,
            dtype=dtype,
        )

        with torch.inference_mode():
            mu, std = auto.predict(
                X_outer_train[cal_idx],
                batch_size=batch_size,
            )
        mu = np.asarray(mu, dtype=float)
        std = np.maximum(
            np.asarray(std, dtype=float),
            1e-12,
        )

        z_oof[cal_idx] = (
            np.abs(
                Y_outer_train[cal_idx]
                - mu
            )
            / std
        )

        fold_records.append(
            {
                "calibration_fold": int(
                    cal_fold
                ),
                "n_fit": int(len(fit_idx)),
                "n_cal": int(len(cal_idx)),
                "family": str(
                    selected_family
                ),
                "kernel": _serialise_kernel(
                    selected_kernel
                ),
            }
        )

        del auto
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(z_oof).any():
        raise RuntimeError(
            "Cross-fitted calibration did not produce one residual "
            "for every outer-training observation."
        )

    return {
        "source": "cross_fitted",
        "standardized_residuals": z_oof,
        "K": int(K),
        "n_cal": int(n_samples),
        "fold_records": fold_records,
    }


def _holdout_calibration_nested_selection(
    X_outer_train,
    Y_outer_train,
    *,
    feature_names,
    target_names,
    seed,
    calibration_frac,
    inner_folds,
    bags,
    kernel_grid,
    icm_rank,
    noise_floor,
    init_noise,
    obj_weights,
    device,
    dtype,
    batch_size,
):
    """
    Preserve the original holdout-calibration idea:
      fit subset -> calibration subset.

    To keep it leakage-free under the new protocol, surrogate configuration
    selection is performed by inner CV using the FIT subset only.
    """
    X_outer_train = np.asarray(
        X_outer_train,
        dtype=float,
    )
    Y_outer_train = np.asarray(
        Y_outer_train,
        dtype=float,
    )

    all_idx = np.arange(
        X_outer_train.shape[0]
    )
    fit_idx, cal_idx = train_test_split(
        all_idx,
        test_size=float(calibration_frac),
        random_state=int(seed),
        shuffle=True,
    )

    selected = (
        _inner_cv_select_surrogate_configuration(
            X_outer_train[fit_idx],
            Y_outer_train[fit_idx],
            feature_names=feature_names,
            target_names=target_names,
            seed=int(seed) + 17,
            n_splits=inner_folds,
            kernel_grid=kernel_grid,
            icm_rank=icm_rank,
            noise_floor=noise_floor,
            init_noise=init_noise,
            obj_weights=obj_weights,
            device=device,
            dtype=dtype,
            verbose=False,
        )
    )

    auto = _fit_fixed_surrogate_configuration(
        X_outer_train[fit_idx],
        Y_outer_train[fit_idx],
        selected_family=selected["family"],
        selected_kernel=selected["kernel"],
        feature_names=feature_names,
        target_names=target_names,
        seed=int(seed) + 29,
        bags=bags,
        kernel_grid=kernel_grid,
        icm_rank=icm_rank,
        noise_floor=noise_floor,
        init_noise=init_noise,
        obj_weights=obj_weights,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        mu_cal, std_cal = auto.predict(
            X_outer_train[cal_idx],
            batch_size=batch_size,
        )

    mu_cal = np.asarray(
        mu_cal,
        dtype=float,
    )
    std_cal = np.maximum(
        np.asarray(std_cal, dtype=float),
        1e-12,
    )

    z_cal = (
        np.abs(
            Y_outer_train[cal_idx]
            - mu_cal
        )
        / std_cal
    )

    state = {
        "source": "holdout",
        "standardized_residuals": z_cal,
        "n_fit": int(len(fit_idx)),
        "n_cal": int(len(cal_idx)),
        "calibration_frac": float(
            calibration_frac
        ),
        "selected_family": selected[
            "family"
        ],
        "selected_kernel": selected[
            "kernel"
        ],
        "inner_cv_score": float(
            selected["score"]
        ),
    }
    return auto, state


def _calibration_from_state(
    *,
    method,
    state,
    alpha,
    n_targets,
    conformal_mode,
):
    if method == "uncalibrated":
        return _uncalibrated_gaussian_calibration(
            n_targets=n_targets,
            alpha=alpha,
        )

    return _calibration_from_standardized_residuals(
        np.asarray(
            state["standardized_residuals"],
            dtype=float,
        ),
        alpha=alpha,
        conformal_mode=conformal_mode,
    )


def _interval_from_mean_std(
    y_mean,
    y_std,
    calibration,
):
    y_mean = np.asarray(
        y_mean,
        dtype=float,
    )
    y_std = np.maximum(
        np.asarray(y_std, dtype=float),
        1e-12,
    )
    q_vec = np.asarray(
        calibration["q_vec"],
        dtype=float,
    ).reshape(1, -1)
    half_width = q_vec * y_std
    return (
        y_mean - half_width,
        y_mean + half_width,
    )


def _ece_from_reliability_cube(
    *,
    y_true,
    lower_cube,
    upper_cube,
    nominal_coverages,
    target_names,
    observation_ids=None,
):
    """
    ECE for one repeat.

    lower_cube / upper_cube shape:
        (n_nominal_levels, N, M)
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    if observation_ids is None:
        observation_ids = np.arange(
            y_true.shape[0]
        )
    observation_ids = np.asarray(
        observation_ids,
        dtype=int,
    )

    target_ece = {}
    for target_id, target in enumerate(
        target_names
    ):
        errors = []
        for level_id, nominal in enumerate(
            nominal_coverages
        ):
            covered = (
                (
                    y_true[
                        observation_ids,
                        target_id,
                    ]
                    >= lower_cube[
                        level_id,
                        observation_ids,
                        target_id,
                    ]
                )
                & (
                    y_true[
                        observation_ids,
                        target_id,
                    ]
                    <= upper_cube[
                        level_id,
                        observation_ids,
                        target_id,
                    ]
                )
            )
            empirical = float(
                np.mean(covered)
            )
            errors.append(
                abs(
                    empirical
                    - float(nominal)
                )
            )

        target_ece[
            f"{target}_ECE"
        ] = float(np.mean(errors))

    target_ece["macro_ECE"] = float(
        np.mean(
            list(target_ece.values())
        )
    )
    return target_ece


def _bootstrap_calibration_metric_cis(
    *,
    y_true,
    interval_store,
    target_names,
    nominal_coverages,
    alpha,
    point_metrics,
    n_bootstrap=2000,
    confidence=0.95,
    seed=2026,
):
    """
    Dataset-level CIs by bootstrapping ORIGINAL observation IDs.

    Each resampled observation carries all of its repeated-CV OOF intervals
    with it.  Repeats are averaged inside each bootstrap replicate and are
    never resampled as if they were independent experiments.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    n_samples = y_true.shape[0]
    normalization_scale = np.maximum(
        np.std(
            y_true,
            axis=0,
            ddof=0,
        ),
        1e-12,
    )

    rng = np.random.default_rng(
        int(seed)
    )
    tail = (
        1.0 - float(confidence)
    ) / 2.0
    records = []

    for method, store in interval_store.items():
        lower = np.asarray(
            store["lower"],
            dtype=float,
        )
        upper = np.asarray(
            store["upper"],
            dtype=float,
        )
        rel_lower = np.asarray(
            store["reliability_lower"],
            dtype=float,
        )
        rel_upper = np.asarray(
            store["reliability_upper"],
            dtype=float,
        )

        n_repeats = lower.shape[0]
        bootstrap_values = {}

        for _ in range(int(n_bootstrap)):
            sampled_ids = rng.integers(
                0,
                n_samples,
                size=n_samples,
            )

            repeat_dicts = []
            for repeat_id in range(
                n_repeats
            ):
                d = _evaluate_prediction_intervals(
                    y_true=y_true[
                        sampled_ids
                    ],
                    lower=lower[
                        repeat_id,
                        sampled_ids,
                        :,
                    ],
                    upper=upper[
                        repeat_id,
                        sampled_ids,
                        :,
                    ],
                    target_names=target_names,
                    alpha=alpha,
                    normalization_scale=(
                        normalization_scale
                    ),
                )

                ece = _ece_from_reliability_cube(
                    y_true=y_true,
                    lower_cube=rel_lower[
                        repeat_id
                    ],
                    upper_cube=rel_upper[
                        repeat_id
                    ],
                    nominal_coverages=(
                        nominal_coverages
                    ),
                    target_names=target_names,
                    observation_ids=sampled_ids,
                )
                d.update(ece)
                repeat_dicts.append(d)

            averaged = _average_metric_dicts(
                repeat_dicts
            )
            for metric, value in averaged.items():
                if metric in {
                    "nominal_coverage",
                    "n_test",
                }:
                    continue
                bootstrap_values.setdefault(
                    metric, []
                ).append(value)

        point = (
            point_metrics[
                point_metrics[
                    "calibration_method"
                ] == method
            ]
            .drop(
                columns=[
                    c for c in [
                        "repeat",
                        "partition_seed",
                        "calibration_method",
                        "n_observations",
                    ]
                    if c in point_metrics.columns
                ],
                errors="ignore",
            )
            .mean(numeric_only=True)
        )

        for metric, values in bootstrap_values.items():
            values = np.asarray(
                values,
                dtype=float,
            )
            records.append(
                {
                    "calibration_method": (
                        method
                    ),
                    "metric": metric,
                    "point_estimate": float(
                        point.get(
                            metric,
                            np.nan,
                        )
                    ),
                    "ci_lower": float(
                        np.quantile(
                            values,
                            tail,
                        )
                    ),
                    "ci_upper": float(
                        np.quantile(
                            values,
                            1.0 - tail,
                        )
                    ),
                    "confidence": float(
                        confidence
                    ),
                    "bootstrap_unit": (
                        "original_observation_id"
                    ),
                    "n_bootstrap": int(
                        n_bootstrap
                    ),
                }
            )

    return pd.DataFrame(records)


def run_model_selection_bagging_nested_repeated_calibration(
    X,
    Y,
    feature_names,
    target_names,
    *,
    output_dir="nested_repeated_oof_calibration",
    repeat_seeds=tuple(range(10)),
    outer_folds=5,
    inner_folds=5,
    alpha=0.10,
    calibration_folds=5,
    calibration_frac=0.20,
    conformal_mode="per_target",
    bags=20,
    kernel_grid=None,
    icm_rank=2,
    noise_floor=1e-4,
    init_noise=1e-3,
    obj_weights=None,
    nominal_coverages=(
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ),
    batch_size=4096,
    device=torch.device("cpu"),
    dtype=torch.double,
    bootstrap_reps=2000,
    bootstrap_confidence=0.95,
    save_predictions=True,
    verbose=True,
):
    """
    Calibration comparison using nested repeated outer CV and complete
    observation-level OOF aggregation.

    Methods retained from the original notebook:
      1. uncalibrated Gaussian
      2. holdout conformal calibration
      3. cross-fitted conformal scale calibration

    The surrogate family/kernel search, GP construction, bagging and
    standardized-residual conformal calibration are retained.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]

    if kernel_grid is None:
        kernel_grid = [
            ("RBF", None),
            ("Matern", 0.5),
            ("Matern", 1.5),
            ("Matern", 2.5),
        ]

    kernel_grid = list(kernel_grid)
    feature_names = list(feature_names)
    target_names = list(target_names)
    repeat_seeds = [
        int(s) for s in repeat_seeds
    ]
    nominal_coverages = [
        float(c)
        for c in nominal_coverages
    ]

    n_samples, n_targets = Y.shape
    n_repeats = len(repeat_seeds)
    n_levels = len(
        nominal_coverages
    )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_path / "reliability_diagrams").mkdir(
        parents=True,
        exist_ok=True,
    )
    if save_predictions:
        (output_path / "predictions").mkdir(
            parents=True,
            exist_ok=True,
        )

    methods = [
        "uncalibrated",
        "holdout",
        "cross_fitted",
    ]

    interval_store = {
        method: {
            "mean": np.full(
                (
                    n_repeats,
                    n_samples,
                    n_targets,
                ),
                np.nan,
                dtype=float,
            ),
            "std": np.full(
                (
                    n_repeats,
                    n_samples,
                    n_targets,
                ),
                np.nan,
                dtype=float,
            ),
            "lower": np.full(
                (
                    n_repeats,
                    n_samples,
                    n_targets,
                ),
                np.nan,
                dtype=float,
            ),
            "upper": np.full(
                (
                    n_repeats,
                    n_samples,
                    n_targets,
                ),
                np.nan,
                dtype=float,
            ),
            "reliability_lower": np.full(
                (
                    n_repeats,
                    n_levels,
                    n_samples,
                    n_targets,
                ),
                np.nan,
                dtype=float,
            ),
            "reliability_upper": np.full(
                (
                    n_repeats,
                    n_levels,
                    n_samples,
                    n_targets,
                ),
                np.nan,
                dtype=float,
            ),
            "outer_fold": np.full(
                (
                    n_repeats,
                    n_samples,
                ),
                -1,
                dtype=int,
            ),
        }
        for method in methods
    }

    selected_model_records = []
    calibration_fold_records = []
    metric_records = []
    reliability_records = []
    oof_long_records = []

    normalization_scale = np.maximum(
        np.std(
            Y,
            axis=0,
            ddof=0,
        ),
        1e-12,
    )

    for repeat_id, partition_seed in enumerate(
        repeat_seeds
    ):
        if verbose:
            print("\n" + "=" * 88)
            print(
                f"Calibration outer repeat "
                f"{repeat_id + 1}/{n_repeats} "
                f"(partition seed={partition_seed})"
            )
            print("=" * 88)

        splitter = KFold(
            n_splits=int(outer_folds),
            shuffle=True,
            random_state=int(partition_seed),
        )

        for outer_fold, (
            outer_train_idx,
            outer_test_idx,
        ) in enumerate(
            splitter.split(
                np.arange(n_samples)
            )
        ):
            X_outer_train = X[
                outer_train_idx
            ]
            Y_outer_train = Y[
                outer_train_idx
            ]
            X_test = X[
                outer_test_idx
            ]

            if verbose:
                print(
                    f"\nOuter fold {outer_fold + 1}/{outer_folds}: "
                    f"train={len(outer_train_idx)}, "
                    f"test={len(outer_test_idx)}"
                )

            selection_seed = (
                int(partition_seed) * 10000
                + outer_fold * 100
                + 41
            )

            selected = (
                _inner_cv_select_surrogate_configuration(
                    X_outer_train,
                    Y_outer_train,
                    feature_names=feature_names,
                    target_names=target_names,
                    seed=selection_seed,
                    n_splits=inner_folds,
                    kernel_grid=kernel_grid,
                    icm_rank=icm_rank,
                    noise_floor=noise_floor,
                    init_noise=init_noise,
                    obj_weights=obj_weights,
                    device=device,
                    dtype=dtype,
                    verbose=False,
                )
            )

            selected_model_records.append(
                {
                    "repeat": int(repeat_id),
                    "partition_seed": int(
                        partition_seed
                    ),
                    "outer_fold": int(
                        outer_fold
                    ),
                    "model_role": (
                        "final_outer_model"
                    ),
                    "selected_family": (
                        selected["family"]
                    ),
                    "selected_kernel": (
                        _serialise_kernel(
                            selected["kernel"]
                        )
                    ),
                    "inner_cv_score": float(
                        selected["score"]
                    ),
                    "n_outer_train": int(
                        len(outer_train_idx)
                    ),
                    "n_outer_test": int(
                        len(outer_test_idx)
                    ),
                }
            )

            # Final full-outer-training surrogate used by uncalibrated and
            # cross-fitted CFSC predictions.
            final_auto = (
                _fit_fixed_surrogate_configuration(
                    X_outer_train,
                    Y_outer_train,
                    selected_family=(
                        selected["family"]
                    ),
                    selected_kernel=(
                        selected["kernel"]
                    ),
                    feature_names=feature_names,
                    target_names=target_names,
                    seed=selection_seed + 1,
                    bags=bags,
                    kernel_grid=kernel_grid,
                    icm_rank=icm_rank,
                    noise_floor=noise_floor,
                    init_noise=init_noise,
                    obj_weights=obj_weights,
                    device=device,
                    dtype=dtype,
                )
            )

            # Holdout comparator preserves a genuinely held-out calibration
            # subset and performs its own nested selection using only the fit
            # subset.
            holdout_auto, holdout_state = (
                _holdout_calibration_nested_selection(
                    X_outer_train,
                    Y_outer_train,
                    feature_names=feature_names,
                    target_names=target_names,
                    seed=selection_seed + 100,
                    calibration_frac=(
                        calibration_frac
                    ),
                    inner_folds=inner_folds,
                    bags=bags,
                    kernel_grid=kernel_grid,
                    icm_rank=icm_rank,
                    noise_floor=noise_floor,
                    init_noise=init_noise,
                    obj_weights=obj_weights,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                )
            )

            # CFSC residuals: selected configuration is fixed; each
            # calibration fold re-estimates preprocessing / GP parameters on
            # the fold complement.
            cross_state = (
                _cross_fitted_calibration_fixed_configuration(
                    X_outer_train,
                    Y_outer_train,
                    selected_family=(
                        selected["family"]
                    ),
                    selected_kernel=(
                        selected["kernel"]
                    ),
                    feature_names=feature_names,
                    target_names=target_names,
                    seed=selection_seed + 200,
                    K=calibration_folds,
                    bags=bags,
                    kernel_grid=kernel_grid,
                    icm_rank=icm_rank,
                    noise_floor=noise_floor,
                    init_noise=init_noise,
                    obj_weights=obj_weights,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                    verbose=False,
                )
            )

            for fold_info in cross_state[
                "fold_records"
            ]:
                calibration_fold_records.append(
                    {
                        "repeat": int(
                            repeat_id
                        ),
                        "partition_seed": int(
                            partition_seed
                        ),
                        "outer_fold": int(
                            outer_fold
                        ),
                        **fold_info,
                    }
                )

            # Predict once per prediction model; all nominal levels reuse the
            # same predictive mean/std and only change q.
            with torch.inference_mode():
                final_mean, final_std = (
                    final_auto.predict(
                        X_test,
                        batch_size=batch_size,
                    )
                )
                hold_mean, hold_std = (
                    holdout_auto.predict(
                        X_test,
                        batch_size=batch_size,
                    )
                )

            final_mean = np.asarray(
                final_mean,
                dtype=float,
            )
            final_std = np.maximum(
                np.asarray(
                    final_std,
                    dtype=float,
                ),
                1e-12,
            )
            hold_mean = np.asarray(
                hold_mean,
                dtype=float,
            )
            hold_std = np.maximum(
                np.asarray(
                    hold_std,
                    dtype=float,
                ),
                1e-12,
            )

            method_context = {
                "uncalibrated": {
                    "mean": final_mean,
                    "std": final_std,
                    "state": None,
                },
                "holdout": {
                    "mean": hold_mean,
                    "std": hold_std,
                    "state": holdout_state,
                },
                "cross_fitted": {
                    "mean": final_mean,
                    "std": final_std,
                    "state": cross_state,
                },
            }

            for method in methods:
                context = method_context[
                    method
                ]
                y_mean = context["mean"]
                y_std = context["std"]

                main_calibration = (
                    _calibration_from_state(
                        method=method,
                        state=context["state"],
                        alpha=alpha,
                        n_targets=n_targets,
                        conformal_mode=(
                            conformal_mode
                        ),
                    )
                )
                lower, upper = (
                    _interval_from_mean_std(
                        y_mean,
                        y_std,
                        main_calibration,
                    )
                )

                interval_store[method][
                    "mean"
                ][
                    repeat_id,
                    outer_test_idx,
                    :,
                ] = y_mean
                interval_store[method][
                    "std"
                ][
                    repeat_id,
                    outer_test_idx,
                    :,
                ] = y_std
                interval_store[method][
                    "lower"
                ][
                    repeat_id,
                    outer_test_idx,
                    :,
                ] = lower
                interval_store[method][
                    "upper"
                ][
                    repeat_id,
                    outer_test_idx,
                    :,
                ] = upper
                interval_store[method][
                    "outer_fold"
                ][
                    repeat_id,
                    outer_test_idx,
                ] = outer_fold

                for level_id, nominal in enumerate(
                    nominal_coverages
                ):
                    level_calibration = (
                        _calibration_from_state(
                            method=method,
                            state=context[
                                "state"
                            ],
                            alpha=(
                                1.0
                                - float(nominal)
                            ),
                            n_targets=(
                                n_targets
                            ),
                            conformal_mode=(
                                conformal_mode
                            ),
                        )
                    )
                    lo, hi = (
                        _interval_from_mean_std(
                            y_mean,
                            y_std,
                            level_calibration,
                        )
                    )
                    interval_store[method][
                        "reliability_lower"
                    ][
                        repeat_id,
                        level_id,
                        outer_test_idx,
                        :,
                    ] = lo
                    interval_store[method][
                        "reliability_upper"
                    ][
                        repeat_id,
                        level_id,
                        outer_test_idx,
                        :,
                    ] = hi

            del final_auto
            del holdout_auto
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Aggregate the COMPLETE N-observation OOF set.
        # ----------------------------------------------------
        for method in methods:
            store = interval_store[
                method
            ]

            required_arrays = [
                store["mean"][repeat_id],
                store["std"][repeat_id],
                store["lower"][repeat_id],
                store["upper"][repeat_id],
                store["reliability_lower"][
                    repeat_id
                ],
                store["reliability_upper"][
                    repeat_id
                ],
            ]
            if any(
                np.isnan(arr).any()
                for arr in required_arrays
            ):
                raise RuntimeError(
                    f"Incomplete OOF calibration predictions for "
                    f"repeat={repeat_id}, method={method}."
                )

            interval_metrics = (
                _evaluate_prediction_intervals(
                    y_true=Y,
                    lower=store["lower"][
                        repeat_id
                    ],
                    upper=store["upper"][
                        repeat_id
                    ],
                    target_names=target_names,
                    alpha=alpha,
                    normalization_scale=(
                        normalization_scale
                    ),
                )
            )

            ece = _ece_from_reliability_cube(
                y_true=Y,
                lower_cube=(
                    store[
                        "reliability_lower"
                    ][repeat_id]
                ),
                upper_cube=(
                    store[
                        "reliability_upper"
                    ][repeat_id]
                ),
                nominal_coverages=(
                    nominal_coverages
                ),
                target_names=target_names,
            )

            metric_records.append(
                {
                    "repeat": int(
                        repeat_id
                    ),
                    "partition_seed": int(
                        partition_seed
                    ),
                    "calibration_method": (
                        method
                    ),
                    "n_observations": int(
                        n_samples
                    ),
                    **interval_metrics,
                    **ece,
                }
            )

            for level_id, nominal in enumerate(
                nominal_coverages
            ):
                covered = (
                    (
                        Y
                        >= store[
                            "reliability_lower"
                        ][
                            repeat_id,
                            level_id,
                        ]
                    )
                    & (
                        Y
                        <= store[
                            "reliability_upper"
                        ][
                            repeat_id,
                            level_id,
                        ]
                    )
                )

                for target_id, target in enumerate(
                    target_names
                ):
                    empirical = float(
                        np.mean(
                            covered[
                                :,
                                target_id,
                            ]
                        )
                    )
                    reliability_records.append(
                        {
                            "repeat": int(
                                repeat_id
                            ),
                            "partition_seed": int(
                                partition_seed
                            ),
                            "calibration_method": (
                                method
                            ),
                            "target": target,
                            "nominal_coverage": float(
                                nominal
                            ),
                            "empirical_coverage": (
                                empirical
                            ),
                            "absolute_calibration_error": (
                                abs(
                                    empirical
                                    - nominal
                                )
                            ),
                            "n_observations": int(
                                n_samples
                            ),
                        }
                    )

            for observation_id in range(
                n_samples
            ):
                row = {
                    "repeat": int(
                        repeat_id
                    ),
                    "partition_seed": int(
                        partition_seed
                    ),
                    "calibration_method": (
                        method
                    ),
                    "observation_id": int(
                        observation_id
                    ),
                    "outer_fold": int(
                        store["outer_fold"][
                            repeat_id,
                            observation_id,
                        ]
                    ),
                }
                for target_id, target in enumerate(
                    target_names
                ):
                    row[
                        f"{target}_true"
                    ] = float(
                        Y[
                            observation_id,
                            target_id,
                        ]
                    )
                    row[
                        f"{target}_mean"
                    ] = float(
                        store["mean"][
                            repeat_id,
                            observation_id,
                            target_id,
                        ]
                    )
                    row[
                        f"{target}_std"
                    ] = float(
                        store["std"][
                            repeat_id,
                            observation_id,
                            target_id,
                        ]
                    )
                    row[
                        f"{target}_lower"
                    ] = float(
                        store["lower"][
                            repeat_id,
                            observation_id,
                            target_id,
                        ]
                    )
                    row[
                        f"{target}_upper"
                    ] = float(
                        store["upper"][
                            repeat_id,
                            observation_id,
                            target_id,
                        ]
                    )
                oof_long_records.append(
                    row
                )

            if verbose:
                print(
                    f"Repeat {repeat_id + 1}: "
                    f"{method:<14} | "
                    f"coverage="
                    f"{interval_metrics['macro_coverage']:.4f} | "
                    f"ECE={ece['macro_ECE']:.4f} | "
                    f"norm MPIW="
                    f"{interval_metrics['macro_normalized_MPIW']:.4f}"
                )

    metrics_by_repeat = pd.DataFrame(
        metric_records
    )
    reliability_data = pd.DataFrame(
        reliability_records
    )
    oof_predictions = pd.DataFrame(
        oof_long_records
    )
    selected_models = pd.DataFrame(
        selected_model_records
    )
    calibration_folds_df = pd.DataFrame(
        calibration_fold_records
    )

    excluded = {
        "repeat",
        "partition_seed",
        "n_observations",
        "nominal_coverage",
        "n_test",
    }
    metric_columns = [
        c for c in metrics_by_repeat.columns
        if (
            pd.api.types.is_numeric_dtype(
                metrics_by_repeat[c]
            )
            and c not in excluded
        )
    ]

    calibration_summary = (
        metrics_by_repeat
        .groupby(
            "calibration_method"
        )[metric_columns]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )
    calibration_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in calibration_summary.columns
    ]
    calibration_summary = (
        calibration_summary.reset_index()
    )

    reliability_summary = (
        reliability_data
        .groupby(
            [
                "calibration_method",
                "target",
                "nominal_coverage",
            ]
        )["empirical_coverage"]
        .agg(
            [
                "mean",
                "std",
                "median",
                "count",
            ]
        )
        .reset_index()
    )

    bootstrap_ci = (
        _bootstrap_calibration_metric_cis(
            y_true=Y,
            interval_store=interval_store,
            target_names=target_names,
            nominal_coverages=(
                nominal_coverages
            ),
            alpha=alpha,
            point_metrics=metrics_by_repeat,
            n_bootstrap=bootstrap_reps,
            confidence=(
                bootstrap_confidence
            ),
            seed=20260902,
        )
    )

    metrics_by_repeat.to_csv(
        output_path
        / "calibration_metrics_by_outer_repeat.csv",
        index=False,
    )
    calibration_summary.to_csv(
        output_path
        / "repeat_partition_sensitivity_summary.csv",
        index=False,
    )
    reliability_data.to_csv(
        output_path
        / "reliability_by_outer_repeat.csv",
        index=False,
    )
    reliability_summary.to_csv(
        output_path
        / "reliability_summary.csv",
        index=False,
    )
    bootstrap_ci.to_csv(
        output_path
        / "observation_bootstrap_confidence_intervals.csv",
        index=False,
    )
    oof_predictions.to_csv(
        output_path
        / "observation_level_oof_intervals.csv",
        index=False,
    )
    selected_models.to_csv(
        output_path
        / "selected_models_by_outer_fold.csv",
        index=False,
    )
    calibration_folds_df.to_csv(
        output_path
        / "cross_fitted_calibration_folds.csv",
        index=False,
    )

    _save_calibration_reliability_diagrams(
        reliability_summary=(
            reliability_summary
        ),
        target_names=target_names,
        output_dir=(
            output_path
            / "reliability_diagrams"
        ),
    )

    if save_predictions:
        for method in methods:
            safe = method.replace(
                " ", "_"
            )
            store = interval_store[
                method
            ]
            np.savez_compressed(
                output_path
                / "predictions"
                / f"{safe}_oof_intervals.npz",
                y_true=Y,
                y_mean=store["mean"],
                y_std=store["std"],
                lower=store["lower"],
                upper=store["upper"],
                reliability_lower=(
                    store[
                        "reliability_lower"
                    ]
                ),
                reliability_upper=(
                    store[
                        "reliability_upper"
                    ]
                ),
                outer_fold=(
                    store[
                        "outer_fold"
                    ]
                ),
                nominal_coverages=np.asarray(
                    nominal_coverages,
                    dtype=float,
                ),
                repeat_seeds=np.asarray(
                    repeat_seeds,
                    dtype=int,
                ),
            )

    settings = {
        "evaluation_protocol": (
            "nested repeated outer K-fold CV with "
            "observation-level OOF predictions"
        ),
        "surrogate_method": (
            "Model Selection + Bagging"
        ),
        "repeat_seeds": repeat_seeds,
        "n_repeats": len(
            repeat_seeds
        ),
        "outer_folds": int(
            outer_folds
        ),
        "inner_folds": int(
            inner_folds
        ),
        "calibration_folds": int(
            calibration_folds
        ),
        "alpha": float(alpha),
        "nominal_coverage": float(
            1.0 - alpha
        ),
        "calibration_frac": float(
            calibration_frac
        ),
        "conformal_mode": str(
            conformal_mode
        ),
        "bags": int(bags),
        "total_ensemble_members": int(
            1 + bags
        ),
        "kernel_grid": [
            _serialise_kernel(k)
            for k in kernel_grid
        ],
        "feature_names": feature_names,
        "target_names": target_names,
        "independent_observational_units": int(
            n_samples
        ),
        "repeats_interpretation": (
            "partition sensitivity only; not independent replication"
        ),
        "dataset_level_ci_unit": (
            "original_observation_id"
        ),
        "bootstrap_reps": int(
            bootstrap_reps
        ),
        "bootstrap_confidence": float(
            bootstrap_confidence
        ),
    }

    with open(
        output_path / "settings.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            settings,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return (
        metrics_by_repeat,
        calibration_summary,
        reliability_data,
        reliability_summary,
        bootstrap_ci,
    )
