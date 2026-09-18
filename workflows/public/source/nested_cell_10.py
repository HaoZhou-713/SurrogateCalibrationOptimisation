import gc
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import erf


# ============================================================
# Evaluation
# ============================================================

def evaluate_probabilistic_predictions(
    y_true: Union[np.ndarray, list],
    y_mean: Union[np.ndarray, list],
    y_std: Union[np.ndarray, list],
    target_names: Optional[Sequence[str]] = None,
    normalization_location: Optional[np.ndarray] = None,
    normalization_scale: Optional[np.ndarray] = None,
    std_floor: float = 1e-8,
) -> Dict[str, float]:
    """
    Evaluate deterministic and probabilistic multi-output predictions.

    For each objective, this function calculates metrics in the original
    physical scale:

        - RMSE
        - MAE
        - R2
        - Gaussian NLL
        - Gaussian CRPS

    It also calculates scale-independent metrics after target-wise
    standardisation:

        y_normalized = (y - normalization_location) / normalization_scale

    The recommended aggregate metrics are:

        - macro_NRMSE
        - macro_NMAE
        - macro_R2
        - macro_normalized_NLL
        - macro_normalized_CRPS

    Each objective receives equal weight in the macro averages.

    Parameters
    ----------
    y_true
        Ground-truth values with shape (N,) or (N, M).

    y_mean
        Predictive means with the same shape as y_true.

    y_std
        Predictive standard deviations with the same shape as y_true.
        This must contain standard deviations, not variances.

    target_names
        Names of the M objectives.

    normalization_location
        Per-objective normalization location with shape (M,).
        It is recommended to use the training-data mean.

        If None, the mean of y_true is used.

    normalization_scale
        Per-objective normalization scale with shape (M,).
        It is recommended to use the training-data standard deviation.

        If None, the standard deviation of y_true is used.

    std_floor
        Numerical lower bound for predictive and normalization standard
        deviations.

    Returns
    -------
    Dict[str, float]
        Per-objective and macro-average metrics.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    # Convert single-output arrays from (N,) to (N, 1)
    if y_true.ndim == 1:
        y_true = y_true[:, None]

    if y_mean.ndim == 1:
        y_mean = y_mean[:, None]

    if y_std.ndim == 1:
        y_std = y_std[:, None]

    if y_true.ndim != 2:
        raise ValueError(
            f"y_true must have shape (N,) or (N, M), "
            f"but received {y_true.shape}."
        )

    if y_true.shape != y_mean.shape:
        raise ValueError(
            f"y_true and y_mean must have the same shape, "
            f"but received {y_true.shape} and {y_mean.shape}."
        )

    if y_true.shape != y_std.shape:
        raise ValueError(
            f"y_true and y_std must have the same shape, "
            f"but received {y_true.shape} and {y_std.shape}."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.all(np.isfinite(y_mean)):
        raise ValueError("y_mean contains NaN or infinite values.")

    if not np.all(np.isfinite(y_std)):
        raise ValueError("y_std contains NaN or infinite values.")

    if std_floor <= 0:
        raise ValueError("std_floor must be greater than zero.")

    n_samples, n_targets = y_true.shape

    if target_names is None:
        target_names = [
            f"target_{j}" for j in range(n_targets)
        ]
    else:
        target_names = list(target_names)

    if len(target_names) != n_targets:
        raise ValueError(
            f"Expected {n_targets} target names, "
            f"but received {len(target_names)}."
        )

    # --------------------------------------------------------
    # Predictive standard-deviation protection
    # --------------------------------------------------------
    y_std = np.maximum(y_std, std_floor)

    # --------------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------------
    if normalization_location is None:
        normalization_location = np.mean(
            y_true,
            axis=0,
        )
    else:
        normalization_location = np.asarray(
            normalization_location,
            dtype=np.float64,
        ).reshape(-1)

    if normalization_scale is None:
        normalization_scale = np.std(
            y_true,
            axis=0,
            ddof=0,
        )
    else:
        normalization_scale = np.asarray(
            normalization_scale,
            dtype=np.float64,
        ).reshape(-1)

    if normalization_location.shape != (n_targets,):
        raise ValueError(
            "normalization_location must have shape "
            f"({n_targets},), but received "
            f"{normalization_location.shape}."
        )

    if normalization_scale.shape != (n_targets,):
        raise ValueError(
            "normalization_scale must have shape "
            f"({n_targets},), but received "
            f"{normalization_scale.shape}."
        )

    if not np.all(np.isfinite(normalization_location)):
        raise ValueError(
            "normalization_location contains NaN or infinite values."
        )

    if not np.all(np.isfinite(normalization_scale)):
        raise ValueError(
            "normalization_scale contains NaN or infinite values."
        )

    normalization_scale = np.maximum(
        normalization_scale,
        std_floor,
    )

    # --------------------------------------------------------
    # Standardised predictions and targets
    # --------------------------------------------------------
    y_true_normalized = (
        y_true - normalization_location
    ) / normalization_scale

    y_mean_normalized = (
        y_mean - normalization_location
    ) / normalization_scale

    y_std_normalized = (
        y_std / normalization_scale
    )

    y_std_normalized = np.maximum(
        y_std_normalized,
        std_floor,
    )

    # --------------------------------------------------------
    # Internal metric calculator
    # --------------------------------------------------------
    def calculate_metrics(
        true_values: np.ndarray,
        predicted_mean: np.ndarray,
        predicted_std: np.ndarray,
    ) -> Dict[str, np.ndarray]:

        error = predicted_mean - true_values

        squared_error = error ** 2
        absolute_error = np.abs(error)

        # RMSE and MAE per objective
        rmse = np.sqrt(
            np.mean(squared_error, axis=0)
        )

        mae = np.mean(
            absolute_error,
            axis=0,
        )

        # R2 per objective
        ss_res = np.sum(
            squared_error,
            axis=0,
        )

        true_mean = np.mean(
            true_values,
            axis=0,
        )

        ss_tot = np.sum(
            (true_values - true_mean) ** 2,
            axis=0,
        )

        r2 = np.full(
            true_values.shape[1],
            np.nan,
            dtype=np.float64,
        )

        valid_r2 = (
            ss_tot > np.finfo(np.float64).eps
        )

        r2[valid_r2] = (
            1.0
            - ss_res[valid_r2]
            / ss_tot[valid_r2]
        )

        # Standardized residual for Gaussian metrics
        z = (
            true_values - predicted_mean
        ) / predicted_std

        # Gaussian negative log-likelihood
        nll_values = (
            0.5 * np.log(2.0 * np.pi)
            + np.log(predicted_std)
            + 0.5 * z ** 2
        )

        nll = np.mean(
            nll_values,
            axis=0,
        )

        # Gaussian CRPS
        standard_normal_pdf = (
            np.exp(-0.5 * z ** 2)
            / np.sqrt(2.0 * np.pi)
        )

        standard_normal_cdf = (
            0.5
            * (
                1.0
                + erf(z / np.sqrt(2.0))
            )
        )

        crps_values = predicted_std * (
            z * (
                2.0 * standard_normal_cdf
                - 1.0
            )
            + 2.0 * standard_normal_pdf
            - 1.0 / np.sqrt(np.pi)
        )

        crps = np.mean(
            crps_values,
            axis=0,
        )

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "NLL": nll,
            "CRPS": crps,
        }

    # Metrics in original physical units
    original_metrics = calculate_metrics(
        true_values=y_true,
        predicted_mean=y_mean,
        predicted_std=y_std,
    )

    # Metrics in normalized target space
    normalized_metrics = calculate_metrics(
        true_values=y_true_normalized,
        predicted_mean=y_mean_normalized,
        predicted_std=y_std_normalized,
    )

    metrics: Dict[str, float] = {}

    # --------------------------------------------------------
    # Per-objective metrics
    # --------------------------------------------------------
    for j, target_name in enumerate(target_names):

        # Original physical scale
        metrics[f"{target_name}_RMSE"] = float(
            original_metrics["RMSE"][j]
        )

        metrics[f"{target_name}_MAE"] = float(
            original_metrics["MAE"][j]
        )

        metrics[f"{target_name}_R2"] = float(
            original_metrics["R2"][j]
        )

        metrics[f"{target_name}_NLL"] = float(
            original_metrics["NLL"][j]
        )

        metrics[f"{target_name}_CRPS"] = float(
            original_metrics["CRPS"][j]
        )

        # Standardized scale
        metrics[f"{target_name}_NRMSE"] = float(
            normalized_metrics["RMSE"][j]
        )

        metrics[f"{target_name}_NMAE"] = float(
            normalized_metrics["MAE"][j]
        )

        metrics[
            f"{target_name}_normalized_NLL"
        ] = float(
            normalized_metrics["NLL"][j]
        )

        metrics[
            f"{target_name}_normalized_CRPS"
        ] = float(
            normalized_metrics["CRPS"][j]
        )

        metrics[
            f"{target_name}_normalization_location"
        ] = float(
            normalization_location[j]
        )

        metrics[
            f"{target_name}_normalization_scale"
        ] = float(
            normalization_scale[j]
        )

    # --------------------------------------------------------
    # Recommended equal-objective macro averages
    # --------------------------------------------------------
    metrics["macro_NRMSE"] = float(
        np.mean(normalized_metrics["RMSE"])
    )

    metrics["macro_NMAE"] = float(
        np.mean(normalized_metrics["MAE"])
    )

    metrics["macro_R2"] = float(
        np.nanmean(original_metrics["R2"])
    )

    metrics["macro_normalized_NLL"] = float(
        np.mean(normalized_metrics["NLL"])
    )

    metrics["macro_normalized_CRPS"] = float(
        np.mean(normalized_metrics["CRPS"])
    )

    # Original-unit macro averages are retained only for reference.
    # They are not scale invariant.
    metrics["macro_raw_RMSE"] = float(
        np.mean(original_metrics["RMSE"])
    )

    metrics["macro_raw_MAE"] = float(
        np.mean(original_metrics["MAE"])
    )

    metrics["macro_raw_NLL"] = float(
        np.mean(original_metrics["NLL"])
    )

    metrics["macro_raw_CRPS"] = float(
        np.mean(original_metrics["CRPS"])
    )

    metrics["n_samples"] = int(n_samples)
    metrics["n_targets"] = int(n_targets)

    return metrics


# ============================================================
# Benchmark
# ============================================================

def run_gp_seed_benchmark(
    X,
    Y,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    output_dir: str = "gp_seed_benchmark",
    seeds: Sequence[int] = tuple(range(20)),
    test_frac: float = 0.10,
    val_frac: float = 0.12,
    fixed_family: str = "IND",
    fixed_kernel: Tuple[str, Optional[float]] = (
        "Matern",
        2.5,
    ),
    kernel_grid: Optional[
        Sequence[Tuple[str, Optional[float]]]
    ] = None,
    icm_rank: int = 2,
    bagging_bags: int = 5,
    noise_floor: float = 1e-4,
    init_noise: float = 1e-3,
    obj_weights: Optional[np.ndarray] = None,
    use_ks_for_test: bool = False,
    device: Any = torch.device("cpu"),
    dtype: Any = torch.double,
    save_predictions: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare three GP configurations over multiple random seeds.

    Methods
    -------
    1. Single GP
       - model selection disabled
       - fixed family and kernel
       - no bagging

    2. Model Selection
       - selects ICM or IND
       - performs kernel selection
       - no bagging

    3. Model Selection + Bagging
       - selects ICM or IND
       - performs kernel selection
       - uses bootstrap bagging

    Evaluation
    ----------
    Metrics are calculated separately for every objective.

    Aggregate metrics use target-wise standardization based on the
    train + validation data for the corresponding seed.

    All three methods for the same seed must use the same test split.

    Returns
    -------
    all_results
        One row per seed and method.

    summary
        Mean, standard deviation, median, minimum and maximum across seeds.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (N, D), "
            f"but received {X.shape}."
        )

    if Y.ndim == 1:
        Y = Y[:, None]

    if Y.ndim != 2:
        raise ValueError(
            f"Y must have shape (N, M), "
            f"but received {Y.shape}."
        )

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of samples. "
            f"Received {X.shape[0]} and {Y.shape[0]}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "Y contains NaN or infinite values."
        )

    feature_names = list(feature_names)
    target_names = list(target_names)

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Expected {X.shape[1]} feature names, "
            f"but received {len(feature_names)}."
        )

    if len(target_names) != Y.shape[1]:
        raise ValueError(
            f"Expected {Y.shape[1]} target names, "
            f"but received {len(target_names)}."
        )

    if bagging_bags <= 0:
        raise ValueError(
            "bagging_bags must be greater than zero."
        )

    fixed_family = fixed_family.upper()

    if fixed_family not in {"ICM", "IND"}:
        raise ValueError(
            "fixed_family must be either 'ICM' or 'IND'."
        )

    if kernel_grid is None:
        kernel_grid = [
            ("RBF", None),
            ("Matern", 0.5),
            ("Matern", 1.5),
            ("Matern", 2.5),
        ]

    kernel_grid = list(kernel_grid)
    seeds = [int(seed) for seed in seeds]

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        output_path / "predictions"
    )

    if save_predictions:
        prediction_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Experimental methods
    # --------------------------------------------------------
    experiment_settings = [
        {
            "method": "Single GP",
            "use_model_selection": False,
            "bags": 0,
        },
        {
            "method": "Model Selection",
            "use_model_selection": True,
            "bags": 0,
        },
        {
            "method": "Model Selection + Bagging",
            "use_model_selection": True,
            "bags": int(bagging_bags),
        },
    ]

    all_records = []
    selected_model_records = []
    failed_records = []

    # Store the first method's split for each seed.
    # The other two methods must match it.
    split_reference_by_seed: Dict[
        int,
        Dict[str, np.ndarray],
    ] = {}

    total_runs = (
        len(seeds)
        * len(experiment_settings)
    )

    completed_runs = 0

    # --------------------------------------------------------
    # Seed loop
    # --------------------------------------------------------
    for seed in seeds:

        if verbose:
            print("\n" + "=" * 80)
            print(f"Seed {seed}")
            print("=" * 80)

        for setting in experiment_settings:

            method = setting["method"]
            model = None

            completed_runs += 1

            if verbose:
                print(
                    f"\n[{completed_runs}/{total_runs}] "
                    f"Seed={seed}, method={method}"
                )

            try:
                # ------------------------------------------------
                # Reproducibility
                # ------------------------------------------------
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                try:
                    torch.use_deterministic_algorithms(
                        True,
                        warn_only=True,
                    )
                except Exception:
                    pass

                # ------------------------------------------------
                # Configuration
                # ------------------------------------------------
                config = AutoConfig(
                    kernel_grid=list(kernel_grid),
                    icm_rank=icm_rank,
                    bags=setting["bags"],
                    noise_floor=noise_floor,
                    init_noise=init_noise,
                    seed=seed,
                    val_frac=val_frac,
                    device=device,
                    dtype=dtype,
                    obj_weights=obj_weights,
                    test_frac=test_frac,
                    use_ks_for_test=use_ks_for_test,
                    use_model_selection=setting[
                        "use_model_selection"
                    ],
                    fixed_family=fixed_family,
                    fixed_kernel=fixed_kernel,
                )

                model = AutoMultiOutputGP(
                    feature_names=feature_names,
                    target_names=target_names,
                    config=config,
                )

                # ------------------------------------------------
                # Fit
                # ------------------------------------------------
                training_start = (
                    time.perf_counter()
                )

                model.fit(
                    X,
                    Y,
                    test_frac=test_frac,
                    val_frac=val_frac,
                    verbose=verbose,
                )

                training_time = (
                    time.perf_counter()
                    - training_start
                )

                if not model._is_fitted:
                    raise RuntimeError(
                        "model.fit() returned, but "
                        "model._is_fitted is still False."
                    )

                # ------------------------------------------------
                # Read splits
                # ------------------------------------------------
                required_splits = {
                    "tr",
                    "va",
                    "te",
                }

                missing_splits = (
                    required_splits
                    - set(model.split.keys())
                )

                if missing_splits:
                    raise RuntimeError(
                        "The fitted model is missing split indices: "
                        f"{sorted(missing_splits)}"
                    )

                train_indices = np.asarray(
                    model.split["tr"],
                    dtype=int,
                )

                validation_indices = np.asarray(
                    model.split["va"],
                    dtype=int,
                )

                test_indices = np.asarray(
                    model.split["te"],
                    dtype=int,
                )

                if train_indices.size == 0:
                    raise RuntimeError(
                        "The training set is empty."
                    )

                if test_indices.size == 0:
                    raise RuntimeError(
                        "The test set is empty."
                    )

                # ------------------------------------------------
                # Confirm fair, identical splits
                # ------------------------------------------------
                current_split = {
                    "tr": np.sort(train_indices),
                    "va": np.sort(validation_indices),
                    "te": np.sort(test_indices),
                }

                if seed not in split_reference_by_seed:
                    split_reference_by_seed[
                        seed
                    ] = {
                        key: value.copy()
                        for key, value
                        in current_split.items()
                    }

                else:
                    reference_split = (
                        split_reference_by_seed[seed]
                    )

                    for split_name in (
                        "tr",
                        "va",
                        "te",
                    ):
                        if not np.array_equal(
                            current_split[split_name],
                            reference_split[split_name],
                        ):
                            raise RuntimeError(
                                f"Split '{split_name}' differs "
                                f"between methods for seed {seed}."
                            )

                # ------------------------------------------------
                # Evaluation data
                # ------------------------------------------------
                X_test = X[test_indices]
                Y_test = Y[test_indices]

                # Use train + validation statistics for normalization.
                # This avoids using test-set information to define scale.
                reference_indices = np.concatenate(
                    [
                        train_indices,
                        validation_indices,
                    ]
                )

                Y_reference = Y[
                    reference_indices
                ]

                normalization_location = np.mean(
                    Y_reference,
                    axis=0,
                )

                normalization_scale = np.std(
                    Y_reference,
                    axis=0,
                    ddof=0,
                )

                normalization_scale = np.maximum(
                    normalization_scale,
                    1e-8,
                )

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------
                prediction_start = (
                    time.perf_counter()
                )

                Y_mean, Y_std = model.predict(
                    X_test
                )

                prediction_time = (
                    time.perf_counter()
                    - prediction_start
                )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------
                metrics = (
                    evaluate_probabilistic_predictions(
                        y_true=Y_test,
                        y_mean=Y_mean,
                        y_std=Y_std,
                        target_names=target_names,
                        normalization_location=(
                            normalization_location
                        ),
                        normalization_scale=(
                            normalization_scale
                        ),
                    )
                )

                # ------------------------------------------------
                # Model information
                # ------------------------------------------------
                model_info = model.info()

                selected_family = (
                    model_info["family"]
                )

                selected_kernel = (
                    model_info["kernel"]
                )

                record = {
                    "seed": seed,
                    "method": method,
                    "use_model_selection": setting[
                        "use_model_selection"
                    ],
                    "configured_bags": setting[
                        "bags"
                    ],
                    "total_ensemble_members": (
                        1 + setting["bags"]
                    ),
                    "selected_family": (
                        selected_family
                    ),
                    "selected_kernel": (
                        _serialise_kernel(
                            selected_kernel
                        )
                    ),
                    "fixed_family": fixed_family,
                    "fixed_kernel": (
                        _serialise_kernel(
                            fixed_kernel
                        )
                    ),
                    "n_train": len(
                        train_indices
                    ),
                    "n_validation": len(
                        validation_indices
                    ),
                    "n_test": len(
                        test_indices
                    ),
                    "training_time_seconds": (
                        training_time
                    ),
                    "prediction_time_seconds": (
                        prediction_time
                    ),
                    **metrics,
                }

                all_records.append(record)

                selected_model_records.append(
                    {
                        "seed": seed,
                        "method": method,
                        "selected_family": (
                            selected_family
                        ),
                        "selected_kernel": (
                            _serialise_kernel(
                                selected_kernel
                            )
                        ),
                        "configured_bags": (
                            setting["bags"]
                        ),
                        "total_ensemble_members": (
                            1 + setting["bags"]
                        ),
                    }
                )

                # ------------------------------------------------
                # Optional prediction files
                # ------------------------------------------------
                if save_predictions:

                    safe_method_name = (
                        method.lower()
                        .replace(" ", "_")
                        .replace("+", "plus")
                    )

                    np.savez_compressed(
                        prediction_path
                        / (
                            f"seed_{seed:03d}_"
                            f"{safe_method_name}.npz"
                        ),
                        seed=np.asarray(seed),
                        method=np.asarray(method),
                        train_indices=train_indices,
                        validation_indices=(
                            validation_indices
                        ),
                        test_indices=test_indices,
                        normalization_location=(
                            normalization_location
                        ),
                        normalization_scale=(
                            normalization_scale
                        ),
                        y_true=Y_test,
                        y_mean=Y_mean,
                        y_std=Y_std,
                    )

                # ------------------------------------------------
                # Incremental saving
                # ------------------------------------------------
                pd.DataFrame(
                    all_records
                ).to_csv(
                    output_path
                    / "all_seed_metrics.csv",
                    index=False,
                )

                pd.DataFrame(
                    selected_model_records
                ).to_csv(
                    output_path
                    / "selected_models.csv",
                    index=False,
                )

                if verbose:
                    print(
                        f"Completed: "
                        f"family={selected_family}, "
                        f"kernel={selected_kernel}, "
                        f"macro NRMSE="
                        f"{metrics['macro_NRMSE']:.6g}, "
                        f"macro R2="
                        f"{metrics['macro_R2']:.6g}, "
                        f"macro normalized NLL="
                        f"{metrics['macro_normalized_NLL']:.6g}, "
                        f"macro normalized CRPS="
                        f"{metrics['macro_normalized_CRPS']:.6g}"
                    )

            except Exception as exc:

                failure = {
                    "seed": seed,
                    "method": method,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_message": str(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }

                failed_records.append(failure)

                pd.DataFrame(
                    failed_records
                ).to_csv(
                    output_path
                    / "failed_runs.csv",
                    index=False,
                )

                print(
                    f"FAILED: seed={seed}, "
                    f"method={method}, "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                if model is not None:
                    del model

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Check successful runs
    # --------------------------------------------------------
    if not all_records:
        raise RuntimeError(
            "All GP runs failed. "
            "Check failed_runs.csv."
        )

    # --------------------------------------------------------
    # All per-seed results
    # --------------------------------------------------------
    all_results = pd.DataFrame(
        all_records
    )

    all_results = (
        all_results
        .sort_values(
            by=["seed", "method"]
        )
        .reset_index(drop=True)
    )

    all_results.to_csv(
        output_path
        / "all_seed_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary across seeds
    # --------------------------------------------------------
    metric_columns = (
        _find_metric_columns(
            all_results
        )
    )

    summary = (
        all_results
        .groupby("method")[metric_columns]
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

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in summary.columns
    ]

    summary = summary.reset_index()

    summary.to_csv(
        output_path
        / "summary_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compact main summary
    # --------------------------------------------------------
    main_metric_columns = [
        "macro_NRMSE",
        "macro_NMAE",
        "macro_R2",
        "macro_normalized_NLL",
        "macro_normalized_CRPS",
    ]

    main_summary = (
        all_results
        .groupby("method")[
            main_metric_columns
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
            ]
        )
    )

    main_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in main_summary.columns
    ]

    main_summary = (
        main_summary
        .reset_index()
    )

    main_summary.to_csv(
        output_path
        / "main_summary_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Per-objective summary only
    # --------------------------------------------------------
    target_metric_columns = (
        _find_target_metric_columns(
            results=all_results,
            target_names=target_names,
        )
    )

    target_summary = (
        all_results
        .groupby("method")[
            target_metric_columns
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

    target_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in target_summary.columns
    ]

    target_summary = (
        target_summary
        .reset_index()
    )

    target_summary.to_csv(
        output_path
        / "per_objective_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Model-selection frequency
    # --------------------------------------------------------
    selected_models = pd.DataFrame(
        selected_model_records
    )

    if not selected_models.empty:

        selection_frequency = (
            selected_models
            .groupby(
                [
                    "method",
                    "selected_family",
                    "selected_kernel",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        selection_frequency[
            "frequency"
        ] = (
            selection_frequency["count"]
            / selection_frequency
            .groupby("method")["count"]
            .transform("sum")
        )

        selection_frequency.to_csv(
            output_path
            / "model_selection_frequency.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------
    settings_to_save = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "test_frac": test_frac,
        "val_frac": val_frac,
        "fixed_family": fixed_family,
        "fixed_kernel": (
            _serialise_kernel(
                fixed_kernel
            )
        ),
        "kernel_grid": [
            _serialise_kernel(kernel)
            for kernel in kernel_grid
        ],
        "icm_rank": icm_rank,
        "bagging_bags": (
            bagging_bags
        ),
        "bagging_total_models": (
            1 + bagging_bags
        ),
        "noise_floor": noise_floor,
        "init_noise": init_noise,
        "use_ks_for_test": (
            use_ks_for_test
        ),
        "feature_names": (
            feature_names
        ),
        "target_names": (
            target_names
        ),
        "normalization_source": (
            "train_plus_validation"
        ),
        "normalization_type": (
            "per_target_z_score"
        ),
        "save_predictions": (
            save_predictions
        ),
        "device": str(device),
        "dtype": str(dtype),
    }

    with open(
        output_path
        / "experiment_settings.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            settings_to_save,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if failed_records:
        pd.DataFrame(
            failed_records
        ).to_csv(
            output_path
            / "failed_runs.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("Benchmark completed")
        print("=" * 80)

        print(
            f"Successful runs: "
            f"{len(all_results)}"
        )

        print(
            f"Failed runs: "
            f"{len(failed_records)}"
        )

        print(
            f"Results saved to: "
            f"{output_path.resolve()}"
        )

        display_columns = [
            column
            for column in [
                "method",
                "macro_NRMSE_mean",
                "macro_NRMSE_std",
                "macro_NMAE_mean",
                "macro_NMAE_std",
                "macro_R2_mean",
                "macro_R2_std",
                "macro_normalized_NLL_mean",
                "macro_normalized_NLL_std",
                "macro_normalized_CRPS_mean",
                "macro_normalized_CRPS_std",
            ]
            if column
            in main_summary.columns
        ]

        print("\nMain normalized summary:")
        print(
            main_summary[
                display_columns
            ].to_string(index=False)
        )

    return all_results, summary


# ============================================================
# Helpers
# ============================================================

def _serialise_kernel(
    kernel: Any,
) -> str:
    """
    Convert kernel configurations to a stable string.
    """

    if kernel is None:
        return "None"

    try:
        return json.dumps(kernel)

    except TypeError:
        return str(kernel)


def _find_metric_columns(
    results: pd.DataFrame,
) -> list:
    """
    Find numerical evaluation metric columns.

    Normalization locations and scales are excluded because they are
    auxiliary quantities, not model-performance metrics.
    """

    metric_tokens = (
        "RMSE",
        "MAE",
        "R2",
        "NLL",
        "CRPS",
    )

    excluded_suffixes = (
        "_normalization_location",
        "_normalization_scale",
    )

    metric_columns = []

    for column in results.columns:

        if column.endswith(
            excluded_suffixes
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            results[column]
        ):
            continue

        if any(
            token in column
            for token in metric_tokens
        ):
            metric_columns.append(column)

    if not metric_columns:
        raise RuntimeError(
            "No evaluation metric columns were found."
        )

    return metric_columns


def _find_target_metric_columns(
    results: pd.DataFrame,
    target_names: Sequence[str],
) -> list:
    """
    Find objective-specific metric columns.

    Macro-average columns are not included.
    """

    target_metric_columns = []

    for target_name in target_names:

        possible_columns = [
            f"{target_name}_RMSE",
            f"{target_name}_MAE",
            f"{target_name}_R2",
            f"{target_name}_NLL",
            f"{target_name}_CRPS",
            f"{target_name}_NRMSE",
            f"{target_name}_NMAE",
            f"{target_name}_normalized_NLL",
            f"{target_name}_normalized_CRPS",
        ]

        for column in possible_columns:
            if column in results.columns:
                target_metric_columns.append(
                    column
                )

    if not target_metric_columns:
        raise RuntimeError(
            "No target-specific metric columns were found."
        )

    return target_metric_columns


# ============================================================
# Nested repeated CV with observation-level OOF predictions
# ============================================================
#
# This section intentionally leaves the original GP construction code
# unchanged.  The only change is the evaluation / model-selection wrapper:
#
#   repeated outer K-fold CV
#       -> inner K-fold model selection on outer-training data only
#       -> refit the selected surrogate on ALL outer-training observations
#       -> predict the held-out outer fold
#       -> assemble one OOF prediction for every original observation
#
# Random repeats quantify partition sensitivity.  They are NOT treated as
# independent physical replications.  Dataset-level confidence intervals are
# obtained by bootstrapping original observation IDs.


def _make_manual_auto(
    *,
    feature_names,
    target_names,
    seed,
    bags,
    kernel_grid,
    icm_rank,
    noise_floor,
    init_noise,
    obj_weights,
    device,
    dtype,
):
    """Create the existing AutoMultiOutputGP without invoking its split logic."""
    config = AutoConfig(
        kernel_grid=list(kernel_grid),
        icm_rank=icm_rank,
        bags=int(bags),
        noise_floor=noise_floor,
        init_noise=init_noise,
        seed=int(seed),
        val_frac=0.12,          # unused by the manual fixed-configuration fit
        test_frac=0.10,         # unused by the manual fixed-configuration fit
        use_ks_for_test=False,
        device=device,
        dtype=dtype,
        obj_weights=obj_weights,
        use_model_selection=False,
        fixed_family="IND",
        fixed_kernel=("Matern", 2.5),
    )
    return AutoMultiOutputGP(
        feature_names=list(feature_names),
        target_names=list(target_names),
        config=config,
    )


def _scale_from_training_data(
    X_train,
    Y_train,
    X_eval,
    Y_eval,
    *,
    device,
    dtype,
):
    """
    Fit preprocessing ONLY on the supplied training subset.

    This is used inside inner CV so that validation observations cannot
    influence scaling, and in final refits so outer-test observations cannot
    influence scaling.
    """
    Xtr = _to_tensor(X_train, device, dtype)
    Ytr = _to_tensor(Y_train, device, dtype)
    Xev = _to_tensor(X_eval, device, dtype)
    Yev = _to_tensor(Y_eval, device, dtype)

    with torch.no_grad():
        x_min = Xtr.min(dim=0).values
        x_max = Xtr.max(dim=0).values
        x_span = (x_max - x_min).clamp_min(1e-12)

        y_mu = Ytr.mean(dim=0)
        y_sd = Ytr.std(dim=0, unbiased=False).clamp_min(1e-12)

    Xtr_s = (Xtr - x_min) / x_span
    Ytr_s = (Ytr - y_mu) / y_sd
    Xev_s = (Xev - x_min) / x_span
    Yev_s = (Yev - y_mu) / y_sd

    return (
        Xtr_s,
        Ytr_s,
        Xev_s,
        Yev_s,
        x_min,
        x_span,
        y_mu,
        y_sd,
    )


def _inner_cv_select_surrogate_configuration(
    X_train,
    Y_train,
    *,
    feature_names,
    target_names,
    seed,
    n_splits,
    kernel_grid,
    icm_rank,
    noise_floor,
    init_noise,
    obj_weights,
    device,
    dtype,
    verbose=False,
):
    """
    Inner K-fold CV for surrogate configuration selection.

    The candidate logic is the same as the original implementation:
      * ICM: search one shared kernel.
      * IND: search the best kernel separately for each target.
      * Choose ICM vs IND using the weighted validation RMSE.

    The difference is that validation scores are now aggregated across
    K inner folds rather than coming from one random validation split.
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]

    n_samples = X_train.shape[0]
    n_targets = Y_train.shape[1]

    if n_splits < 2:
        raise ValueError("inner n_splits must be at least 2.")
    if n_samples < n_splits:
        raise ValueError(
            f"Outer-training size {n_samples} is smaller than inner_folds={n_splits}."
        )

    weights = (
        np.asarray(obj_weights, dtype=float)
        if obj_weights is not None
        else np.ones(n_targets, dtype=float) / n_targets
    )
    weights = weights / weights.sum()

    kernel_grid = list(kernel_grid)

    # ICM score: candidate kernel -> one weighted score per inner fold.
    icm_fold_scores = {
        tuple(kernel): [] for kernel in kernel_grid
    }

    # IND score: target -> candidate kernel -> one RMSE per inner fold.
    ind_fold_scores = {
        j: {tuple(kernel): [] for kernel in kernel_grid}
        for j in range(n_targets)
    }

    splitter = KFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(seed),
    )

    for inner_fold, (fit_idx, val_idx) in enumerate(
        splitter.split(np.arange(n_samples))
    ):
        fold_seed = int(seed) * 1000 + inner_fold + 1
        _set_seed(fold_seed) if "_set_seed" in globals() else None

        auto = _make_manual_auto(
            feature_names=feature_names,
            target_names=target_names,
            seed=fold_seed,
            bags=0,
            kernel_grid=kernel_grid,
            icm_rank=icm_rank,
            noise_floor=noise_floor,
            init_noise=init_noise,
            obj_weights=obj_weights,
            device=device,
            dtype=dtype,
        )

        (
            Xfit_s,
            Yfit_s,
            Xval_s,
            Yval_s,
            x_min,
            x_span,
            y_mu,
            y_sd,
        ) = _scale_from_training_data(
            X_train[fit_idx],
            Y_train[fit_idx],
            X_train[val_idx],
            Y_train[val_idx],
            device=device,
            dtype=dtype,
        )

        # Set scalers only so the existing private GP builders/predictors have
        # the same context as in the original class.
        auto.x_min = x_min
        auto.x_span = x_span
        auto.y_mu = y_mu
        auto.y_sd = y_sd

        # ---------- ICM candidates ----------
        for kernel in kernel_grid:
            model = auto._build_icm(
                Xfit_s,
                Yfit_s,
                kernel=tuple(kernel),
            )
            with torch.no_grad():
                mu_val_s, _ = auto._predict_icm_single(
                    model,
                    Xval_s,
                )
                mu_val = mu_val_s * y_sd + y_mu
                y_val = Yval_s * y_sd + y_mu

            rmse_each = np.sqrt(
                np.mean(
                    (
                        mu_val.detach().cpu().numpy()
                        - y_val.detach().cpu().numpy()
                    ) ** 2,
                    axis=0,
                )
            )
            icm_fold_scores[tuple(kernel)].append(
                float(np.sum(rmse_each * weights))
            )
            del model

        # ---------- IND candidates ----------
        for target_id in range(n_targets):
            yfit_j = Yfit_s[:, [target_id]]
            yval_j = Yval_s[:, [target_id]]

            for kernel in kernel_grid:
                model = auto._build_ind(
                    Xfit_s,
                    yfit_j,
                    kernel=tuple(kernel),
                )
                with torch.no_grad():
                    mu_j_s = model.posterior(
                        Xval_s
                    ).mean.squeeze(-1)
                    mu_j = mu_j_s * y_sd[target_id] + y_mu[target_id]
                    y_j = (
                        yval_j.squeeze(-1) * y_sd[target_id]
                        + y_mu[target_id]
                    )

                rmse_j = float(
                    np.sqrt(
                        np.mean(
                            (
                                mu_j.detach().cpu().numpy()
                                - y_j.detach().cpu().numpy()
                            ) ** 2
                        )
                    )
                )
                ind_fold_scores[target_id][tuple(kernel)].append(
                    rmse_j
                )
                del model

        del auto
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Mean validation score across inner folds.
    icm_mean_scores = {
        kernel: float(np.mean(scores))
        for kernel, scores in icm_fold_scores.items()
    }
    best_icm_kernel = min(
        icm_mean_scores,
        key=icm_mean_scores.get,
    )
    best_icm_score = icm_mean_scores[best_icm_kernel]

    best_ind_kernels = []
    best_ind_target_scores = []
    for target_id in range(n_targets):
        target_means = {
            kernel: float(np.mean(scores))
            for kernel, scores
            in ind_fold_scores[target_id].items()
        }
        best_kernel = min(
            target_means,
            key=target_means.get,
        )
        best_ind_kernels.append(best_kernel)
        best_ind_target_scores.append(
            target_means[best_kernel]
        )

    best_ind_score = float(
        np.sum(
            np.asarray(best_ind_target_scores)
            * weights
        )
    )

    if best_icm_score <= best_ind_score:
        selected_family = "ICM"
        selected_kernel = best_icm_kernel
        selected_score = best_icm_score
    else:
        selected_family = "IND"
        selected_kernel = best_ind_kernels
        selected_score = best_ind_score

    diagnostics = []
    for kernel, fold_scores in icm_fold_scores.items():
        diagnostics.append(
            {
                "family": "ICM",
                "target": "__joint__",
                "kernel": _serialise_kernel(kernel)
                if "_serialise_kernel" in globals()
                else str(kernel),
                "mean_inner_cv_score": float(np.mean(fold_scores)),
                "std_inner_cv_score": float(np.std(fold_scores, ddof=1))
                if len(fold_scores) > 1 else 0.0,
            }
        )

    for target_id, target_name in enumerate(target_names):
        for kernel, fold_scores in ind_fold_scores[target_id].items():
            diagnostics.append(
                {
                    "family": "IND",
                    "target": target_name,
                    "kernel": _serialise_kernel(kernel)
                    if "_serialise_kernel" in globals()
                    else str(kernel),
                    "mean_inner_cv_score": float(np.mean(fold_scores)),
                    "std_inner_cv_score": float(np.std(fold_scores, ddof=1))
                    if len(fold_scores) > 1 else 0.0,
                }
            )

    if verbose:
        print(
            f"Inner CV selected {selected_family}, "
            f"kernel={selected_kernel}, score={selected_score:.6g}"
        )

    return {
        "family": selected_family,
        "kernel": selected_kernel,
        "score": float(selected_score),
        "best_icm_kernel": best_icm_kernel,
        "best_icm_score": float(best_icm_score),
        "best_ind_kernels": best_ind_kernels,
        "best_ind_score": float(best_ind_score),
        "diagnostics": pd.DataFrame(diagnostics),
    }


def _fit_fixed_surrogate_configuration(
    X_train,
    Y_train,
    *,
    selected_family,
    selected_kernel,
    feature_names,
    target_names,
    seed,
    bags,
    kernel_grid,
    icm_rank,
    noise_floor,
    init_noise,
    obj_weights,
    device,
    dtype,
):
    """
    Refit one selected surrogate configuration on ALL supplied training data.

    Numerical GP hyperparameters, preprocessing and bagging are re-estimated
    from this training set.  No observation outside X_train/Y_train is used.
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    auto = _make_manual_auto(
        feature_names=feature_names,
        target_names=target_names,
        seed=seed,
        bags=bags,
        kernel_grid=kernel_grid,
        icm_rank=icm_rank,
        noise_floor=noise_floor,
        init_noise=init_noise,
        obj_weights=obj_weights,
        device=device,
        dtype=dtype,
    )

    Xt = _to_tensor(X_train, device, dtype)
    Yt = _to_tensor(Y_train, device, dtype)

    with torch.no_grad():
        x_min = Xt.min(dim=0).values
        x_max = Xt.max(dim=0).values
        x_span = (x_max - x_min).clamp_min(1e-12)
        y_mu = Yt.mean(dim=0)
        y_sd = Yt.std(dim=0, unbiased=False).clamp_min(1e-12)

    Xs = (Xt - x_min) / x_span
    Ys = (Yt - y_mu) / y_sd

    auto.x_min = x_min
    auto.x_span = x_span
    auto.y_mu = y_mu
    auto.y_sd = y_sd
    auto.X_all = Xt
    auto.Y_all = Yt
    auto.best_family = str(selected_family).upper()
    auto.best_kernel = selected_kernel

    if auto.best_family == "ICM":
        auto.models = auto._train_icm_ensemble(
            Xs,
            Ys,
            kernel=selected_kernel,
            bags=int(bags),
        )
    elif auto.best_family == "IND":
        auto.models = auto._train_ind_ensemble_per_target(
            Xs,
            Ys,
            kernels=selected_kernel,
            bags=int(bags),
        )
    else:
        raise ValueError(
            "selected_family must be 'ICM' or 'IND'."
        )

    auto.split = {
        "tr": np.arange(X_train.shape[0], dtype=int),
        "va": np.asarray([], dtype=int),
        "te": np.asarray([], dtype=int),
    }
    auto._is_fitted = True
    return auto


def _average_metric_dicts(metric_dicts):
    """Average numeric metric dictionaries with identical keys."""
    keys = metric_dicts[0].keys()
    out = {}
    for key in keys:
        values = [
            d[key]
            for d in metric_dicts
            if isinstance(d.get(key), (int, float, np.integer, np.floating))
        ]
        if values:
            out[key] = float(np.mean(values))
    return out


def _bootstrap_prediction_metric_cis(
    *,
    y_true,
    prediction_store,
    target_names,
    point_metrics,
    n_bootstrap=2000,
    confidence=0.95,
    seed=2026,
):
    """
    Observation-level bootstrap for repeated-CV predictive metrics.

    The resampling unit is the ORIGINAL observation ID.  For each bootstrap
    sample, the same resampled IDs are evaluated in every repeat; repeat-level
    metrics are then averaged.  Repeats are therefore not treated as
    independent replications.
    """
    y_true = np.asarray(y_true, dtype=float)
    n_samples = y_true.shape[0]
    location = np.mean(y_true, axis=0)
    scale = np.maximum(np.std(y_true, axis=0, ddof=0), 1e-12)

    rng = np.random.default_rng(int(seed))
    tail = (1.0 - confidence) / 2.0
    records = []

    for method, store in prediction_store.items():
        mean_cube = np.asarray(store["mean"], dtype=float)
        std_cube = np.asarray(store["std"], dtype=float)
        n_repeats = mean_cube.shape[0]

        bootstrap_values = {}

        for _ in range(int(n_bootstrap)):
            sampled_ids = rng.integers(
                0,
                n_samples,
                size=n_samples,
            )

            repeat_metric_dicts = []
            for repeat_id in range(n_repeats):
                repeat_metric_dicts.append(
                    evaluate_probabilistic_predictions(
                        y_true=y_true[sampled_ids],
                        y_mean=mean_cube[repeat_id, sampled_ids],
                        y_std=std_cube[repeat_id, sampled_ids],
                        target_names=target_names,
                        normalization_location=location,
                        normalization_scale=scale,
                    )
                )

            averaged = _average_metric_dicts(
                repeat_metric_dicts
            )
            for metric, value in averaged.items():
                if (
                    metric.endswith("_normalization_location")
                    or metric.endswith("_normalization_scale")
                ):
                    continue
                bootstrap_values.setdefault(
                    metric, []
                ).append(value)

        method_point = (
            point_metrics[
                point_metrics["method"] == method
            ]
            .drop(
                columns=[
                    c for c in [
                        "repeat",
                        "partition_seed",
                        "method",
                    ]
                    if c in point_metrics.columns
                ],
                errors="ignore",
            )
            .mean(numeric_only=True)
        )

        for metric, values in bootstrap_values.items():
            values = np.asarray(values, dtype=float)
            records.append(
                {
                    "method": method,
                    "metric": metric,
                    "point_estimate": float(
                        method_point.get(
                            metric,
                            np.nan,
                        )
                    ),
                    "ci_lower": float(
                        np.quantile(values, tail)
                    ),
                    "ci_upper": float(
                        np.quantile(
                            values,
                            1.0 - tail,
                        )
                    ),
                    "confidence": float(confidence),
                    "bootstrap_unit": "original_observation_id",
                    "n_bootstrap": int(n_bootstrap),
                }
            )

    return pd.DataFrame(records)


def run_gp_nested_repeated_oof_benchmark(
    X,
    Y,
    feature_names,
    target_names,
    *,
    output_dir="nested_repeated_oof_surrogate_benchmark",
    repeat_seeds=tuple(range(10)),
    outer_folds=5,
    inner_folds=5,
    fixed_family="IND",
    fixed_kernel=("Matern", 2.5),
    kernel_grid=None,
    icm_rank=2,
    bagging_bags=20,
    noise_floor=1e-4,
    init_noise=1e-3,
    obj_weights=None,
    device=torch.device("cpu"),
    dtype=torch.double,
    bootstrap_reps=2000,
    bootstrap_confidence=0.95,
    save_predictions=True,
    verbose=True,
):
    """
    Surrogate ablation under nested repeated outer CV.

    IMPORTANT:
      * Every original observation receives exactly one OOF prediction in
        every repeat.
      * Metrics are calculated from the complete N-observation OOF set for
        each repeat, not separately on small outer folds.
      * Repeats quantify partition sensitivity only.
      * Dataset-level CIs bootstrap original observation IDs.
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
    repeat_seeds = [int(s) for s in repeat_seeds]

    n_samples, n_targets = Y.shape
    n_repeats = len(repeat_seeds)

    if n_samples < outer_folds:
        raise ValueError("N must be at least outer_folds.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        (output_path / "predictions").mkdir(
            parents=True,
            exist_ok=True,
        )

    methods = [
        "Single GP",
        "Model Selection",
        "Model Selection + Bagging",
    ]

    prediction_store = {
        method: {
            "mean": np.full(
                (n_repeats, n_samples, n_targets),
                np.nan,
                dtype=float,
            ),
            "std": np.full(
                (n_repeats, n_samples, n_targets),
                np.nan,
                dtype=float,
            ),
            "outer_fold": np.full(
                (n_repeats, n_samples),
                -1,
                dtype=int,
            ),
        }
        for method in methods
    }

    repeat_metric_records = []
    selection_records = []
    inner_diagnostic_records = []
    long_oof_records = []

    dataset_location = np.mean(Y, axis=0)
    dataset_scale = np.maximum(
        np.std(Y, axis=0, ddof=0),
        1e-12,
    )

    for repeat_id, partition_seed in enumerate(
        repeat_seeds
    ):
        if verbose:
            print("\n" + "=" * 88)
            print(
                f"Outer repeat {repeat_id + 1}/{n_repeats} "
                f"(partition seed={partition_seed})"
            )
            print("=" * 88)

        outer_splitter = KFold(
            n_splits=int(outer_folds),
            shuffle=True,
            random_state=int(partition_seed),
        )

        for outer_fold, (
            outer_train_idx,
            outer_test_idx,
        ) in enumerate(
            outer_splitter.split(
                np.arange(n_samples)
            )
        ):
            if verbose:
                print(
                    f"\nOuter fold {outer_fold + 1}/{outer_folds}: "
                    f"train={len(outer_train_idx)}, "
                    f"test={len(outer_test_idx)}"
                )

            selection_seed = (
                int(partition_seed) * 10000
                + outer_fold * 100
                + 11
            )

            selected = (
                _inner_cv_select_surrogate_configuration(
                    X[outer_train_idx],
                    Y[outer_train_idx],
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

            selection_records.append(
                {
                    "repeat": int(repeat_id),
                    "partition_seed": int(
                        partition_seed
                    ),
                    "outer_fold": int(outer_fold),
                    "n_outer_train": int(
                        len(outer_train_idx)
                    ),
                    "n_outer_test": int(
                        len(outer_test_idx)
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
                    "best_icm_kernel": (
                        _serialise_kernel(
                            selected[
                                "best_icm_kernel"
                            ]
                        )
                    ),
                    "best_icm_score": float(
                        selected["best_icm_score"]
                    ),
                    "best_ind_kernels": (
                        _serialise_kernel(
                            selected[
                                "best_ind_kernels"
                            ]
                        )
                    ),
                    "best_ind_score": float(
                        selected["best_ind_score"]
                    ),
                }
            )

            diag_df = selected["diagnostics"].copy()
            diag_df.insert(
                0, "outer_fold", int(outer_fold)
            )
            diag_df.insert(
                0,
                "partition_seed",
                int(partition_seed),
            )
            diag_df.insert(
                0, "repeat", int(repeat_id)
            )
            inner_diagnostic_records.extend(
                diag_df.to_dict(orient="records")
            )

            method_settings = [
                (
                    "Single GP",
                    str(fixed_family).upper(),
                    (
                        tuple(fixed_kernel)
                        if str(fixed_family).upper()
                        == "ICM"
                        else [
                            tuple(fixed_kernel)
                            for _ in target_names
                        ]
                    ),
                    0,
                ),
                (
                    "Model Selection",
                    selected["family"],
                    selected["kernel"],
                    0,
                ),
                (
                    "Model Selection + Bagging",
                    selected["family"],
                    selected["kernel"],
                    int(bagging_bags),
                ),
            ]

            for method_id, (
                method,
                family,
                kernel,
                bags,
            ) in enumerate(method_settings):
                model_seed = (
                    int(partition_seed) * 100000
                    + outer_fold * 1000
                    + method_id * 100
                    + 31
                )

                auto = (
                    _fit_fixed_surrogate_configuration(
                        X[outer_train_idx],
                        Y[outer_train_idx],
                        selected_family=family,
                        selected_kernel=kernel,
                        feature_names=feature_names,
                        target_names=target_names,
                        seed=model_seed,
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

                with torch.inference_mode():
                    y_mean, y_std = auto.predict(
                        X[outer_test_idx]
                    )
                y_mean = np.asarray(
                    y_mean, dtype=float
                )
                y_std = np.maximum(
                    np.asarray(y_std, dtype=float),
                    1e-12,
                )

                prediction_store[method]["mean"][
                    repeat_id,
                    outer_test_idx,
                    :
                ] = y_mean
                prediction_store[method]["std"][
                    repeat_id,
                    outer_test_idx,
                    :
                ] = y_std
                prediction_store[method][
                    "outer_fold"
                ][
                    repeat_id,
                    outer_test_idx,
                ] = outer_fold

                del auto
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # One full OOF prediction set of size N has now been created.
        for method in methods:
            mean_r = prediction_store[method][
                "mean"
            ][repeat_id]
            std_r = prediction_store[method][
                "std"
            ][repeat_id]

            if (
                np.isnan(mean_r).any()
                or np.isnan(std_r).any()
            ):
                raise RuntimeError(
                    f"Incomplete OOF predictions for repeat={repeat_id}, "
                    f"method={method}."
                )

            metrics = (
                evaluate_probabilistic_predictions(
                    y_true=Y,
                    y_mean=mean_r,
                    y_std=std_r,
                    target_names=target_names,
                    normalization_location=(
                        dataset_location
                    ),
                    normalization_scale=dataset_scale,
                )
            )

            repeat_metric_records.append(
                {
                    "repeat": int(repeat_id),
                    "partition_seed": int(
                        partition_seed
                    ),
                    "method": method,
                    "n_observations": int(
                        n_samples
                    ),
                    **metrics,
                }
            )

            for observation_id in range(
                n_samples
            ):
                row = {
                    "repeat": int(repeat_id),
                    "partition_seed": int(
                        partition_seed
                    ),
                    "method": method,
                    "observation_id": int(
                        observation_id
                    ),
                    "outer_fold": int(
                        prediction_store[method][
                            "outer_fold"
                        ][
                            repeat_id,
                            observation_id,
                        ]
                    ),
                }
                for j, target in enumerate(
                    target_names
                ):
                    row[
                        f"{target}_true"
                    ] = float(
                        Y[observation_id, j]
                    )
                    row[
                        f"{target}_mean"
                    ] = float(
                        mean_r[observation_id, j]
                    )
                    row[
                        f"{target}_std"
                    ] = float(
                        std_r[observation_id, j]
                    )
                long_oof_records.append(row)

            if verbose:
                print(
                    f"Repeat {repeat_id + 1}: {method:<28} "
                    f"macro NRMSE={metrics['macro_NRMSE']:.5f}, "
                    f"macro R2={metrics['macro_R2']:.5f}"
                )

    metrics_by_repeat = pd.DataFrame(
        repeat_metric_records
    )
    oof_predictions = pd.DataFrame(
        long_oof_records
    )
    selected_models = pd.DataFrame(
        selection_records
    )
    inner_diagnostics = pd.DataFrame(
        inner_diagnostic_records
    )

    excluded = {
        "repeat",
        "partition_seed",
        "n_observations",
    }
    metric_columns = [
        c for c in _find_metric_columns(
            metrics_by_repeat
        )
        if c not in excluded
    ]

    repeat_summary = (
        metrics_by_repeat
        .groupby("method")[metric_columns]
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
    repeat_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in repeat_summary.columns
    ]
    repeat_summary = repeat_summary.reset_index()

    bootstrap_ci = (
        _bootstrap_prediction_metric_cis(
            y_true=Y,
            prediction_store=prediction_store,
            target_names=target_names,
            point_metrics=metrics_by_repeat,
            n_bootstrap=bootstrap_reps,
            confidence=bootstrap_confidence,
            seed=20260901,
        )
    )

    metrics_by_repeat.to_csv(
        output_path
        / "metrics_by_outer_repeat.csv",
        index=False,
    )
    repeat_summary.to_csv(
        output_path
        / "repeat_partition_sensitivity_summary.csv",
        index=False,
    )
    bootstrap_ci.to_csv(
        output_path
        / "observation_bootstrap_confidence_intervals.csv",
        index=False,
    )
    oof_predictions.to_csv(
        output_path
        / "observation_level_oof_predictions.csv",
        index=False,
    )
    selected_models.to_csv(
        output_path
        / "selected_models_by_outer_fold.csv",
        index=False,
    )
    inner_diagnostics.to_csv(
        output_path
        / "inner_cv_candidate_scores.csv",
        index=False,
    )

    if save_predictions:
        for method in methods:
            safe = (
                method.lower()
                .replace(" ", "_")
                .replace("+", "plus")
            )
            np.savez_compressed(
                output_path
                / "predictions"
                / f"{safe}_oof_cube.npz",
                y_true=Y,
                y_mean=(
                    prediction_store[method][
                        "mean"
                    ]
                ),
                y_std=(
                    prediction_store[method][
                        "std"
                    ]
                ),
                outer_fold=(
                    prediction_store[method][
                        "outer_fold"
                    ]
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
        "repeat_seeds": repeat_seeds,
        "n_repeats": len(repeat_seeds),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "independent_observational_units": int(
            n_samples
        ),
        "repeats_interpretation": (
            "partition sensitivity only; not independent replication"
        ),
        "dataset_level_ci_unit": (
            "original_observation_id"
        ),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_confidence": float(
            bootstrap_confidence
        ),
        "fixed_family": str(fixed_family),
        "fixed_kernel": (
            _serialise_kernel(fixed_kernel)
        ),
        "kernel_grid": [
            _serialise_kernel(k)
            for k in kernel_grid
        ],
        "bagging_bags": int(bagging_bags),
        "bagging_total_models": int(
            1 + bagging_bags
        ),
        "feature_names": feature_names,
        "target_names": target_names,
        "normalization_for_reported_normalized_metrics": (
            "fixed full-dataset target mean/std; "
            "used only to scale evaluation metrics, never for model fitting"
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
        repeat_summary,
        bootstrap_ci,
    )
