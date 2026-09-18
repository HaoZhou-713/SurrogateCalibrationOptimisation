
def run_benchmark_seeds(
    *,
    gt_dict,
    make_auto,
    pf_cfg,
    calibration_cfg=None,
    n_train: int = 30,
    seeds=range(10),
    eps_ratio: float = 0.01,
    verbose: bool = False,
    compute_model_metrics: bool = True,
    n_eval_metric: int = 5000,
    target_coverage: float = 0.90,
):
    """
    Original seed runner, but now each seed row also contains
    surrogate accuracy and calibration quality metrics.

    Important:
    - GD/IGD/HV plotting code from the original notebook still works because
      the old columns are preserved.
    - New columns are added for manuscript tables/figures.
    """
    X_pool = np.asarray(gt_dict["X_pool"])
    Y_pool = np.asarray(gt_dict["Y_pool"])
    N_pool = X_pool.shape[0]

    rows = []

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        idx_train = rng.choice(N_pool, size=n_train, replace=False)

        X_train = X_pool[idx_train]
        Y_train = Y_pool[idx_train]

        # External benchmark hold-out points for surrogate/calibration metrics.
        # These are NOT used for training. This keeps metrics paired with the seed.
        X_eval_metric = None
        Y_eval_metric = None
        if compute_model_metrics:
            mask = np.ones(N_pool, dtype=bool)
            mask[idx_train] = False
            idx_left = np.flatnonzero(mask)
            if n_eval_metric is not None and len(idx_left) > n_eval_metric:
                idx_eval = rng.choice(idx_left, size=int(n_eval_metric), replace=False)
            else:
                idx_eval = idx_left
            X_eval_metric = X_pool[idx_eval]
            Y_eval_metric = Y_pool[idx_eval]

        out = evaluate_one_trial(
            make_auto=make_auto,
            X_train=X_train,
            Y_train=Y_train,
            gt_dict=gt_dict,
            pf_cfg=pf_cfg,
            calibration_cfg=calibration_cfg,
            eps_ratio=eps_ratio,
            verbose=verbose,
            X_eval_metric=X_eval_metric,
            Y_eval_metric=Y_eval_metric,
            target_coverage=target_coverage,
            compute_model_metrics=compute_model_metrics,
        )

        out["seed"] = int(seed)
        out["n_train"] = int(n_train)
        rows.append(out)

    df_all = pd.DataFrame(rows)

    # Include both old final-decision metrics and new surrogate/calibration metrics.
    metric_cols = [
        c for c in [
            "RMSE_mean", "nRMSE_mean", "MAE_mean", "nMAE_mean", "R2_mean",
            "Coverage_mean", "Coverage_error_mean", "ECE", "MPIW_mean", "nMPIW_mean",
            "IntervalScore_mean", "nIntervalScore_mean", "NLL_mean",
            "GD", "IGD", "Hausdorff", "Recall", "HV_ratio", "HV_pred", "HV_gt",
            "n_pf_gt", "n_pareto_x"
        ]
        if c in df_all.columns
    ]

    summary = df_all[metric_cols].agg(
        ["mean", "std", "median", "min", "max",
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75)]
    )
    summary.index = ["mean", "std", "median", "min", "max", "q25", "q75"]

    return df_all, summary
