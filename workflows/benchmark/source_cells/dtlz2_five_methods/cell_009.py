# ============================================================
# Run 10 seeds
# ============================================================
def run_benchmark_seeds(
    *,
    gt_dict,
    make_auto,
    pf_cfg,
    calibration_cfg=None,
    n_train: int = 50,
    seeds=range(10),
    n_eval_metric: int = 5000,
    target_coverage: float = 0.90,
    eps_ratio: float = 0.01,
    method_name: str = "CV+",
    verbose: bool = False,
):
    X_pool = np.asarray(gt_dict["X_pool"])
    Y_pool = np.asarray(gt_dict["Y_pool"])
    N_pool = X_pool.shape[0]

    rows = []

    for seed in seeds:
        print(f"Running {BENCHMARK_NAME} seed={seed}, method={method_name}")

        rng = np.random.default_rng(int(seed))
        idx_train = rng.choice(N_pool, size=n_train, replace=False)

        mask = np.ones(N_pool, dtype=bool)
        mask[idx_train] = False
        idx_rest = np.where(mask)[0]

        if n_eval_metric is not None and len(idx_rest) > n_eval_metric:
            idx_eval = rng.choice(idx_rest, size=n_eval_metric, replace=False)
        else:
            idx_eval = idx_rest

        X_train = X_pool[idx_train]
        Y_train = Y_pool[idx_train]
        X_eval_metric = X_pool[idx_eval]
        Y_eval_metric = Y_pool[idx_eval]

        try:
            out = evaluate_one_trial(
                make_auto=make_auto,
                X_train=X_train,
                Y_train=Y_train,
                gt_dict=gt_dict,
                pf_cfg=pf_cfg,
                calibration_cfg=calibration_cfg,
                X_eval_metric=X_eval_metric,
                Y_eval_metric=Y_eval_metric,
                target_coverage=target_coverage,
                eps_ratio=eps_ratio,
                verbose=verbose,
            )

            out["benchmark"] = BENCHMARK_NAME
            out["Method"] = method_name
            out["seed"] = int(seed)
            out["n_train"] = int(n_train)
            out["n_eval_metric"] = int(len(idx_eval))
            rows.append(out)

        except Exception as exc:
            print(f"[FAILED] seed={seed}: {repr(exc)}")
            rows.append({
                "benchmark": BENCHMARK_NAME,
                "Method": method_name,
                "seed": int(seed),
                "n_train": int(n_train),
                "failed": True,
                "error": repr(exc),
            })

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_all = pd.DataFrame(rows)

    metric_cols = [
        c for c in [
            "RMSE_mean", "nRMSE_mean", "MAE_mean", "nMAE_mean", "R2_mean",
            "Coverage_mean", "Coverage_error_mean", "ECE",
            "MPIW_mean", "nMPIW_mean", "IntervalScore_mean", "nIntervalScore_mean",
            "NLL_mean", "CRPS_mean",
            "GD", "IGD", "Hausdorff", "Recall", "HV_ratio", "HV_pred", "HV_gt",
            "n_pf_gt", "n_pareto_x",
            "fit_time_sec", "calibration_time_sec", "optimisation_time_sec", "total_time_sec",
        ]
        if c in df_all.columns
    ]

    summary = df_all[metric_cols].agg([
        "mean", "std", "median", "min", "max",
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
    ])
    summary.index = ["mean", "std", "median", "min", "max", "q25", "q75"]

    return df_all, summary
