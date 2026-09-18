# ============================================================
# Execute: run all five methods together
# ============================================================
final_eval_on = "test"
scaler_mode_eval = "frozen"
refit_scope = "full"
scaler_mode_final = "frozen"

all_dfs = []
all_summaries = []

for method_cfg in METHOD_CONFIGS:
    method_name = method_cfg["name"]
    cal_type = method_cfg["calibration_type"]
    use_cal = bool(method_cfg["use_calibration"])
    make_auto_method = make_auto_factory(method_cfg)

    print("\n" + "=" * 90)
    print(f"Running method: {method_name}")
    print("=" * 90)

    pf_cfg = dict(
        sense=sense,
        use_hull=False,
        hull_ref="all",
        use_calibration=use_cal,
        calibrator=calibrate_for_auto,
        sampler="sobol",
        model_key="final_models",
        bounds=bounds,
        pop_size=NSGA2_POP_SIZE,
        n_gen=NSGA2_N_GEN,
        seed=0,
    )

    calibration_cfg = None
    if use_cal:
        calibration_cfg = {
            "enabled": True,
            "type": cal_type,              # "split" or "cvplus"
            "alpha": 1.0 - TARGET_COVERAGE,
            "method": "per_target",
            "K": K_FOLDS,
            "cal_frac": SPLIT_CAL_FRAC,
            "seed": 7,
        }

    df_method, summary_method = run_benchmark_seeds(
        gt_dict=gt_dict,
        make_auto=make_auto_method,
        pf_cfg=pf_cfg,
        calibration_cfg=calibration_cfg,
        n_train=N_TRAIN,
        seeds=SEEDS,
        n_eval_metric=N_EVAL_METRIC,
        target_coverage=TARGET_COVERAGE,
        eps_ratio=0.01,
        method_name=method_name,
        verbose=False,
    )

    all_dfs.append(df_method)
    summary_method = summary_method.copy()
    summary_method.insert(0, "Method", method_name)
    all_summaries.append(summary_method)

# Combined seed-level dataframe
df_all = pd.concat(all_dfs, ignore_index=True)

# Method-wise summary for paper tables
def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

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

method_order = [m["name"] for m in METHOD_CONFIGS]
df_all["Method"] = pd.Categorical(df_all["Method"], categories=method_order, ordered=True)
df_all = df_all.sort_values(["Method", "seed"]).reset_index(drop=True)

summary_by_method = df_all.groupby("Method", observed=False)[metric_cols].agg(["median", q25, q75, "mean", "std"])

display(df_all.head())
display(summary_by_method)
