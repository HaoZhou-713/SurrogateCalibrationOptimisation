# ============================================================
# Run DTLZ2 selected + bagging + CV+
# ============================================================

# Problem setup
d = 6
sense = ["min", "min"]
lb = np.zeros(d)
ub = np.ones(d)
bounds = np.column_stack([lb, ub])

gt_dict = build_gt_dict(
    d=d,
    sense=sense,
    lb=lb,
    ub=ub,
    N_pool=30000,
    seed=0,
)

# Switch
flag_calibration = True

calibration_cfg = {
    "enabled": flag_calibration,
    "type": "cvplus",
    "alpha": 0.10,
    "method": "per_target",
    "K": 5,
    "model_key": "final_models",
    "seed": 7,
}

# NSGA-II Pareto search config.
# Do NOT include sampler/n_candidates here because surrogate_pareto_front_nsga2
# does not accept them.
pf_cfg = dict(
    sense=sense,
    use_hull=False,
    hull_ref="all",
    hull_tol=1e-9,
    use_calibration=flag_calibration,
    calibrator=calibrate_for_auto,
    model_key="final_models",
    bounds=bounds,
    pop_size=128,
    n_gen=100,
    seed=0,
)

final_eval_on = "test"
scaler_mode_eval = "frozen"
refit_scope = "full"
scaler_mode_final = "frozen"

for n in [30, 50, 90, 110]:

    df_all, summary, autos = run_benchmark_seeds(
        gt_dict=gt_dict,
        make_auto=make_auto,
        pf_cfg=pf_cfg,
        calibration_cfg=calibration_cfg,
        n_train=n,
        seeds=range(20),          # change to range(1) for a quick test
        eps_ratio=0.01,
        verbose=False,
        compute_model_metrics=True,
        n_eval_metric=5000,
        target_coverage=0.90,
        method_name="Selected + Bagging + CV+",
        output_dir="results/dtlz2_samples_sensitivity",
        csv_name=f"dtlz2_{n}.csv",
    )

display(df_all.head())
display(summary)
