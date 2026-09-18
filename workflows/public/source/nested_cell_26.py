(
    metrics_by_repeat,
    calibration_summary,
    reliability_data,
    reliability_summary,
    bootstrap_ci,
) = run_model_selection_bagging_nested_repeated_calibration(
    X=df[FEATURES].values,
    Y=df[TARGETS].values,
    feature_names=FEATURES,
    target_names=TARGETS,

    output_dir="results/verma_nested_repeated_oof/calibration",

    # Repeated outer CV: each repeat produces N OOF intervals.
    repeat_seeds=range(10),
    outer_folds=5,

    # Surrogate configuration selection inside outer-training only.
    inner_folds=5,

    # Main 90% interval.
    alpha=0.10,

    # Existing cross-fitted calibration, now entirely inside outer-training.
    calibration_folds=5,

    # Existing holdout-calibration comparator.
    calibration_frac=0.20,

    # Each objective separately targets nominal coverage.
    conformal_mode="per_target",

    # Same Model Selection + Bagging construction as before.
    bags=20,

    kernel_grid=[
        ("RBF", None),
        ("Matern", 0.5),
        ("Matern", 1.5),
        ("Matern", 2.5),
    ],

    icm_rank=2,

    nominal_coverages=[
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ],

    # Observation-level dataset CI.
    bootstrap_reps=2000,
    bootstrap_confidence=0.95,

    device=torch.device("cpu"),
    dtype=torch.double,

    save_predictions=True,
    verbose=True,
)
