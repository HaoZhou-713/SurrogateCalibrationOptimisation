(
    metrics_by_repeat,
    repeat_summary,
    bootstrap_ci,
) = run_gp_nested_repeated_oof_benchmark(
    X=df[FEATURES].values,
    Y=df[TARGETS].values,
    feature_names=FEATURES,
    target_names=TARGETS,
    output_dir="results/li_nested_repeated_oof/surrogate_ablation",

    # Repeated OUTER CV. Repeats are partition-sensitivity analyses,
    # not independent physical replications.
    repeat_seeds=range(10),
    outer_folds=5,

    # Nested model-selection CV inside each outer-training set.
    inner_folds=5,

    fixed_family="IND",
    fixed_kernel=("Matern", 2.5),

    kernel_grid=[
        ("RBF", None),
        ("Matern", 0.5),
        ("Matern", 1.5),
        ("Matern", 2.5),
    ],

    # Same bagging construction as before:
    # 1 full-data base model + 20 bootstrap members.
    bagging_bags=20,

    # Dataset-level CI resamples ORIGINAL observation IDs.
    bootstrap_reps=2000,
    bootstrap_confidence=0.95,

    device=torch.device("cpu"),
    dtype=torch.double,
    save_predictions=True,
    verbose=True,
)
