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
):
    """
    Run benchmark by varying random seeds only (single design).

    Parameters
    ----------
    gt_dict : dict
        Must contain X_pool, Y_pool, Y_pf_gt, sense
    make_auto : callable
        Factory returning a fresh auto model
    pf_cfg : dict
        surrogate_pareto_front configuration
    calibration_cfg : dict or None
        Calibration configuration (or None to disable)
    n_train : int
        Number of training samples per seed
    seeds : iterable
        Random seeds controlling training set selection
    eps_ratio : float
        Recall epsilon scaling
    verbose : bool

    Returns
    -------
    df_all : pd.DataFrame
        One row per seed with all metrics
    summary : pd.Series
        Mean / std across seeds
    """

    X_pool = np.asarray(gt_dict["X_pool"])
    Y_pool = np.asarray(gt_dict["Y_pool"])
    N_pool = X_pool.shape[0]

    rows = []

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(N_pool, size=n_train, replace=False)

        X_train = X_pool[idx]
        Y_train = Y_pool[idx]

        out = evaluate_one_trial(
            make_auto=make_auto,
            X_train=X_train,
            Y_train=Y_train,
            gt_dict=gt_dict,
            pf_cfg=pf_cfg,
            calibration_cfg=calibration_cfg,
            eps_ratio=eps_ratio,
            verbose=verbose,
        )

        out["seed"] = int(seed)
        out["n_train"] = int(n_train)
        rows.append(out)

    df_all = pd.DataFrame(rows)

    # 你关心的核心指标
    metric_cols = [
        c for c in
        ["GD", "IGD", "Hausdorff", "Recall", "HV_ratio", "HV_pred", "HV_gt",
         "n_pf_pred", "n_pf_gt", "n_pareto_x"]
        if c in df_all.columns
    ]

    # summary = df_all[metric_cols].agg(["mean", "std", "median", "min", "max"])
    summary = df_all[metric_cols].agg(
    ["mean", "std", "median", "min", "max",
     lambda x: x.quantile(0.25),
     lambda x: x.quantile(0.75)]
    )

    # 给 lambda 行改个好名字
    summary.index = ["mean", "std", "median", "min", "max", "q25", "q75"]

    return df_all, summary
