# ============================================================
# Extra metrics and fixed seed runner
# ============================================================

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _fix_bounds(bounds, d=None):
    b = np.asarray(bounds, dtype=float)

    if b.ndim == 2 and b.shape[1] == 2:
        pass
    elif b.ndim == 2 and b.shape[0] == 2 and b.shape[1] != 2:
        b = b.T
    elif b.ndim == 2 and b.shape[0] == 1 and b.shape[1] % 2 == 0:
        b = b.reshape(-1, 2)
    else:
        raise ValueError(f"bounds must have shape (d, 2), got {b.shape}")

    if d is not None and b.shape[0] != d:
        raise ValueError(f"bounds dimension mismatch: expected d={d}, got {b.shape[0]}")
    return b

def _predict_mu_std(auto, X, model_key="final_models", batch_size=4096):
    try:
        pred = auto.predict(X, model_key=model_key, batch_size=batch_size)
    except TypeError:
        try:
            pred = auto.predict(X, model_key=model_key)
        except TypeError:
            pred = auto.predict(X)

    if isinstance(pred, dict):
        mu = pred.get("mean", pred.get("mu", None))
        std = pred.get("std", pred.get("sigma", None))
    elif isinstance(pred, (tuple, list)):
        mu, std = pred[0], pred[1]
    else:
        raise ValueError("Unknown auto.predict output format.")

    if mu is None or std is None:
        raise ValueError("Could not extract mean/std from auto.predict output.")

    mu = _to_numpy(mu).astype(float)
    std = np.maximum(_to_numpy(std).astype(float), 1e-12)
    return mu, std

def compute_surrogate_calibration_metrics(
    auto,
    X_eval,
    Y_eval,
    *,
    model_key="final_models",
    target_coverage=0.90,
):
    Y_true = np.asarray(Y_eval, dtype=float)
    mu, std = _predict_mu_std(auto, X_eval, model_key=model_key)

    if mu.shape != Y_true.shape:
        raise ValueError(f"Prediction shape {mu.shape} != true shape {Y_true.shape}")

    alpha = 1.0 - target_coverage
    n, m = Y_true.shape

    # Accuracy
    err = mu - Y_true
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    y_range = np.ptp(Y_true, axis=0) + 1e-12
    nrmse = rmse / y_range
    nmae = mae / y_range

    ss_res = np.sum((Y_true - mu) ** 2, axis=0)
    ss_tot = np.sum((Y_true - np.mean(Y_true, axis=0)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # Intervals: conformal if auto.calibration exists, otherwise Gaussian nominal interval
    calib = getattr(auto, "calibration", None)
    use_conformal_interval = (
        isinstance(calib, dict)
        and calib.get("enabled", False)
        and ("q_vec" in calib)
    )

    if use_conformal_interval:
        q_vec = np.asarray(calib["q_vec"], dtype=float).reshape(1, -1)
        if q_vec.shape[1] != m:
            raise ValueError(f"q_vec dimension {q_vec.shape[1]} != objectives {m}")
        radius = q_vec * std
        interval_source = "conformal"
        q1 = float(q_vec[0, 0])
        q2 = float(q_vec[0, 1]) if m > 1 else np.nan
    else:
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        radius = z * std
        interval_source = "gaussian"
        q1 = np.nan
        q2 = np.nan

    lo = mu - radius
    hi = mu + radius

    covered = (Y_true >= lo) & (Y_true <= hi)
    coverage = np.mean(covered, axis=0)
    coverage_error = np.abs(coverage - target_coverage)

    width = hi - lo
    mpiw = np.mean(width, axis=0)
    nmpiw = mpiw / y_range

    interval_score = width.copy()
    interval_score += (2.0 / alpha) * (lo - Y_true) * (Y_true < lo)
    interval_score += (2.0 / alpha) * (Y_true - hi) * (Y_true > hi)
    interval_score = np.mean(interval_score, axis=0)
    ninterval_score = interval_score / y_range

    nll = 0.5 * np.log(2.0 * np.pi * std ** 2) + 0.5 * ((Y_true - mu) / std) ** 2
    nll = np.mean(nll, axis=0)

    out = {}
    for j in range(m):
        key = f"f{j+1}"
        out[f"{key}_RMSE"] = float(rmse[j])
        out[f"{key}_nRMSE"] = float(nrmse[j])
        out[f"{key}_MAE"] = float(mae[j])
        out[f"{key}_nMAE"] = float(nmae[j])
        out[f"{key}_R2"] = float(r2[j])
        out[f"{key}_Coverage"] = float(coverage[j])
        out[f"{key}_Coverage_error"] = float(coverage_error[j])
        out[f"{key}_MPIW"] = float(mpiw[j])
        out[f"{key}_nMPIW"] = float(nmpiw[j])
        out[f"{key}_IntervalScore"] = float(interval_score[j])
        out[f"{key}_nIntervalScore"] = float(ninterval_score[j])
        out[f"{key}_NLL"] = float(nll[j])

    out["RMSE_mean"] = float(np.mean(rmse))
    out["nRMSE_mean"] = float(np.mean(nrmse))
    out["MAE_mean"] = float(np.mean(mae))
    out["nMAE_mean"] = float(np.mean(nmae))
    out["R2_mean"] = float(np.mean(r2))
    out["Coverage_mean"] = float(np.mean(coverage))
    out["Coverage_error_mean"] = float(np.mean(coverage_error))
    out["ECE"] = float(np.mean(coverage_error))
    out["MPIW_mean"] = float(np.mean(mpiw))
    out["nMPIW_mean"] = float(np.mean(nmpiw))
    out["IntervalScore_mean"] = float(np.mean(interval_score))
    out["nIntervalScore_mean"] = float(np.mean(ninterval_score))
    out["NLL_mean"] = float(np.mean(nll))
    out["interval_source"] = interval_source
    out["q1"] = q1
    out["q2"] = q2

    return out

def evaluate_one_trial(
    make_auto,
    X_train,
    Y_train,
    gt_dict,
    *,
    pf_cfg,
    calibration_cfg=None,
    eps_ratio=0.01,
    verbose=False,
    compute_model_metrics=True,
    X_eval_metric=None,
    Y_eval_metric=None,
    n_eval_metric=5000,
    target_coverage=0.90,
):
    sense = gt_dict["sense"]
    Y_pf_gt = gt_dict["Y_pf_gt"]

    # 1. Fit + finalise selected/bagged surrogate
    auto = make_auto()
    auto.fit(X_train, Y_train)

    if hasattr(auto, "evaluate_final"):
        try:
            _ = auto.evaluate_final(
                X_train,
                Y_train,
                final_eval_on=globals().get("final_eval_on", "test"),
                scaler_mode=globals().get("scaler_mode_eval", "frozen"),
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print("[Warning] evaluate_final failed:", repr(e))

    if hasattr(auto, "finalize"):
        auto.finalize(
            X_train,
            Y_train,
            refit_scope=globals().get("refit_scope", "full"),
            scaler_mode=globals().get("scaler_mode_final", "frozen"),
            set_as_current=True,
            verbose=verbose,
        )

    # 2. CV+ or split calibration
    if calibration_cfg and calibration_cfg.get("enabled", False):
        cal_type = calibration_cfg.get("type", calibration_cfg.get("calibration_type", "cvplus")).lower()
        alpha = calibration_cfg.get("alpha", 0.10)
        method = calibration_cfg.get("method", "per_target")
        model_key_cal = calibration_cfg.get("model_key", pf_cfg.get("model_key", "final_models"))

        if cal_type in ["cvplus", "cv+"]:
            print("Calibrating with CV+")
            auto.calibration = calibrate_from_cv_plus(
                auto=auto,
                X=getattr(auto, "X_all", X_train),
                Y=getattr(auto, "Y_all", Y_train),
                K=calibration_cfg.get("K", 5),
                alpha=alpha,
                method=method,
                model_key=model_key_cal,
                seed=calibration_cfg.get("seed", 7),
            )
        elif cal_type in ["split", "split_cp", "split conformal"]:
            print("Calibrating with Split CP")
            auto.calibration = calibrate_from_auto_test(
                auto,
                alpha=alpha,
                method=method,
            )
        else:
            raise ValueError(f"Unknown calibration type: {cal_type}")

        if isinstance(auto.calibration, dict):
            auto.calibration["enabled"] = True

    # 3. Surrogate accuracy + calibration metrics
    out = {}
    if compute_model_metrics:
        if X_eval_metric is None or Y_eval_metric is None:
            X_eval_metric = np.asarray(gt_dict["X_pool"])
            Y_eval_metric = np.asarray(gt_dict["Y_pool"])

            if len(X_eval_metric) > n_eval_metric:
                rng = np.random.default_rng(12345)
                idx = rng.choice(len(X_eval_metric), size=n_eval_metric, replace=False)
                X_eval_metric = X_eval_metric[idx]
                Y_eval_metric = Y_eval_metric[idx]

        metric_out = compute_surrogate_calibration_metrics(
            auto,
            X_eval_metric,
            Y_eval_metric,
            model_key=pf_cfg.get("model_key", "final_models"),
            target_coverage=target_coverage,
        )
        out.update(metric_out)

    # 4. Pareto search by NSGA-II
    bounds = pf_cfg.get("bounds", None)
    if bounds is None:
        x_min = auto.x_min.detach().cpu().numpy().reshape(-1)
        x_span = auto.x_span.detach().cpu().numpy().reshape(-1)
        bounds = np.column_stack([x_min, x_min + x_span])
    else:
        bounds = _fix_bounds(bounds, d=X_train.shape[1])

    # IMPORTANT:
    # surrogate_pareto_front_nsga2 does NOT accept sampler/n_candidates.
    # Therefore we call it explicitly with only compatible keywords.
    res = surrogate_pareto_front_nsga2(
        auto,
        bounds=bounds,
        pop_size=pf_cfg.get("pop_size", 128),
        n_gen=pf_cfg.get("n_gen", 100),
        seed=pf_cfg.get("seed", 0),
        model_key=pf_cfg.get("model_key", "final_models"),
        sense=pf_cfg.get("sense", sense),
        use_calibration=pf_cfg.get("use_calibration", False),
        calibrator=pf_cfg.get("calibrator", calibrate_for_auto),
        use_hull=pf_cfg.get("use_hull", False),
        hull_ref=pf_cfg.get("hull_ref", "all"),
        hull_tol=pf_cfg.get("hull_tol", 1e-9),
    )

    X_pf_pred = res.X_pareto
    Y_pf_pred = f_gt(X_pf_pred)

    Y_pf_pred_max = to_max_space(Y_pf_pred, sense)
    Y_pf_gt_max = to_max_space(Y_pf_gt, sense)

    rng = Y_pf_gt_max.max(axis=0) - Y_pf_gt_max.min(axis=0)
    eps = eps_ratio * float(np.linalg.norm(rng) + 1e-12)

    out["GD"] = GD(Y_pf_pred_max, Y_pf_gt_max)
    out["IGD"] = IGD(Y_pf_pred_max, Y_pf_gt_max)
    out["Hausdorff"] = hausdorff(Y_pf_pred_max, Y_pf_gt_max)
    out["Recall"] = pf_recall(Y_pf_pred_max, Y_pf_gt_max, eps=eps)

    Y_pool_max = to_max_space(gt_dict["Y_pool"], sense)
    ref = choose_ref_point_max(Y_pool_max, margin=0.05)

    out["HV_pred"] = hypervolume_max(Y_pf_pred_max, ref)
    out["HV_gt"] = hypervolume_max(Y_pf_gt_max, ref)
    out["HV_ratio"] = out["HV_pred"] / (out["HV_gt"] + 1e-12)

    out["n_pf_gt"] = int(len(gt_dict["Y_pf_gt"]))
    out["n_pareto_x"] = int(len(X_pf_pred))
    out["actual_best_family"] = str(getattr(auto, "best_family", None))
    out["actual_best_kernel"] = str(getattr(auto, "best_kernel", None))
    out["has_calibration"] = bool(
        isinstance(getattr(auto, "calibration", None), dict)
        and auto.calibration.get("enabled", False)
    )

    return out, auto

def run_benchmark_seeds(
    *,
    gt_dict,
    make_auto,
    pf_cfg,
    calibration_cfg=None,
    n_train=70,
    seeds=range(20),
    eps_ratio=0.01,
    verbose=False,
    compute_model_metrics=True,
    n_eval_metric=5000,
    target_coverage=0.90,
    method_name=None,
    output_dir="results/dtlz2_cvplus_metrics_fixed",
    csv_name="dtlz2_cvplus_results_with_surrogate_calibration_metrics.csv",
):
    rows = []
    autos = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_pool = np.asarray(gt_dict["X_pool"])
    Y_pool = np.asarray(gt_dict["Y_pool"])
    N_pool = X_pool.shape[0]

    for seed in seeds:
        print(f"Running seed={seed}")
        t0 = time.time()

        rng = np.random.default_rng(int(seed))
        idx_train = rng.choice(N_pool, size=n_train, replace=False)
        X_train = X_pool[idx_train]
        Y_train = Y_pool[idx_train]

        # external evaluation subset for metrics, not used for training
        if N_pool > n_eval_metric:
            idx_eval = rng.choice(N_pool, size=n_eval_metric, replace=False)
            X_eval_metric = X_pool[idx_eval]
            Y_eval_metric = Y_pool[idx_eval]
        else:
            X_eval_metric = X_pool
            Y_eval_metric = Y_pool

        try:
            out, auto = evaluate_one_trial(
                make_auto=make_auto,
                X_train=X_train,
                Y_train=Y_train,
                gt_dict=gt_dict,
                pf_cfg=pf_cfg,
                calibration_cfg=calibration_cfg,
                eps_ratio=eps_ratio,
                verbose=verbose,
                compute_model_metrics=compute_model_metrics,
                X_eval_metric=X_eval_metric,
                Y_eval_metric=Y_eval_metric,
                n_eval_metric=n_eval_metric,
                target_coverage=target_coverage,
            )

            out["seed"] = int(seed)
            out["n_train"] = int(n_train)
            out["method"] = method_name if method_name is not None else (
                "CV+" if calibration_cfg and calibration_cfg.get("enabled", False) else "Raw"
            )
            out["failed"] = False
            out["total_time_sec"] = time.time() - t0
            rows.append(out)
            autos[int(seed)] = auto

        except Exception as e:
            print(f"[FAILED] seed={seed}: {repr(e)}")
            traceback.print_exc()
            rows.append({
                "seed": int(seed),
                "n_train": int(n_train),
                "method": method_name,
                "failed": True,
                "error": repr(e),
                "total_time_sec": time.time() - t0,
            })

        gc.collect()

    df_all = pd.DataFrame(rows)
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    if "method" in df_all.columns and len(df_all) > 0:
        summary = df_all.groupby("method")[numeric_cols].agg(
            ["mean", "std", "median", q25, q75, "min", "max"]
        )
    else:
        summary = df_all[numeric_cols].agg(["mean", "std", "median", q25, q75, "min", "max"]).T

    csv_path = output_dir / csv_name
    summary_path = output_dir / csv_name.replace(".csv", "_summary.csv")
    df_all.to_csv(csv_path, index=False)
    summary.to_csv(summary_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")

    return df_all, summary, autos
