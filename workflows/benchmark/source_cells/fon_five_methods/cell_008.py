# ============================================================
# One trial: fit -> optional calibration -> metrics -> Pareto search
# ============================================================
def evaluate_one_trial(
    make_auto,
    X_train,
    Y_train,
    gt_dict,
    *,
    pf_cfg,
    calibration_cfg=None,
    X_eval_metric=None,
    Y_eval_metric=None,
    target_coverage=0.90,
    eps_ratio=0.01,
    verbose=False,
):
    sense = gt_dict["sense"]
    Y_pf_gt = gt_dict["Y_pf_gt"]

    start_total = time.perf_counter()

    # 1) fit
    start_fit = time.perf_counter()
    auto = make_auto()
    auto.fit(X_train, Y_train)

    if hasattr(auto, "evaluate_final"):
        _ = auto.evaluate_final(
            X_train,
            Y_train,
            final_eval_on=final_eval_on,
            scaler_mode=scaler_mode_eval,
            verbose=verbose,
        )

    if hasattr(auto, "finalize"):
        auto.finalize(
            X_train,
            Y_train,
            refit_scope=refit_scope,
            scaler_mode=scaler_mode_final,
            set_as_current=True,
            verbose=verbose,
        )

    fit_time = time.perf_counter() - start_fit

    # 2) optional calibration: none / split CP / CV+
    start_cal = time.perf_counter()
    calibration_type_used = "none"
    if calibration_cfg and calibration_cfg.get("enabled", False):
        alpha = calibration_cfg.get("alpha", 0.10)
        method = calibration_cfg.get("method", "per_target")
        cal_type = calibration_cfg.get("type", "cvplus").lower()
        calibration_type_used = cal_type

        if cal_type in ["split", "split_cp", "calib"]:
            if verbose:
                print("Calibrating with split conformal...")
            auto.calibration = calibrate_from_split_conformal_up(
                auto=auto,
                X=auto.X_all,
                Y=auto.Y_all,
                cal_frac=calibration_cfg.get("cal_frac", SPLIT_CAL_FRAC),
                alpha=alpha,
                method=method,
                model_key=pf_cfg.get("model_key", "final_models"),
                seed=calibration_cfg.get("seed", 7),
            )

        elif cal_type in ["cvplus", "cv+"]:
            if verbose:
                print("Calibrating with CV+...")
            auto.calibration = calibrate_from_cv_plus_up(
                auto=auto,
                X=auto.X_all,
                Y=auto.Y_all,
                K=calibration_cfg.get("K", K_FOLDS),
                alpha=alpha,
                method=method,
                model_key=pf_cfg.get("model_key", "final_models"),
                seed=calibration_cfg.get("seed", 7),
            )
        else:
            raise ValueError(f"Unknown calibration type: {cal_type}")

        # Refit final model after computing calibration scores, preserving original workflow.
        if hasattr(auto, "finalize"):
            auto.finalize(
                X_train,
                Y_train,
                refit_scope=refit_scope,
                scaler_mode=scaler_mode_final,
                set_as_current=True,
                verbose=verbose,
            )

    cal_time = time.perf_counter() - start_cal

    # 3) surrogate/calibration metrics
    metric_out = {}
    if X_eval_metric is not None and Y_eval_metric is not None:
        metric_out = compute_surrogate_calibration_metrics(
            auto,
            X_eval_metric,
            Y_eval_metric,
            target_names=TARGETS,
            model_key=pf_cfg.get("model_key", "final_models"),
            target_coverage=target_coverage,
            use_calibrated_interval=True,
        )

    # 4) bounds
    bounds_local = pf_cfg.get("bounds", None)
    if bounds_local is None:
        x_min = auto.x_min.detach().cpu().numpy().reshape(-1)
        x_span = auto.x_span.detach().cpu().numpy().reshape(-1)
        bounds_local = np.column_stack([x_min, x_min + x_span])
    bounds_local = fix_bounds_shape(bounds_local, d=X_train.shape[1])

    # 5) surrogate Pareto search
    start_opt = time.perf_counter()
    res = surrogate_pareto_front_nsga2(
        auto,
        bounds=bounds_local,
        pop_size=pf_cfg.get("pop_size", NSGA2_POP_SIZE),
        n_gen=pf_cfg.get("n_gen", NSGA2_N_GEN),
        seed=pf_cfg.get("seed", 0),
        model_key=pf_cfg.get("model_key", "final_models"),
        sense=pf_cfg.get("sense", sense),
        use_calibration=pf_cfg.get("use_calibration", False),
        calibrator=pf_cfg.get("calibrator", calibrate_for_auto),
        use_hull=pf_cfg.get("use_hull", False),
        hull_ref=pf_cfg.get("hull_ref", "train_val"),
    )
    opt_time = time.perf_counter() - start_opt

    # 6) evaluate selected Pareto candidates by TRUE benchmark function
    X_pf_pred = np.asarray(res.X_pareto, float)
    Y_pf_pred_true = f_gt(X_pf_pred)

    Y_pf_pred_max = to_max_space(Y_pf_pred_true, sense)
    Y_pf_gt_max = to_max_space(Y_pf_gt, sense)

    rng = Y_pf_gt_max.max(axis=0) - Y_pf_gt_max.min(axis=0)
    eps = eps_ratio * float(np.linalg.norm(rng) + 1e-12)

    out = {}
    out.update(metric_out)
    out.update({
        "GD": GD(Y_pf_pred_max, Y_pf_gt_max),
        "IGD": IGD(Y_pf_pred_max, Y_pf_gt_max),
        "Hausdorff": hausdorff(Y_pf_pred_max, Y_pf_gt_max),
        "Recall": pf_recall(Y_pf_pred_max, Y_pf_gt_max, eps=eps),
        "n_pf_gt": int(len(gt_dict["Y_pf_gt"])),
        "n_pareto_x": int(len(X_pf_pred)),
    })

    Y_pool_max = to_max_space(gt_dict["Y_pool"], sense)
    ref = choose_ref_point_max(Y_pool_max, margin=0.05)
    out["HV_pred"] = hypervolume_max(Y_pf_pred_max, ref)
    out["HV_gt"] = hypervolume_max(Y_pf_gt_max, ref)
    out["HV_ratio"] = out["HV_pred"] / (out["HV_gt"] + 1e-12)

    out["fit_time_sec"] = fit_time
    out["calibration_time_sec"] = cal_time
    out["optimisation_time_sec"] = opt_time
    out["total_time_sec"] = time.perf_counter() - start_total

    return out
