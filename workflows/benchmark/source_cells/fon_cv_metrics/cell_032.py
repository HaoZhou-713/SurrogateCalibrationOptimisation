
import numpy as np


def evaluate_one_trial(
    make_auto,
    X_train, Y_train,
    gt_dict,
    *,
    pf_cfg,
    calibration_cfg=None,
    eps_ratio=0.01,
    verbose=False,
    X_eval_metric=None,
    Y_eval_metric=None,
    target_coverage=0.90,
    compute_model_metrics=True,
):
    """
    Original CaseStudy trial runner + extra surrogate/calibration metrics.

    It keeps the original final Pareto metrics unchanged:
    GD, IGD, Hausdorff, Recall, HV_ratio.

    Extra metrics are added to the same output row:
    RMSE/nRMSE/MAE/R2, Coverage/ECE/MPIW/IntervalScore/NLL.
    """
    sense   = gt_dict["sense"]
    Y_pf_gt = gt_dict["Y_pf_gt"]

    # 1) build + fit
    auto = make_auto()
    auto.fit(X_train, Y_train)

    # 2) optional internal final evaluation, same as your original code
    if hasattr(auto, "evaluate_final"):
        _ = auto.evaluate_final(
            X_train, Y_train,
            final_eval_on=final_eval_on,
            scaler_mode=scaler_mode_eval,
            verbose=verbose,
        )

    # 3) finalize for deployment, same as your original code
    if hasattr(auto, "finalize"):
        auto.finalize(
            X_train, Y_train,
            refit_scope=refit_scope,
            scaler_mode=scaler_mode_final,
            set_as_current=True,
            verbose=verbose,
        )

    # 4) optional CV+ calibration, same mechanism as your original code
    if calibration_cfg and calibration_cfg.get("enabled", False):
        alpha  = calibration_cfg.get("alpha", 0.10)
        method = calibration_cfg.get("method", "per_target")
        K      = calibration_cfg.get("K", 5)
        seed_c = calibration_cfg.get("seed", 7)
        model_key_cal = calibration_cfg.get("model_key", "current")

        if verbose:
            print("Calibrating")

        auto.calibration = calibrate_from_cv_plus(
            auto=auto,
            X=auto.X_all,
            Y=auto.Y_all,
            K=K,
            alpha=alpha,
            method=method,
            model_key=model_key_cal,
            seed=seed_c,
        )

        # Keep your original behaviour: refit after calibration for final surrogate use
        if hasattr(auto, "finalize"):
            auto.finalize(
                X_train, Y_train,
                refit_scope=refit_scope,
                scaler_mode=scaler_mode_final,
                set_as_current=True,
                verbose=verbose,
            )

    # 5) surrogate accuracy + calibration quality on external benchmark hold-out pool
    out = {}
    if compute_model_metrics:
        if X_eval_metric is None or Y_eval_metric is None:
            # fallback: use all pool points. In run_benchmark_seeds below we use non-training points instead.
            X_eval_metric = gt_dict["X_pool"]
            Y_eval_metric = gt_dict["Y_pool"]
        metric_model_key = pf_cfg.get("model_key", "final_models")
        out.update(
            compute_surrogate_calibration_metrics(
                auto,
                X_eval_metric,
                Y_eval_metric,
                gt_dict=gt_dict,
                target_names=getattr(auto, "target_names", TARGETS),
                target_coverage=target_coverage,
                model_key=metric_model_key,
            )
        )

    # 6) bounds: preserve your original logic
    bounds = pf_cfg.get("bounds", None)
    if bounds is None:
        x_min  = auto.x_min.detach().cpu().numpy()
        x_span = auto.x_span.detach().cpu().numpy()
        bounds = np.column_stack([x_min, x_min + x_span])

    # 7) original surrogate Pareto front search
    res = surrogate_pareto_front(
        auto,
        **{**pf_cfg, "bounds": bounds, "sense": sense},
    )

    # 8) evaluate selected Pareto candidates by TRUE benchmark function
    X_pf_pred = res.X_pareto
    Y_pf_pred = f_gt(X_pf_pred)

    # 9) original final decision metrics in max-space
    Y_pf_pred_max = to_max_space(Y_pf_pred, sense)
    Y_pf_gt_max   = to_max_space(Y_pf_gt, sense)

    rng = Y_pf_gt_max.max(axis=0) - Y_pf_gt_max.min(axis=0)
    eps = eps_ratio * float(np.linalg.norm(rng) + 1e-12)

    out.update({
        "GD": GD(Y_pf_pred_max, Y_pf_gt_max),
        "IGD": IGD(Y_pf_pred_max, Y_pf_gt_max),
        "Hausdorff": hausdorff(Y_pf_pred_max, Y_pf_gt_max),
        "Recall": pf_recall(Y_pf_pred_max, Y_pf_gt_max, eps=eps),
    })

    Y_pool_max = to_max_space(gt_dict["Y_pool"], sense)
    ref = choose_ref_point_max(Y_pool_max, margin=0.05)
    out["HV_pred"]  = hypervolume_max(Y_pf_pred_max, ref)
    out["HV_gt"]    = hypervolume_max(Y_pf_gt_max, ref)
    out["HV_ratio"] = out["HV_pred"] / (out["HV_gt"] + 1e-12)

    out["n_pf_gt"] = int(len(gt_dict["Y_pf_gt"]))
    out["n_pareto_x"] = int(len(X_pf_pred))
    out["pareto_X"] = X_pf_pred

    # Some useful metadata for later filtering
    out["calibration_enabled"] = bool(calibration_cfg and calibration_cfg.get("enabled", False))
    out["model_metric_key"] = str(pf_cfg.get("model_key", "final_models"))

    return out
