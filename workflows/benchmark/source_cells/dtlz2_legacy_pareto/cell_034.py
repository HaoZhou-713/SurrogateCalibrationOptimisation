import numpy as np

def evaluate_one_trial(
    make_auto,                 # <- 你的建模 factory：auto = make_auto()
    X_train, Y_train,
    gt_dict,
    *,
    pf_cfg,                    # <- surrogate_pareto_front 的设置（必传）
    calibration_cfg=None,      # <- dict: {"enabled":True, "alpha":0.1, "method":"per_target", "source":"test"}
    eps_ratio=0.01,
    verbose=False,
):
    sense   = gt_dict["sense"]
    Y_pf_gt = gt_dict["Y_pf_gt"]

    # 1) build + fit
    auto = make_auto()
    auto.fit(X_train, Y_train)
    # -------- 2) optional: final evaluation on held-out split inside auto --------
    if hasattr(auto, "evaluate_final"):
        _ = auto.evaluate_final(
            X_train, Y_train,
            final_eval_on=final_eval_on,
            scaler_mode=scaler_mode_eval,
            verbose=verbose,
        )

    # -------- 3) finalize (refit for deployment) --------
    if hasattr(auto, "finalize"):
        auto.finalize(
            X_train, Y_train,
            refit_scope=refit_scope,
            scaler_mode=scaler_mode_final,
            set_as_current=True,
            verbose=verbose,
        )

    # 2) calibration (可选)：用 auto 内部 test split 做
    # 你需要自己已有一个函数：calibrate_from_auto_test(auto, alpha, method, ...)
    if calibration_cfg and calibration_cfg.get("enabled", False):
        # source = calibration_cfg.get("source", "test")  # "test" | "train" | "val" | "train_val" | "all"
        alpha  = calibration_cfg.get("alpha", 0.10)
        method = calibration_cfg.get("method", "per_target")
        
        # calibrate_from_auto_test(auto, alpha, method)
        print("Calibrating")
        # calibrate_from_auto_test(auto, alpha=alpha, method=method)  # 你已有/我之前给过版本
        auto.calibration = calibrate_from_cv_plus_up(
            auto = auto,
            X=auto.X_all,
            Y=auto.Y_all,
            K=5,
            alpha=0.10,
            method="per_target",
            model_key="current",   # 或 "current"
            seed=7,
        )
        
        auto.finalize(
            X_train, Y_train,
            refit_scope=refit_scope,
            scaler_mode=scaler_mode_final,
            set_as_current=True,
            verbose=verbose,
        )
    # 3) bounds：默认从 auto 的 x_min/x_span 推断，也允许 pf_cfg 覆盖
    bounds = pf_cfg.get("bounds", None)

    if bounds is None:
        x_min  = auto.x_min.detach().cpu().numpy()
        x_span = auto.x_span.detach().cpu().numpy()
        bounds = np.column_stack([x_min, x_min + x_span])  # (d,2)

    # 4) surrogate PF（setting 全从 pf_cfg 输入）
    #    这里强制把 sense 用 gt_dict 的 sense（避免你忘了同步）
    # res = surrogate_pareto_front_nsga2(
    #     auto,
    #     # bounds,
    #     **{**pf_cfg, "bounds": bounds, "sense": sense},
    # )

    res = surrogate_pareto_front_nsga2(
        auto,
        bounds=pf_cfg["bounds"],
        pop_size=256,
        n_gen=100,
        seed=pf_cfg.get("seed", 0),
        model_key=pf_cfg.get("model_key", "final_models"),
        sense=pf_cfg.get("sense", ["min","min"]),
        use_calibration=pf_cfg.get("use_calibration", True),
        calibrator=pf_cfg.get("calibrator", calibrate_for_auto),
        use_hull=pf_cfg.get("use_hull", False),
        hull_ref=pf_cfg.get("hull_ref", "train_val"),
    )

    # 5) PF_pred：surrogate 已经返回 Pareto set（真值目标）
    X_pf_pred = res.X_pareto
    Y_pf_pred_or = res.Y_pareto
    Y_pf_pred = f_gt(X_pf_pred)          # (Np, M) -- 你确认这是“真值目标”时才这样用

    # 6) 指标统一在 max-space 计算
    Y_pf_pred_max = to_max_space(Y_pf_pred, sense)
    Y_pf_gt_max   = to_max_space(Y_pf_gt, sense)


    rng = Y_pf_gt_max.max(axis=0) - Y_pf_gt_max.min(axis=0)
    eps = eps_ratio * float(np.linalg.norm(rng) + 1e-12)

    out = {
        "GD": GD(Y_pf_pred_max, Y_pf_gt_max),
        "IGD": IGD(Y_pf_pred_max, Y_pf_gt_max),
        "Hausdorff": hausdorff(Y_pf_pred_max, Y_pf_gt_max),
        "Recall": pf_recall(Y_pf_pred_max, Y_pf_gt_max, eps=eps),
    }

    # HV
    Y_pool_max = to_max_space(gt_dict["Y_pool"], sense)
    ref = choose_ref_point_max(Y_pool_max, margin=0.05)
    out["HV_pred"]  = hypervolume_max(Y_pf_pred_max, ref)
    out["HV_gt"]    = hypervolume_max(Y_pf_gt_max, ref)
    out["HV_ratio"] = out["HV_pred"] / (out["HV_gt"] + 1e-12)

    # out["n_pf_pred"] = int(len(idx_pf_pred))
    out["n_pf_gt"]   = int(len(gt_dict["Y_pf_gt"]))
    out["n_pareto_x"] = int(len(X_pf_pred))
    out['pareto_X'] = X_pf_pred
    out['pareto_Y'] = Y_pf_pred_or

    return out, auto
