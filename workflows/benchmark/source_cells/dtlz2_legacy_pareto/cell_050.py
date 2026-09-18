device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 
dtype  = torch.double

X_pool = gt_dict["X_pool"]
Y_pool = gt_dict["Y_pool"]

flag_calibration = True

pf_cfg = dict(
    sense=['min','min'],
    use_hull=False,
    hull_ref="all",
    use_calibration=flag_calibration,     # 第一次先关掉 calibration，减少变量
    calibrator=calibrate_for_auto,
    sampler="sobol",
    n_candidates=100000,         # 测试先小一点
    model_key="final_models",
    bounds = np.tile(np.array([0.0, 1.0]), (d, 1))
)

calibration_cfg = {
    "enabled": flag_calibration,
    "alpha": 0.1,
    "method": "per_target",
    # "source": "test",
}

final_eval_on = "test"     # "train" | "train_val" | "test" | "all"
scaler_mode_eval = "frozen"  # 推荐 frozen（口径与选参一致）
refit_scope = "full"         # "train_val" | "full"
scaler_mode_final = "refit"  # 部署模型通常 refit 更稳（也可 frozen）

df_all, summary = run_benchmark_seeds(
    gt_dict=gt_dict,
    make_auto=make_auto,
    pf_cfg=pf_cfg,
    calibration_cfg=calibration_cfg,
    n_train=70,
    seeds=range(20),
    eps_ratio=0.01,
    verbose=False,
)