def make_auto(test_frac=0.1):
    cfg = AutoConfig(
        seed=7,
        test_frac=test_frac,
        val_frac=0.15,
        use_ks_for_test=False,
        bags=10,              # 测试用 2，正式再开 10
        device=device,
        dtype=dtype,
    )
    auto = AutoMultiOutputGP(
        feature_names=FEATURES,
        target_names=TARGETS,
        config=cfg,
    )
    return auto