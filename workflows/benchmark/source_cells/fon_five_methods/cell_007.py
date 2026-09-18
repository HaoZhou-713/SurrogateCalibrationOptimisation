# ============================================================
# Auto surrogate factory for five ablation methods
# ============================================================
def build_auto_config(**kwargs):
    """Build AutoConfig robustly across slightly different local versions.

    If your local `surrogate_pipeline.AutoConfig` already supports
    `use_model_selection`, `fixed_family`, and `fixed_kernel`, they will be
    passed through normally. If not, they are attached after construction so
    updated `AutoMultiOutputGP` implementations can still read them.
    """
    try:
        sig = inspect.signature(AutoConfig)
        accepted = set(sig.parameters.keys())
    except Exception:
        accepted = set()

    init_kwargs = {k: v for k, v in kwargs.items() if (not accepted or k in accepted)}
    try:
        cfg = AutoConfig(**init_kwargs)
    except TypeError:
        # Fall back to the minimal fields known to exist in the original case-study code.
        minimal_keys = ["seed", "test_frac", "val_frac", "use_ks_for_test", "bags", "device", "dtype"]
        cfg = AutoConfig(**{k: kwargs[k] for k in minimal_keys if k in kwargs})

    for k, v in kwargs.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            pass
    return cfg


def make_auto_factory(method_cfg, *, test_frac=0.1):
    """Return a fresh AutoMultiOutputGP factory for one method configuration."""
    def _factory():
        cfg = build_auto_config(
            seed=7,
            test_frac=test_frac,
            val_frac=0.15,
            use_ks_for_test=False,
            bags=int(method_cfg.get("bags", 0)),
            device=device,
            dtype=dtype,
            use_model_selection=bool(method_cfg.get("use_model_selection", True)),
            fixed_family=method_cfg.get("fixed_family", "IND"),
            fixed_kernel=method_cfg.get("fixed_kernel", ("RBF", None)),
            kernel_grid=[
                ("RBF", None),
                ("Matern", 0.5),
                ("Matern", 1.5),
                ("Matern", 2.5),
            ],
        )
        auto = AutoMultiOutputGP(
            feature_names=FEATURES,
            target_names=TARGETS,
            config=cfg,
        )
        return auto
    return _factory
