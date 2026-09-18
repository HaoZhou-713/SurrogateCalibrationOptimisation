# 你自己的 make_auto
def make_auto(seed):
    cfg = AutoConfig(
        seed=int(seed),
        test_frac=0.15,
        val_frac=0.15,
        use_ks_for_test=False,
        bags=20,
        device=torch.device("cpu"),
        dtype=torch.double,
    )
    return AutoMultiOutputGP(
        feature_names=FEATURES,
        target_names=TARGETS,
        config=cfg
    )

# 读数据
df = pd.read_excel(DATA_PATH)
X = df[FEATURES].to_numpy(float)
Y = df[TARGETS].to_numpy(float)

df_metrics, pf_store = multi_seed_pf_compare(
    make_auto,
    X, Y,
    seeds=range(10),
    alpha=0.10,
    K=5,
    pop_size=256,
    n_gen=150,
)

print(df_metrics.groupby("method")[["HV_mu","spread","n_pf"]].agg(["mean","std"]))

# plot_pf_overlay(pf_store, title="Battery dataset PF overlay (surrogate mean)")

# violin_compare(df_metrics, "HV_mu", "HV (surrogate mean space) — raw vs calibrated")
# violin_compare(df_metrics, "spread", "PF spread — raw vs calibrated")
