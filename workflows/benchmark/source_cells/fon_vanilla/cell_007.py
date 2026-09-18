# ============================================================
# Run Vanilla GP benchmark
# ============================================================

def run_one_seed(seed):
    t0 = time.time()
    rng = np.random.default_rng(int(seed))

    # Same sampling style as the existing CaseStudy notebooks:
    # draw training points from the benchmark ground-truth pool.
    idx_train = rng.choice(len(X_pool), size=N_TRAIN, replace=False)
    X_train = X_pool[idx_train]
    Y_train = Y_pool[idx_train]

    # Independent RBF GP per objective
    model = VanillaIndependentRBFGP(
        train_iters=TRAIN_ITERS,
        lr=LR,
        device=device,
        dtype=dtype,
        verbose=False,
    )
    model.fit(X_train, Y_train)

    # Surrogate/calibration metrics on an independent subset of the pool
    idx_eval = rng.choice(len(X_pool), size=min(N_EVAL_METRIC, len(X_pool)), replace=False)
    X_eval = X_pool[idx_eval]
    Y_eval = Y_pool[idx_eval]
    mu_eval, std_eval = model.predict(X_eval)

    out = surrogate_accuracy_and_calibration_metrics(
        y_true=Y_eval,
        mu=mu_eval,
        std=std_eval,
        target_coverage=TARGET_COVERAGE,
    )

    # Surrogate Pareto search using predicted means only
    X_cand = sobol_sample(BOUNDS, N_CANDIDATES, seed=10000 + int(seed))
    mu_cand, std_cand = model.predict(X_cand)

    # Vanilla uses mean-only predicted objective for minimisation
    nd_pred = is_non_dominated_min(mu_cand)
    X_pf_pred = X_cand[nd_pred]

    # Avoid a huge PF if the candidate set is very dense
    if len(X_pf_pred) > MAX_PARETO_POINTS:
        # Keep evenly-spaced points after sorting by the first predicted objective
        mu_pf = mu_cand[nd_pred]
        order = np.argsort(mu_pf[:, 0])
        keep = np.linspace(0, len(order) - 1, MAX_PARETO_POINTS).astype(int)
        X_pf_pred = X_pf_pred[order[keep]]

    Y_pf_pred_true = f_gt(X_pf_pred)

    final_metrics = evaluate_pf_metrics(
        Y_pred_true=Y_pf_pred_true,
        Y_ref_true=Y_pf_ref,
        Y_pool_true=Y_pool,
        sense=SENSE,
        eps_ratio=EPS_RATIO,
    )
    out.update(final_metrics)

    out["method"] = "Vanilla GP"
    out["seed"] = int(seed)
    out["n_train"] = int(N_TRAIN)
    out["n_candidates"] = int(N_CANDIDATES)
    out["n_pareto_x"] = int(len(X_pf_pred))
    out["actual_best_family"] = "IND"
    out["actual_best_kernel"] = "[RBF, RBF]"
    out["actual_bags"] = 1
    out["use_model_selection"] = False
    out["use_bagging"] = False
    out["calibration_type"] = "none"
    out["total_time_sec"] = float(time.time() - t0)

    return out


rows = []
for seed in SEEDS:
    print(f"Running {BENCHMARK_NAME} true Vanilla GP seed={seed}")
    try:
        rows.append(run_one_seed(seed))
    except Exception as e:
        print(f"[FAILED] seed={seed}: {repr(e)}")
        rows.append({
            "method": "Vanilla GP",
            "seed": int(seed),
            "failed": True,
            "error": repr(e),
        })
    gc.collect()

df = pd.DataFrame(rows)

csv_path = RESULT_DIR / f"{BENCHMARK_NAME.lower()}_true_vanilla_gp_{len(SEEDS)}seeds.csv"
summary_path = RESULT_DIR / f"{BENCHMARK_NAME.lower()}_true_vanilla_gp_{len(SEEDS)}seeds_summary.csv"

df.to_csv(csv_path, index=False)

numeric_cols = df.select_dtypes(include=[np.number]).columns

# Add q25/q75 for every numeric metric in the summary table.
def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


summary = df.groupby("method")[numeric_cols].agg(["mean", "std", "median", q25, q75, "min", "max"])
summary.to_csv(summary_path)

print("Saved:", csv_path)
print("Saved:", summary_path)

display(df.head())
display(summary.T)