def calibrate_from_cv_plus_up(
    auto,                   # factory: make_auto() -> fresh auto
    X: np.ndarray,
    Y: np.ndarray,
    *,
    K: int = 5,
    alpha: float = 0.10,
    method: str = "per_target",   # "per_target" | "joint"
    eps: float = 1e-12,
    model_key: str = "final_models",   # which models inside each fold-auto to use
    seed: int = 0,
    batch_size: int = 4096,
):
    """
    CV+ / cross-fitted conformal calibration.

    Returns a calibration dict compatible with auto.calibration.
    """

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X must be (N,d), Y must be (N,M)")

    N, M = Y.shape
    if N < K:
        raise ValueError(f"N={N} must be >= K={K}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    folds = np.array_split(perm, K)

    Z_all = []   # standardized residuals from all folds

    for k in range(K):
        idx_te = folds[k]
        idx_tr = np.concatenate([folds[i] for i in range(K) if i != k])

        auto_k = copy.deepcopy(auto)
        # auto_k.fit(X[idx_tr], Y[idx_tr])
        # auto_k.eval()
        # auto_k.finalize(
        #     X[idx_tr], Y[idx_tr], refit_scope="full", scaler_mode = "refit"
        # )
        auto_k.final_models = auto_k._train_models_on(X[idx_tr], Y[idx_tr], scaler_mode="frozen")

        with torch.inference_mode():
            MU, SIG = auto_k.predict(
                X[idx_te],
                model_key=model_key,
                batch_size=batch_size,
            )

        MU = np.asarray(MU, float)
        SIG = np.asarray(SIG, float)
        Yk  = Y[idx_te]

        Zk = np.abs(Yk - MU) / (SIG + eps)   # (n_k, M)
        Z_all.append(Zk)

    Z = np.vstack(Z_all)   # (N, M)  ← out-of-fold residuals

    # ---- conformal quantile ----
    method = method.lower()
    if method == "per_target":
        alpha_j = alpha / M
        q_vec = np.quantile(Z, 1.0 - alpha_j, axis=0)
    elif method == "joint":
        alpha_j = alpha
        s = np.max(Z, axis=1)   # (N,)
        q = float(np.quantile(s, 1.0 - alpha))
        q_vec = np.full(M, q, dtype=float)
    else:
        raise ValueError("method must be 'per_target' or 'joint'")

    return {
        "enabled": True,
        "method": method,
        "alpha": float(alpha),
        "alpha_j": float(alpha_j),
        "q_vec": np.asarray(q_vec, float),
        "source": "cv+",
        "K": int(K),
        "n_cal": int(N),
        "seed": int(seed),
        "model_key": model_key,
    }
