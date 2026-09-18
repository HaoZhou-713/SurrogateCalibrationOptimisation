# ============================================================
# Split conformal and CV+ calibration helpers
# ============================================================
def calibrate_from_split_conformal_up(
    auto,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cal_frac: float = 0.20,
    alpha: float = 0.10,
    method: str = "per_target",
    eps: float = 1e-12,
    model_key: str = "final_models",
    seed: int = 0,
    batch_size: int = 4096,
):
    """
    Split conformal calibration using a held-out subset of the available data.

    This follows the same spirit as the original "Calib" baseline, but is written
    locally so that the notebook does not depend on hidden object attributes.
    It trains a temporary model on the proper-training subset and computes
    normalised residual scores on the calibration subset. The final deployed auto
    object can still be fitted on the full data; only q_vec is attached.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, M = Y.shape
    if N < 5:
        raise ValueError(f"N={N} too small for split conformal calibration")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_cal = max(M + 2, int(round(cal_frac * N)))
    n_cal = min(n_cal, N - 2)
    idx_cal = idx[:n_cal]
    idx_tr = idx[n_cal:]

    auto_split = copy.deepcopy(auto)
    auto_split.final_models = auto_split._train_models_on(
        X[idx_tr],
        Y[idx_tr],
        scaler_mode="frozen",
    )

    mu, sig, _ = predict_auto_safe(auto_split, X[idx_cal], model_key=model_key, batch_size=batch_size)
    Z = np.abs(Y[idx_cal] - mu) / (sig + eps)

    method = method.lower()
    if method == "per_target":
        alpha_j = alpha / M
        q_vec = np.quantile(Z, 1.0 - alpha_j, axis=0)
    elif method == "joint":
        alpha_j = alpha
        s = np.max(Z, axis=1)
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
        "source": "split",
        "cal_frac": float(cal_frac),
        "n_cal": int(len(idx_cal)),
        "seed": int(seed),
        "model_key": model_key,
    }


def calibrate_from_cv_plus_up(
    auto,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    K: int = 5,
    alpha: float = 0.10,
    method: str = "per_target",
    eps: float = 1e-12,
    model_key: str = "final_models",
    seed: int = 0,
    batch_size: int = 4096,
):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, M = Y.shape
    if N < K:
        raise ValueError(f"N={N} must be >= K={K}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    folds = np.array_split(perm, K)

    Z_all = []

    for k in range(K):
        idx_te = folds[k]
        idx_tr = np.concatenate([folds[i] for i in range(K) if i != k])

        auto_k = copy.deepcopy(auto)
        auto_k.final_models = auto_k._train_models_on(
            X[idx_tr],
            Y[idx_tr],
            scaler_mode="frozen",
        )

        mu, sig, _ = predict_auto_safe(auto_k, X[idx_te], model_key=model_key, batch_size=batch_size)
        Zk = np.abs(Y[idx_te] - mu) / (sig + eps)
        Z_all.append(Zk)

    Z = np.vstack(Z_all)

    method = method.lower()
    if method == "per_target":
        alpha_j = alpha / M
        q_vec = np.quantile(Z, 1.0 - alpha_j, axis=0)
    elif method == "joint":
        alpha_j = alpha
        s = np.max(Z, axis=1)
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
