# ============================================================
# Surrogate and calibration metrics
# ============================================================
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_auto_safe(auto, X, *, model_key="final_models", batch_size=4096):
    # Try several model keys because different notebook versions use different names.
    candidate_keys = []
    if model_key is not None:
        candidate_keys.append(model_key)
    candidate_keys += ["final_models", "current", "eval_models"]

    last_err = None
    for key in candidate_keys:
        try:
            with torch.inference_mode():
                mu, sig = auto.predict(X, model_key=key, batch_size=batch_size)
            mu = _as_numpy(mu).astype(float)
            sig = _as_numpy(sig).astype(float)
            sig = np.maximum(sig, 1e-12)
            return mu, sig, key
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"auto.predict failed for all model_key options. Last error: {last_err}")


def _safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-15:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def compute_surrogate_calibration_metrics(
    auto,
    X_eval,
    Y_eval,
    *,
    target_names,
    model_key="final_models",
    target_coverage=0.90,
    use_calibrated_interval=True,
):
    X_eval = np.asarray(X_eval, float)
    Y_eval = np.asarray(Y_eval, float)

    mu, sig, used_key = predict_auto_safe(auto, X_eval, model_key=model_key)

    M = Y_eval.shape[1]
    obj_range = np.ptp(Y_eval, axis=0)
    obj_range = np.where(obj_range <= 1e-12, 1.0, obj_range)

    err = mu - Y_eval
    abs_err = np.abs(err)

    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(abs_err, axis=0)
    nrmse = rmse / obj_range
    nmae = mae / obj_range
    r2 = np.array([_safe_r2(Y_eval[:, j], mu[:, j]) for j in range(M)])

    alpha = 1.0 - target_coverage
    q_vec = None
    if use_calibrated_interval and hasattr(auto, "calibration"):
        cal = getattr(auto, "calibration", None)
        if isinstance(cal, dict) and cal.get("enabled", False) and "q_vec" in cal:
            q_vec = np.asarray(cal["q_vec"], float).reshape(-1)

    if q_vec is None or len(q_vec) != M:
        q = float(norm.ppf(0.5 + target_coverage / 2.0))
        q_vec = np.full(M, q, dtype=float)

    lower = mu - q_vec.reshape(1, -1) * sig
    upper = mu + q_vec.reshape(1, -1) * sig

    inside = (Y_eval >= lower) & (Y_eval <= upper)
    coverage = np.mean(inside, axis=0)
    mpiw = np.mean(upper - lower, axis=0)
    nmpiw = mpiw / obj_range

    coverage_error = np.abs(coverage - target_coverage)
    ece = float(np.mean(coverage_error))

    interval_width = upper - lower
    below = Y_eval < lower
    above = Y_eval > upper
    interval_score = interval_width.copy()
    interval_score += (2.0 / alpha) * (lower - Y_eval) * below
    interval_score += (2.0 / alpha) * (Y_eval - upper) * above
    iscore = np.mean(interval_score, axis=0)
    niscore = iscore / obj_range

    var = np.maximum(sig ** 2, 1e-12)
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (Y_eval - mu) ** 2 / var)
    nll_obj = np.mean(nll, axis=0)

    z = (Y_eval - mu) / sig
    crps = sig * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    crps_obj = np.mean(crps, axis=0)

    out = {
        "predict_model_key": used_key,
        "RMSE_mean": float(np.mean(rmse)),
        "nRMSE_mean": float(np.mean(nrmse)),
        "MAE_mean": float(np.mean(mae)),
        "nMAE_mean": float(np.mean(nmae)),
        "R2_mean": float(np.nanmean(r2)),
        "Coverage_mean": float(np.mean(coverage)),
        "Coverage_error_mean": float(np.mean(coverage_error)),
        "ECE": ece,
        "MPIW_mean": float(np.mean(mpiw)),
        "nMPIW_mean": float(np.mean(nmpiw)),
        "IntervalScore_mean": float(np.mean(iscore)),
        "nIntervalScore_mean": float(np.mean(niscore)),
        "NLL_mean": float(np.mean(nll_obj)),
        "CRPS_mean": float(np.mean(crps_obj)),
    }

    for j, name in enumerate(target_names):
        out[f"{name}_RMSE"] = float(rmse[j])
        out[f"{name}_nRMSE"] = float(nrmse[j])
        out[f"{name}_MAE"] = float(mae[j])
        out[f"{name}_nMAE"] = float(nmae[j])
        out[f"{name}_R2"] = float(r2[j])
        out[f"{name}_Coverage"] = float(coverage[j])
        out[f"{name}_Coverage_error"] = float(coverage_error[j])
        out[f"{name}_MPIW"] = float(mpiw[j])
        out[f"{name}_nMPIW"] = float(nmpiw[j])
        out[f"{name}_IntervalScore"] = float(iscore[j])
        out[f"{name}_nIntervalScore"] = float(niscore[j])
        out[f"{name}_NLL"] = float(nll_obj[j])
        out[f"{name}_CRPS"] = float(crps_obj[j])

    return out


def fix_bounds_shape(bounds, d=None):
    b = np.asarray(bounds, dtype=float)
    if b.ndim == 2 and b.shape[0] == 2 and b.shape[1] != 2:
        b = b.T
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {b.shape}")
    if d is not None and b.shape[0] != d:
        raise ValueError(f"bounds dimension mismatch: expected d={d}, got {b.shape[0]}")
    return b
