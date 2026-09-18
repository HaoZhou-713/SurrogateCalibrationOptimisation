
# ============================================================
# Extra benchmark surrogate/calibration metrics
# Paste this cell before evaluate_one_trial, then replace
# evaluate_one_trial and run_benchmark_seeds with the extended versions below.
# ============================================================
import numpy as np

try:
    from scipy.stats import norm
    _Z_FOR_COVERAGE = lambda cov: float(norm.ppf(0.5 + 0.5 * cov))
except Exception:
    # fallback for 90% two-sided intervals
    _Z_FOR_COVERAGE = lambda cov: 1.6448536269514722 if abs(cov - 0.90) < 1e-12 else 1.959963984540054


def _safe_np(a):
    """Convert tensors/lists to a float numpy array."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype(float)
    except Exception:
        pass
    return np.asarray(a, dtype=float)


def _predict_mu_sigma(auto, X, *, model_key="final_models", batch_size=4096):
    """Robust wrapper around auto.predict for different versions of AutoMultiOutputGP."""
    try:
        mu, sig = auto.predict(X, model_key=model_key, batch_size=batch_size)
    except TypeError:
        try:
            mu, sig = auto.predict(X, model_key=model_key)
        except TypeError:
            mu, sig = auto.predict(X)
    mu = _safe_np(mu)
    sig = _safe_np(sig)
    if mu.ndim == 1:
        mu = mu[:, None]
    if sig.ndim == 1:
        sig = sig[:, None]
    sig = np.maximum(sig, 1e-12)
    return mu, sig


def _get_q_vec_for_intervals(auto, n_obj, target_coverage=0.90, calibration=None):
    """
    Get the interval multiplier q.
    - Raw GP interval: z-score for the requested two-sided target coverage.
    - CV+/conformal interval: use auto.calibration['q_vec'] if available.
    """
    if calibration is None:
        calibration = getattr(auto, "calibration", None)

    if isinstance(calibration, dict) and calibration.get("enabled", False):
        for key in ["q_vec", "q", "Q", "quantile"]:
            if key in calibration:
                q = np.asarray(calibration[key], dtype=float)
                if q.ndim == 0:
                    return np.full(n_obj, float(q))
                if q.size == n_obj:
                    return q.reshape(-1)
                return np.full(n_obj, float(np.ravel(q)[0]))

    z = _Z_FOR_COVERAGE(target_coverage)
    return np.full(n_obj, z, dtype=float)


def _r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def compute_surrogate_calibration_metrics(
    auto,
    X_eval,
    Y_eval,
    *,
    gt_dict=None,
    target_names=None,
    target_coverage=0.90,
    model_key="final_models",
    batch_size=4096,
):
    """
    Compute surrogate accuracy + calibration quality on a benchmark hold-out pool.

    Returns a flat dict that can be merged into the same row as GD/IGD/HV.
    This keeps your old plotting code valid while adding extra columns.
    """
    X_eval = np.asarray(X_eval, dtype=float)
    Y_eval = np.asarray(Y_eval, dtype=float)
    if Y_eval.ndim == 1:
        Y_eval = Y_eval[:, None]

    MU, SIG = _predict_mu_sigma(auto, X_eval, model_key=model_key, batch_size=batch_size)
    M = Y_eval.shape[1]

    if target_names is None:
        target_names = getattr(auto, "target_names", [f"obj{j+1}" for j in range(M)])
    target_names = list(target_names)

    # objective range for normalised metrics
    if gt_dict is not None and "Y_pool" in gt_dict:
        Y_ref = np.asarray(gt_dict["Y_pool"], dtype=float)
    else:
        Y_ref = Y_eval
    if Y_ref.ndim == 1:
        Y_ref = Y_ref[:, None]
    y_range = np.ptp(Y_ref, axis=0)
    y_range = np.maximum(y_range, 1e-12)

    err = MU - Y_eval
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    nrmse = rmse / y_range
    nmae = mae / y_range
    r2 = _r2_score_np(Y_eval, MU)

    # calibrated or raw interval
    q_vec = _get_q_vec_for_intervals(auto, M, target_coverage=target_coverage)
    lower = MU - SIG * q_vec.reshape(1, -1)
    upper = MU + SIG * q_vec.reshape(1, -1)
    width = upper - lower

    covered = (Y_eval >= lower) & (Y_eval <= upper)
    coverage = covered.mean(axis=0)
    coverage_error = np.abs(coverage - target_coverage)
    mpiw = width.mean(axis=0)
    nmpiw = mpiw / y_range

    alpha = max(1.0 - float(target_coverage), 1e-12)
    interval_score = width + (2.0 / alpha) * (lower - Y_eval) * (Y_eval < lower) + (2.0 / alpha) * (Y_eval - upper) * (Y_eval > upper)
    interval_score = interval_score.mean(axis=0)
    n_interval_score = interval_score / y_range

    # Gaussian negative log likelihood based on the reported sigma.
    # For CV+ this is still a diagnostic based on surrogate sigma, not a conformal likelihood.
    nll = 0.5 * np.log(2.0 * np.pi * SIG ** 2) + 0.5 * ((Y_eval - MU) / SIG) ** 2
    nll = nll.mean(axis=0)

    out = {
        "n_eval_metric": int(len(X_eval)),
        "target_coverage": float(target_coverage),
        "calib_q_mean": float(np.mean(q_vec)),
        "RMSE_mean": float(np.mean(rmse)),
        "nRMSE_mean": float(np.mean(nrmse)),
        "MAE_mean": float(np.mean(mae)),
        "nMAE_mean": float(np.mean(nmae)),
        "R2_mean": float(np.mean(r2)),
        "Coverage_mean": float(np.mean(coverage)),
        "Coverage_error_mean": float(np.mean(coverage_error)),
        "ECE": float(np.mean(coverage_error)),
        "MPIW_mean": float(np.mean(mpiw)),
        "nMPIW_mean": float(np.mean(nmpiw)),
        "IntervalScore_mean": float(np.mean(interval_score)),
        "nIntervalScore_mean": float(np.mean(n_interval_score)),
        "NLL_mean": float(np.mean(nll)),
    }

    for j, name in enumerate(target_names):
        # make column names safe and compact
        nm = str(name).replace(" ", "_").replace("/", "_")
        out[f"{nm}_RMSE"] = float(rmse[j])
        out[f"{nm}_nRMSE"] = float(nrmse[j])
        out[f"{nm}_MAE"] = float(mae[j])
        out[f"{nm}_nMAE"] = float(nmae[j])
        out[f"{nm}_R2"] = float(r2[j])
        out[f"{nm}_Coverage"] = float(coverage[j])
        out[f"{nm}_Coverage_error"] = float(coverage_error[j])
        out[f"{nm}_MPIW"] = float(mpiw[j])
        out[f"{nm}_nMPIW"] = float(nmpiw[j])
        out[f"{nm}_IntervalScore"] = float(interval_score[j])
        out[f"{nm}_nIntervalScore"] = float(n_interval_score[j])
        out[f"{nm}_NLL"] = float(nll[j])

    return out
