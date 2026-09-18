# ============================================================
# Utility functions
# ============================================================

def sobol_sample(bounds, n, seed=0):
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    engine = SobolEngine(dimension=d, scramble=True, seed=int(seed))
    X01 = engine.draw(n).cpu().numpy()
    return bounds[:, 0] + X01 * (bounds[:, 1] - bounds[:, 0])


def is_non_dominated_min(Y):
    """
    Return boolean mask of non-dominated points for minimisation.
    O(N^2), fine for candidate/PF sizes used here.
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # point j dominates i if all <= and at least one <
        dominated_by_any = np.any(np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1))
        if dominated_by_any:
            keep[i] = False
    return keep


def is_non_dominated_max(Y):
    """
    Return boolean mask of non-dominated points for maximisation.
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_any = np.any(np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1))
        if dominated_by_any:
            keep[i] = False
    return keep


def to_max_space(Y, sense):
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    for j, s in enumerate(sense):
        if s.lower().startswith("min"):
            out[:, j] = -out[:, j]
    return out


def pairwise_dist(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2))


def GD(Y_pred, Y_ref):
    D = pairwise_dist(Y_pred, Y_ref)
    return float(np.mean(np.min(D, axis=1)))


def IGD(Y_pred, Y_ref):
    D = pairwise_dist(Y_ref, Y_pred)
    return float(np.mean(np.min(D, axis=1)))


def hausdorff(Y_pred, Y_ref):
    D1 = pairwise_dist(Y_pred, Y_ref)
    D2 = pairwise_dist(Y_ref, Y_pred)
    return float(max(np.max(np.min(D1, axis=1)), np.max(np.min(D2, axis=1))))


def pf_recall(Y_pred, Y_ref, eps):
    D = pairwise_dist(Y_ref, Y_pred)
    return float(np.mean(np.min(D, axis=1) <= eps))


def choose_ref_point_max(Y_max, margin=0.05):
    Y_max = np.asarray(Y_max, dtype=float)
    y_min = Y_max.min(axis=0)
    y_max = Y_max.max(axis=0)
    span = y_max - y_min + 1e-12
    return y_min - margin * span


def hypervolume_2d_max(Y_max, ref):
    """
    2D hypervolume for maximisation, with ref dominated by all useful points.
    """
    Y = np.asarray(Y_max, dtype=float)
    ref = np.asarray(ref, dtype=float).reshape(2)

    # keep points that dominate the reference point
    mask = np.all(Y > ref, axis=1)
    Y = Y[mask]
    if len(Y) == 0:
        return 0.0

    # keep max non-dominated points
    Y = Y[is_non_dominated_max(Y)]

    # sort by first objective ascending
    order = np.argsort(Y[:, 0])
    Y = Y[order]

    area = 0.0
    prev_x = ref[0]
    # for non-dominated max front, y decreases as x increases
    for x, y in Y:
        width = max(0.0, x - prev_x)
        height = max(0.0, y - ref[1])
        area += width * height
        prev_x = max(prev_x, x)
    return float(area)


def evaluate_pf_metrics(Y_pred_true, Y_ref_true, Y_pool_true, sense, eps_ratio=0.01):
    """
    Compute final decision quality after evaluating predicted Pareto candidates
    with the true analytical objective.
    """
    Y_pred_max = to_max_space(Y_pred_true, sense)
    Y_ref_max = to_max_space(Y_ref_true, sense)
    Y_pool_max = to_max_space(Y_pool_true, sense)

    # optional max-space non-dominated filtering of predicted true PF
    if len(Y_pred_max) > 0:
        nd = is_non_dominated_max(Y_pred_max)
        Y_pred_max = Y_pred_max[nd]

    rng = Y_ref_max.max(axis=0) - Y_ref_max.min(axis=0)
    eps = eps_ratio * float(np.linalg.norm(rng) + 1e-12)

    ref = choose_ref_point_max(Y_pool_max, margin=0.05)
    hv_pred = hypervolume_2d_max(Y_pred_max, ref)
    hv_gt = hypervolume_2d_max(Y_ref_max, ref)

    return {
        "GD": GD(Y_pred_max, Y_ref_max),
        "IGD": IGD(Y_pred_max, Y_ref_max),
        "Hausdorff": hausdorff(Y_pred_max, Y_ref_max),
        "Recall": pf_recall(Y_pred_max, Y_ref_max, eps=eps),
        "HV_pred": hv_pred,
        "HV_gt": hv_gt,
        "HV_ratio": hv_pred / (hv_gt + 1e-12),
    }


def z_value_for_two_sided_interval(level):
    # hard-coded values avoid scipy dependency
    table = {
        0.10: 0.12566134685507416,
        0.20: 0.2533471031357997,
        0.30: 0.38532046640756773,
        0.40: 0.5244005127080409,
        0.50: 0.6744897501960817,
        0.60: 0.8416212335729143,
        0.70: 1.0364333894937898,
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
    }
    key = round(float(level), 2)
    if key not in table:
        # fallback using torch Normal icdf
        normal = torch.distributions.Normal(0.0, 1.0)
        return float(normal.icdf(torch.tensor((1.0 + level) / 2.0)))
    return table[key]