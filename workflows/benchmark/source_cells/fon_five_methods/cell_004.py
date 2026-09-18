# ============================================================
# Pareto metrics
# ============================================================
def to_max_space(Y, sense):
    Y = np.asarray(Y, dtype=float).copy()
    for j, s in enumerate(sense):
        if str(s).lower() == "min":
            Y[:, j] = -Y[:, j]
        elif str(s).lower() == "max":
            pass
        else:
            raise ValueError("sense entries must be 'min' or 'max'")
    return Y


def pairwise_dist(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0))


def GD(Y_pred, Y_gt):
    D = pairwise_dist(Y_pred, Y_gt)
    return float(np.mean(np.min(D, axis=1)))


def IGD(Y_pred, Y_gt):
    D = pairwise_dist(Y_gt, Y_pred)
    return float(np.mean(np.min(D, axis=1)))


def hausdorff(Y_pred, Y_gt):
    D = pairwise_dist(Y_pred, Y_gt)
    return float(max(np.max(np.min(D, axis=1)), np.max(np.min(D, axis=0))))


def pf_recall(Y_pred, Y_gt, eps):
    D = pairwise_dist(Y_gt, Y_pred)
    covered = np.min(D, axis=1) <= eps
    return float(np.mean(covered))


def hypervolume_max(Ymax_pf, ref_point_max):
    Yt = torch.as_tensor(Ymax_pf, dtype=torch.double)
    ref = torch.as_tensor(ref_point_max, dtype=torch.double)
    hv = Hypervolume(ref_point=ref)
    return float(hv.compute(Yt))


def choose_ref_point_max(Ymax_pool, margin=0.05):
    ymin = Ymax_pool.min(axis=0)
    ymax = Ymax_pool.max(axis=0)
    return ymin - margin * (ymax - ymin)
