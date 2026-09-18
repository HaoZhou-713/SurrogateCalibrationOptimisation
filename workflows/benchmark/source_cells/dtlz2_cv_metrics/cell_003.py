# ============================================================
# Ground-truth pool and Pareto-front utilities
# ============================================================

def sobol_in_box(n_dim, N, lb, ub, seed=0):
    eng = SobolEngine(dimension=n_dim, scramble=True, seed=seed)
    U = eng.draw(N).double().cpu().numpy()
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    return lb + (ub - lb) * U

def sense_to_maximization(Y, sense):
    Y = np.asarray(Y, float)
    Ymax = Y.copy()
    for j, s in enumerate(sense):
        sj = str(s).strip().lower()
        if sj == "min":
            Ymax[:, j] = -Ymax[:, j]
        elif sj == "max":
            pass
        else:
            raise ValueError("sense must be 'min' or 'max'")
    return Ymax

def to_max_space(Y, sense):
    return sense_to_maximization(Y, sense)

def gt_pf_dtlz2_with_X(d=6, n_pf=2000):
    # DTLZ2 PF for M=2: x2..xd = 0.5 and x1 sweeps [0,1]
    x1 = np.linspace(0.0, 1.0, n_pf)
    X_pf = np.full((n_pf, d), 0.5, dtype=float)
    X_pf[:, 0] = x1
    Y_pf = benchmark_dtlz2(X_pf)
    return X_pf, Y_pf

def build_gt_dict(*, d, sense, lb, ub, N_pool=30000, seed=0):
    X_pool = sobol_in_box(d, N_pool, lb, ub, seed=seed)
    Y_pool = f_gt(X_pool)

    X_pf_gt, Y_pf_gt = gt_pf_dtlz2_with_X(d=d, n_pf=2000)

    return {
        "X_pool": X_pool,
        "Y_pool": Y_pool,
        "X_pf_gt": X_pf_gt,
        "Y_pf_gt": Y_pf_gt,
        "sense": sense,
    }

def pairwise_dist(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(A2 + B2 - 2 * A @ B.T, 0.0))

def GD(Y_pred, Y_gt):
    D = pairwise_dist(Y_pred, Y_gt)
    return float(np.mean(np.min(D, axis=1)))

def IGD(Y_pred, Y_gt):
    D = pairwise_dist(Y_gt, Y_pred)
    return float(np.mean(np.min(D, axis=1)))

def hausdorff(Y_pred, Y_gt):
    D_pg = pairwise_dist(Y_pred, Y_gt)
    d1 = np.max(np.min(D_pg, axis=1))
    d2 = np.max(np.min(D_pg, axis=0))
    return float(max(d1, d2))

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
