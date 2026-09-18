# ============================================================
# DTLZ2 analytical function and true Pareto front
# ============================================================
def benchmark_dtlz2(X):
    X = np.asarray(X, dtype=float)
    g = np.sum((X[:, 1:] - 0.5) ** 2, axis=1)
    f1 = (1.0 + g) * np.cos(X[:, 0] * np.pi / 2.0)
    f2 = (1.0 + g) * np.sin(X[:, 0] * np.pi / 2.0)
    return np.column_stack([f1, f2])


def f_gt(X):
    return benchmark_dtlz2(X)


def gt_pf_dtlz2_with_X(d=6, n_pf=2000):
    x1 = np.linspace(0.0, 1.0, n_pf)
    X_pf = np.full((n_pf, d), 0.5, dtype=float)
    X_pf[:, 0] = x1
    Y_pf = benchmark_dtlz2(X_pf)
    return X_pf, Y_pf


def sobol_in_box(n_dim, N, lb, ub, seed=0):
    eng = SobolEngine(dimension=n_dim, scramble=True, seed=int(seed))
    U = eng.draw(int(N)).double().cpu().numpy()
    lb = np.asarray(lb, float).reshape(-1)
    ub = np.asarray(ub, float).reshape(-1)
    return lb + (ub - lb) * U


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


gt_dict = build_gt_dict(d=d, sense=sense, lb=lb, ub=ub, N_pool=N_POOL, seed=0)

print("X_pool:", gt_dict["X_pool"].shape)
print("Y_pool:", gt_dict["Y_pool"].shape)
print("Y_pf_gt:", gt_dict["Y_pf_gt"].shape)
