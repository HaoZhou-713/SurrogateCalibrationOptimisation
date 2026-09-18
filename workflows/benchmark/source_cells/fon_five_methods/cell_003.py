# ============================================================
# FON analytical function and true Pareto front
# ============================================================
def benchmark_fon(X):
    X = np.asarray(X, dtype=float)
    d_local = X.shape[1]
    a = 1.0 / np.sqrt(d_local)
    s1 = np.sum((X - a) ** 2, axis=1)
    s2 = np.sum((X + a) ** 2, axis=1)
    f1 = 1.0 - np.exp(-s1)
    f2 = 1.0 - np.exp(-s2)
    return np.column_stack([f1, f2])


def f_gt(X):
    return benchmark_fon(X)


def gt_pf_fon_with_X(d=4, n_pf=2000):
    # Pareto set lies on x_1 = ... = x_d = t, t in [-1/sqrt(d), 1/sqrt(d)].
    a = 1.0 / np.sqrt(d)
    t = np.linspace(-a, a, n_pf)
    X_pf = np.tile(t.reshape(-1, 1), (1, d))
    Y_pf = benchmark_fon(X_pf)
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

    X_pf_gt, Y_pf_gt = gt_pf_fon_with_X(d=d, n_pf=2000)

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
