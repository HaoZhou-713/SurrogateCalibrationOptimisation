# ============================================================
# FON benchmark definition
# ============================================================

BENCHMARK_NAME = "FON"
D = 4
N_TRAIN = 30
SEEDS = list(range(20))   # change to range(10) if needed

N_POOL = 30000
N_EVAL_METRIC = 5000
N_CANDIDATES = 30000
MAX_PARETO_POINTS = 1000
N_REF_PF = 1000

TARGET_COVERAGE = 0.90
EPS_RATIO = 0.01

TRAIN_ITERS = 120
LR = 0.08

SENSE = ["min", "min"]

# This matches your existing CaseStudy FON setup.
lb = -np.ones(D)
ub = np.ones(D)
BOUNDS = np.column_stack([lb, ub])


def f_gt(X):
    X = np.asarray(X, dtype=float)
    a = 1.0 / np.sqrt(X.shape[1])
    f1 = 1.0 - np.exp(-np.sum((X - a) ** 2, axis=1))
    f2 = 1.0 - np.exp(-np.sum((X + a) ** 2, axis=1))
    return np.column_stack([f1, f2])


# Ground-truth pool
X_pool = sobol_sample(BOUNDS, N_POOL, seed=0)
Y_pool = f_gt(X_pool)

# Analytical reference Pareto front: x_i = t, t in [-1/sqrt(d), 1/sqrt(d)]
t = np.linspace(-1.0 / np.sqrt(D), 1.0 / np.sqrt(D), N_REF_PF)
X_pf_ref = np.tile(t[:, None], (1, D))
Y_pf_ref = f_gt(X_pf_ref)

print("X_pool:", X_pool.shape)
print("Y_pool:", Y_pool.shape)
print("Reference PF:", Y_pf_ref.shape)