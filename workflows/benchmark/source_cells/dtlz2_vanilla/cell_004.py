# ============================================================
# DTLZ2 benchmark definition
# ============================================================

BENCHMARK_NAME = "DTLZ2"
D = 6
N_TRAIN = 50
SEEDS = list(range(20))   # change to range(10) if needed

N_POOL = 30000
N_EVAL_METRIC = 5000
N_CANDIDATES = 30000
MAX_PARETO_POINTS = 1000
N_REF_PF = 1000

TARGET_COVERAGE = 0.90
EPS_RATIO = 0.01

TRAIN_ITERS = 140
LR = 0.08

SENSE = ["min", "min"]

lb = np.zeros(D)
ub = np.ones(D)
BOUNDS = np.column_stack([lb, ub])


def f_gt(X):
    """
    Two-objective DTLZ2 with D decision variables.
    Minimisation objectives.
    """
    X = np.asarray(X, dtype=float)
    g = np.sum((X[:, 1:] - 0.5) ** 2, axis=1)
    f1 = (1.0 + g) * np.cos(0.5 * np.pi * X[:, 0])
    f2 = (1.0 + g) * np.sin(0.5 * np.pi * X[:, 0])
    return np.column_stack([f1, f2])


# Ground-truth pool
X_pool = sobol_sample(BOUNDS, N_POOL, seed=0)
Y_pool = f_gt(X_pool)

# Analytical reference Pareto front: g=0, x2...xd=0.5, x1 in [0,1]
x1 = np.linspace(0.0, 1.0, N_REF_PF)
X_pf_ref = np.full((N_REF_PF, D), 0.5)
X_pf_ref[:, 0] = x1
Y_pf_ref = f_gt(X_pf_ref)

print("X_pool:", X_pool.shape)
print("Y_pool:", Y_pool.shape)
print("Reference PF:", Y_pf_ref.shape)