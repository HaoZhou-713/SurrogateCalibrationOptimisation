# ============================================================
# Benchmark definition: DTLZ2, 2 objectives
# ============================================================

d = 6
FEATURES = [f"x{i+1}" for i in range(d)]
TARGETS  = ["f1", "f2"]

def benchmark_dtlz2(X):
    X = np.asarray(X, float)
    g = np.sum((X[:, 1:] - 0.5) ** 2, axis=1)
    f1 = (1.0 + g) * np.cos(X[:, 0] * np.pi / 2.0)
    f2 = (1.0 + g) * np.sin(X[:, 0] * np.pi / 2.0)
    return np.column_stack([f1, f2])

def f_gt(X):
    return benchmark_dtlz2(X)
