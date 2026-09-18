def benchmark_dtlz2(X):
    X = np.asarray(X, float)
    d = X.shape[1]
    g = np.sum((X[:,1:] - 0.5)**2, axis=1)  # d-1 terms
    f1 = (1 + g) * np.cos(X[:,0] * np.pi/2)
    f2 = (1 + g) * np.sin(X[:,0] * np.pi/2)
    return np.column_stack([f1, f2])
