# ==== 1) 生成 FON 数据（n=3, 2 目标）====
def fonseca_fleming(X):
    # X: (N,n) in [-4, 4]
    n = X.shape[1]
    s1 = np.sum((X - 1/np.sqrt(n))**2, axis=1)
    s2 = np.sum((X + 1/np.sqrt(n))**2, axis=1)
    f1 = 1 - np.exp(-s1)
    f2 = 1 - np.exp(-s2)
    return np.column_stack([f1, f2])

lb, ub = -1.0, 1.0