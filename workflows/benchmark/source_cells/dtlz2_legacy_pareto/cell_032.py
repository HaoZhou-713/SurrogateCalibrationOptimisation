import numpy as np

def gt_pf_dtlz2_with_X(d=6, n_pf=2000):
    """
    Construct Pareto-optimal X and Y for DTLZ2 with M=2.
    Domain: x in [0,1]^d.
    PF achieved by setting x2..xd = 0.5 (=> g=0), sweeping x1 in [0,1].
    """
    x1 = np.linspace(0.0, 1.0, n_pf)
    X_pf = np.full((n_pf, d), 0.5, dtype=float)
    X_pf[:, 0] = x1

    # DTLZ2 (M=2) objectives
    g = np.sum((X_pf[:, 1:] - 0.5)**2, axis=1)  # should be all zeros
    f1 = (1.0 + g) * np.cos(X_pf[:, 0] * np.pi/2)
    f2 = (1.0 + g) * np.sin(X_pf[:, 0] * np.pi/2)
    Y_pf = np.column_stack([f1, f2])

    return X_pf, Y_pf
