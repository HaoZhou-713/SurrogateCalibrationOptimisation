import numpy as np
import torch
from torch.quasirandom import SobolEngine

def sobol_in_box(n_dim, N, lb, ub, seed=0):
    eng = SobolEngine(dimension=n_dim, scramble=True, seed=seed)
    U = eng.draw(N).double().cpu().numpy()
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    return lb + (ub - lb) * U

def f_gt(X):   # X: (N,d) numpy
    return fonseca_fleming(X)  # (N,M)

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

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

def pareto_indices_botorch_max(Ymax: np.ndarray):
    Yt = torch.as_tensor(Ymax, dtype=torch.double)
    mask = is_non_dominated(Yt)  # assumes maximization :contentReference[oaicite:2]{index=2}
    idx = torch.where(mask)[0].cpu().numpy()
    return idx, mask.cpu().numpy()

def build_ground_truth_pf(lb, ub, d, sense, N_pool=20000, seed=0):
    X_pool = sobol_in_box(d, N_pool, lb, ub, seed=seed)
    Y_pool = f_gt(X_pool)
    Y_pool_max = sense_to_maximization(Y_pool, sense)
    idx_pf, mask = pareto_indices_botorch_max(Y_pool_max)
    X_pf = X_pool[idx_pf]
    Y_pf = Y_pool[idx_pf]
    return {
        "X_pool": X_pool,
        "Y_pool": Y_pool,
        "X_pf_gt": X_pf,
        "Y_pf_gt": Y_pf,
        "idx_pf_gt": idx_pf,
        "sense": sense,
    }
