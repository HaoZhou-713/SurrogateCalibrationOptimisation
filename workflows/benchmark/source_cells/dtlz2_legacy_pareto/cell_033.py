import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.utils.multi_objective.pareto import is_non_dominated

def build_gt_dict(
    *,
    d,
    sense,
    lb,
    ub,
    N_pool=5000,
    seed=0,
):
    """
    Build ground-truth dictionary for benchmark.

    Returns gt_dict with keys:
      - X_pool, Y_pool
      - X_pf_gt, Y_pf_gt
      - sense
    """

    # ---- 1) Sobol pool in design space ----
    eng = SobolEngine(dimension=d, scramble=True, seed=seed)
    U = eng.draw(N_pool).double().cpu().numpy()
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    X_pool = lb + (ub - lb) * U

    # ---- 2) Ground-truth evaluation ----
    Y_pool = f_gt(X_pool)          # (N_pool, M)

    # ---- 3) Convert to maximization space ----
    Y_pool_max = Y_pool.copy()
    for j, s in enumerate(sense):
        if s == "min":
            Y_pool_max[:, j] = -Y_pool_max[:, j]

    # ---- 4) Ground-truth Pareto front ----
    # Yt = torch.as_tensor(Y_pool_max, dtype=torch.double)
    # mask = is_non_dominated(Yt)
    # idx_pf = torch.where(mask)[0].cpu().numpy()


    # X_pf_gt = X_pool[idx_pf]
    # Y_pf_gt = Y_pool[idx_pf]

    X_pf_gt, Y_pf_gt = gt_pf_dtlz2_with_X(d)

    return {
        "X_pool": X_pool,
        "Y_pool": Y_pool,
        "X_pf_gt": X_pf_gt,
        "Y_pf_gt": Y_pf_gt,
        "sense": sense,
    }
