def to_max_space(Y, sense):
    return sense_to_maximization(Y, sense)

def pairwise_dist(A, B):
    # A: (n,a), B: (m,a)
    # return (n,m)
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(A2 + B2 - 2*A@B.T, 0.0))

def GD(Y_pred, Y_gt):
    D = pairwise_dist(Y_pred, Y_gt)
    return np.mean(np.min(D, axis=1))

def IGD(Y_pred, Y_gt):
    D = pairwise_dist(Y_gt, Y_pred)
    return np.mean(np.min(D, axis=1))

def hausdorff(Y_pred, Y_gt):
    D_pg = pairwise_dist(Y_pred, Y_gt)
    d1 = np.max(np.min(D_pg, axis=1))  # pred -> gt
    d2 = np.max(np.min(D_pg, axis=0))  # gt -> pred
    return max(d1, d2)

def pf_recall(Y_pred, Y_gt, eps):
    D = pairwise_dist(Y_gt, Y_pred)
    covered = (np.min(D, axis=1) <= eps)
    return float(np.mean(covered))

from botorch.utils.multi_objective.hypervolume import Hypervolume

def hypervolume_max(Ymax_pf, ref_point_max):
    Yt = torch.as_tensor(Ymax_pf, dtype=torch.double)
    ref = torch.as_tensor(ref_point_max, dtype=torch.double)
    hv = Hypervolume(ref_point=ref)  # maximization interface :contentReference[oaicite:4]{index=4}
    return float(hv.compute(Yt))

def choose_ref_point_max(Ymax_pool, margin=0.05):
    # ref = min - margin*(max-min)
    ymin = Ymax_pool.min(axis=0)
    ymax = Ymax_pool.max(axis=0)
    return ymin - margin*(ymax - ymin)
