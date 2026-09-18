# ---------------- small helpers ----------------
def _to_tensor(x, device, dtype):
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), device=device, dtype=dtype)

def _as_np(x):
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)

def _rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _metrics_multi(Yp: np.ndarray, Yt: np.ndarray, target_names: Sequence[str]) -> Dict[str, float]:
    out = {}
    for j, name in enumerate(target_names):
        pj, tj = Yp[:, j], Yt[:, j]
        ss_res = float(np.sum((pj - tj) ** 2))
        ss_tot = float(np.sum((tj - tj.mean()) ** 2) + 1e-12)
        out[f"{name}_RMSE"] = _rmse(pj, tj)
        out[f"{name}_MAE"]  = float(np.mean(np.abs(pj - tj)))
        out[f"{name}_R2"]   = 1.0 - ss_res / ss_tot
    return out


def kennard_stone(X, k: int):
    """
    X: (N, d) numpy array
    返回 k 个代表性的样本索引，保证空间填充度。
    """
    X = np.asarray(X); n = X.shape[0]
    # 两两距离矩阵
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    # 先选距离最远的两个点
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i), int(j)]
    # 迭代选择：每次加一个与已选集合“最远”的点
    while len(selected) < k:
        rest = [r for r in range(n) if r not in selected]
        dmin = np.min(D[rest][:, selected], axis=1)
        selected.append(rest[int(np.argmax(dmin))])
    return np.array(selected, dtype=int)