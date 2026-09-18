from pathlib import Path
import pickle
from datetime import datetime


def save_pf_store(
    pf_store,
    save_dir="pareto_results",
    filename=None,
    metadata=None,
):
    """
    保存 pf_store 及运行参数。

    Parameters
    ----------
    pf_store : dict
        例如：
        {
            "raw": [(seed, Xp, Yp), ...],
            "raw_uncertainty": [(seed, Xp, Yp), ...],
            "cal": [(seed, Xp, Yp), ...],
        }
    save_dir : str or Path
        保存目录。
    filename : str or None
        文件名。None 时自动加入时间戳。
    metadata : dict or None
        可选的运行参数，例如 beta、pop_size、n_gen 等。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pf_store_{timestamp}.pkl"

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    save_path = save_dir / filename

    payload = {
        "pf_store": pf_store,
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved PF results to: {save_path.resolve()}")
    return save_path