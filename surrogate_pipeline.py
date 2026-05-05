"""Battery multi-output GP pipeline (from pipeline_battery.ipynb)

This module bundles the notebook's helpers/classes into a single importable .py.

Main entry points:
- AutoConfig
- AutoMultiOutputGP
- train_val_test_then_finalize (adds flexible 'final_eval_on' option)
- train_in_seeds
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll

from sklearn.model_selection import train_test_split, KFold

# Optional plotting; safe to import even in headless environments if you don't call plotting functions.
import matplotlib.pyplot as plt


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


# ---------------- config ----------------
@dataclass

class AutoConfig:
    # ICM/IND 用到的候选核（IND 每目标各自搜索）
    kernel_grid: List[Tuple[str, Optional[float]]] = None  # e.g. [("RBF",None),("Matern",0.5),("Matern",1.5),("Matern",2.5)]
    icm_rank: int = 2
    bags: int = 0                # bagging 份数 (0=不用)
    noise_floor: float = 1e-4
    init_noise: float = 1e-3
    seed: int = 123
    val_frac: float = 0.12
    device: Any = torch.device("cpu")
    dtype: Any = torch.double
    obj_weights: Optional[np.ndarray] = None  # 合成验证分数的权重；默认等权
    # 评估拆分
    test_frac: float = 0.10
    use_ks_for_test: bool = False  # 留作扩展；本实现随机划分


# ---------------- the class ----------------

class AutoMultiOutputGP:
    """
    - 同时训练/对比两类模型：
        * ICM: 一个 MultiTaskGP（shared kernel + task coreg）
        * IND: M 个 SingleTaskGP（每个目标独立、并且“各自核搜索”）
    - 用验证集加权 RMSE 选最优；可选 bagging 集成
    - 自动缩放：X (min-max on train∪val)，Y (per-target z-score on train∪val)
    """

    def __init__(self, feature_names: List[str], target_names: List[str], config: AutoConfig = AutoConfig()):
        self.feature_names = list(feature_names)
        self.target_names  = list(target_names)
        self.M = len(self.target_names)
        self.C = config

        if self.C.kernel_grid is None:
            self.C.kernel_grid = [("RBF", None), ("Matern", 0.5), ("Matern", 1.5), ("Matern", 2.5)]

        # scalers
        self.x_min = None; self.x_span = None
        self.y_mu  = None; self.y_sd   = None

        # chosen model info
        self.best_family: Optional[str] = None         # "ICM" or "IND"
        self.best_kernel: Optional[Any]  = None        # ICM: (kind,nu); IND: list[(kind,nu)] per target
        self.models      = None                        # ICM: list[model]; IND: list[list per bag][per-target]
        self._is_fitted  = False
        self.eval_models  = None   # evaluate_final 训练出来的模型（不覆盖 self.models）
        self.final_models = None   # finalize 训练出来的最终模型（不覆盖 self.models，除非你指定）

        # splits
        self.split = {}
        self.X_all = None; self.Y_all = None

    # ====================== public API ======================
    def fit(self, X, Y, test_frac: Optional[float] = None, val_frac: Optional[float] = None, verbose=True):
        """
        X: (N,d), Y: (N,M)
        1) 随机划分 train/val/test
        2) 用 train∪val 拟合缩放器；在缩放后的空间训练/搜索
        3) 搜 ICM 与 IND：ICM 共享核搜索；IND 为每个目标单独搜索
        4) 用 val-score 选择 family+kernel
        5) 在 train∪val 上按选择的 family+kernel 重训（带 bagging）
        """
        if test_frac is None: test_frac = self.C.test_frac
        if val_frac  is None: val_frac  = self.C.val_frac
        rng = np.random.default_rng(self.C.seed)

        # to tensors
        X_all = _to_tensor(X, self.C.device, self.C.dtype)
        Y_all = _to_tensor(Y, self.C.device, self.C.dtype)
        N, d = X_all.shape
        assert Y_all.shape[1] == self.M

        # split
        all_idx = np.arange(N)
        if self.C.use_ks_for_test:
            n_test = max(5, int(round(N * test_frac)))
            # 注意：X_all 可能是 torch.tensor，这里转成 numpy
            te_idx = kennard_stone(X_all.detach().cpu().numpy(), k=n_test)
            rest = np.setdiff1d(all_idx, te_idx, assume_unique=False)
            tr_idx, va_idx = train_test_split(
                rest,
                test_size=val_frac / (1.0 - test_frac),
                random_state=self.C.seed,
                shuffle=True
            )
        else:
            if test_frac > 0:
                rest, te_idx = train_test_split(
                    all_idx,
                    test_size=test_frac,
                    random_state=self.C.seed,
                    shuffle=True
                )
            else:
                rest = all_idx
                te_idx = np.array([], dtype=int)

            tr_idx, va_idx = train_test_split(
                rest,
                test_size=val_frac / (1.0 - test_frac),
                random_state=self.C.seed, shuffle=True
            )

        self.split = {"tr": tr_idx, "va": va_idx, "te": te_idx}
        if verbose:
            print(f"Split sizes -> train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)} (KS_test={self.C.use_ks_for_test})")

        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        Ytr, Yva = Y_all[tr_idx], Y_all[va_idx]

        # scale X (min–max on train∪val)
        with torch.no_grad():
            X_ref = torch.cat([Xtr, Xva], dim=0)
            x_min = X_ref.min(dim=0).values
            x_max = X_ref.max(dim=0).values
            span  = (x_max - x_min).clamp_min(1e-12)
        def _scaleX(Xt): return (Xt - x_min) / span
        Xtr_s = _scaleX(Xtr); Xva_s = _scaleX(Xva)

        # scale Y (z-score on train∪val)
        with torch.no_grad():
            Y_ref = torch.cat([Ytr, Yva], dim=0)
            y_mu  = Y_ref.mean(dim=0)
            y_sd  = Y_ref.std (dim=0, unbiased=False).clamp_min(1e-12)
        def _scaleY(Yt):    return (Yt - y_mu) / y_sd
        Ytr_s = _scaleY(Ytr); Yva_s = _scaleY(Yva)

        # keep scalers
        self.x_min, self.x_span, self.y_mu, self.y_sd = x_min, span, y_mu, y_sd
        self.X_all, self.Y_all = X_all, Y_all

        # === search ICM
        best_icm = self._search_icm(Xtr_s, Ytr_s, Xva_s, Yva_s, verbose=verbose)

        # === search IND (per target kernel)
        best_ind = self._search_ind_per_target(Xtr_s, Ytr_s, Xva_s, Yva_s, verbose=verbose)

        # choose family
        best_overall = (None, +np.inf, None, None)  # (family, score, kernel_cfg, fitted_object)
        if best_icm and best_icm[1] < best_overall[1]:
            best_overall = ("ICM", best_icm[1], best_icm[0], best_icm[2])  # (kernel tuple, score, fitted_model)
        if best_ind and best_ind[1] < best_overall[1]:
            best_overall = ("IND", best_ind[1], best_ind[0], best_ind[2])  # (list of kernels per target)

        if best_overall[0] is None:
            raise RuntimeError("No model family produced a valid fit.")

        self.best_family = best_overall[0]
        self.best_kernel = best_overall[2]
        if verbose:
            if self.best_family == "ICM":
                k = self.best_kernel
                ktag = k[0] if k[1] is None else f"{k[0]}({k[1]})"
            else:
                ktag = "[" + ", ".join(
                    (k[0] if k[1] is None else f"{k[0]}({k[1]})") for k in self.best_kernel
                ) + "]"
            print(f"\n==> Chosen family: {self.best_family}, kernel: {ktag}, val-score={best_overall[1]:.6f}")

        # retrain final ensemble on train∪val
        # X_ref_s = _scaleX(torch.cat([Xtr, Xva], 0))
        # Y_ref_s = (torch.cat([Ytr, Yva], 0) - y_mu) / y_sd
        # if self.best_family == "ICM":
        #     self.models = self._train_icm_ensemble(X_ref_s, Y_ref_s, kernel=self.best_kernel, bags=self.C.bags)
        # else:
        #     self.models = self._train_ind_ensemble_per_target(X_ref_s, Y_ref_s, kernels=self.best_kernel, bags=self.C.bags)

        # 注意：fit() 只负责 “train 训练 + val 选参”
        # 不在这里重训 train∪val，不生成 self.models
        self.models = None
        self._is_fitted = True
        return self

    def predict(self, X, batch_size: int=1000, *, model_key: str="current") -> Tuple[np.ndarray, np.ndarray]:
        """返回 (mean, std) in ORIGINAL units, shape = (N, M)"""
        self._check_fitted()
        Xt = _to_tensor(X, self.C.device, self.C.dtype)
        Xt_s = (Xt - self.x_min) / self.x_span

        # -------- choose which model set to use --------
        mk = (model_key or "current").strip().lower()
        family = self.best_family
        if mk == "current":
            models = self.models
            # family = self.best_family

        elif mk == "eval_models":
            if not hasattr(self, "eval_models") or self.eval_models is None:
                raise RuntimeError("model_key='eval_models' requested but self.eval_models is not set. "
                                "Run evaluate_final(..., set_as_current=False) or ensure it is stored.")
            models = self.eval_models
            # family may be stored per-set; fallback to current best_family if not present
            # family = getattr(self, "eval_family", self.best_family)

        elif mk == "final_models":
            if not hasattr(self, "final_models") or self.final_models is None:
                raise RuntimeError("model_key='final_models' requested but self.final_models is not set. "
                                "Run finalize(..., set_as_current=True/False) first.")
            models = self.final_models
            # family = getattr(self, "final_family", self.best_family)

        else:
            raise ValueError("model_key must be one of: 'current', 'eval_models', 'final_models'")

        # -------- predict in scaled space --------
        if family == "ICM":
            MU_s, SIG_s = self._predict_icm_ensemble(models, Xt_s, batch_size=batch_size)
        else:
            MU_s, SIG_s = self._predict_ind_ensemble(models, Xt_s, batch_size=batch_size)

        # -------- unscale to original units --------
        MUo = (MU_s * self.y_sd + self.y_mu).detach().cpu().numpy()
        SIGo = (SIG_s * self.y_sd).detach().cpu().numpy()
        return MUo, SIGo

    def evaluate(self, X, Y, *, which="current", name="split") -> Dict[str, float]:
        self._check_fitted()
        which = which.lower()
        if which in ("current", "models"):
            if self.models is None:
                raise RuntimeError("Current models not trained. Call evaluate_final() or finalize() first.")
            return self._evaluate_with(self.models, X, Y, name=name)
        if which in ("eval", "eval_models"):
            if self.eval_models is None:
                raise RuntimeError("eval_models not found. Call evaluate_final() first.")
            return self._evaluate_with(self.eval_models, X, Y, name=name)
        if which in ("final", "final_models"):
            if self.final_models is None:
                raise RuntimeError("final_models not found. Call finalize() first.")
            return self._evaluate_with(self.final_models, X, Y, name=name)
        raise ValueError("which must be 'current'|'eval'|'final'")


    def info(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "family": self.best_family,
            "kernel": self.best_kernel,  # ICM: (kind,nu)；IND: list[(kind,nu)] per target
            "bags": self.C.bags,
            "features": self.feature_names,
            "targets": self.target_names,
        }

    def evaluate_final(
        self,
        X=None, Y=None,
        *,
        final_eval_on: str = "test",
        scaler_mode: str = "frozen",
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        用 best_family/best_kernel 构造“评估模型”并计算指标。
        规则（你要的）：
        - final_eval_on="test": fit_on=train_val, eval_on=test
        - final_eval_on="train": fit_on=train, eval_on=train
        - final_eval_on="train_val": fit_on=train_val, eval_on=train_val
        - final_eval_on="all": fit_on=all, eval_on=all（一般只用于 debug）
        返回 metrics，并保存 self.eval_models（不覆盖 self.models）。
        """
        self._check_fitted()
        if X is None: X = self.X_all
        if Y is None: Y = self.Y_all

        Xnp = np.asarray(X)
        Ynp = np.asarray(Y)

        mode = final_eval_on.lower()
        if mode in ("test", "te"):
            fit_idx  = self._pick_indices("train_val")
            eval_idx = self._pick_indices("test")
        elif mode in ("train", "tr"):
            fit_idx = eval_idx = self._pick_indices("train")
        elif mode in ("train_val", "train+val", "trva", "tr_val"):
            fit_idx = eval_idx = self._pick_indices("train_val")
        elif mode in ("all", "full", "train+val+test"):
            fit_idx = eval_idx = self._pick_indices("all")
        else:
            raise ValueError("final_eval_on must be 'train'|'train_val'|'test'|'all'")

        # 训练评估模型（不覆盖 self.models）
        out = self._train_models_on(Xnp[fit_idx], Ynp[fit_idx], scaler_mode=scaler_mode)
        if scaler_mode == "frozen":
            self.eval_models = out
            # 用 frozen scalers + eval_models 来预测
            old_models = self.models
            self.models = self.eval_models
            metrics = self.evaluate(Xnp[eval_idx], Ynp[eval_idx], name=f"eval:{final_eval_on}")
            # self.models = old_models
        else:
            # refit 模式下返回 (models, new_scalers)
            self.eval_models, new_scalers = out
            # 临时切换 scalers 计算指标（不永久覆盖）
            old = (self.models, self.x_min, self.x_span, self.y_mu, self.y_sd)
            self.models = self.eval_models
            x_min, span, y_mu, y_sd = new_scalers
            self.x_min, self.x_span, self.y_mu, self.y_sd = x_min, span, y_mu, y_sd
            metrics = self.evaluate(Xnp[eval_idx], Ynp[eval_idx], name=f"eval:{final_eval_on}")
            self.models, self.x_min, self.x_span, self.y_mu, self.y_sd = old

        if verbose:
            print(f"[evaluate_final] fit_on={len(fit_idx)}, eval_on={len(eval_idx)}, mode={final_eval_on}, scaler_mode={scaler_mode}")
            print(metrics)
        return metrics

    def finalize(
        self,
        X=None, Y=None,
        *,
        refit_scope: str = "full",
        scaler_mode: str = "frozen",
        set_as_current: bool = True,
        verbose: bool = True,
    ):
        """
        训练最终部署模型。
        - refit_scope="train_val": 用 train+val
        - refit_scope="full": 用 train+val+test
        scaler_mode:
        - "frozen": 用 fit() 的 scalers 缩放（和选参口径一致）
        - "refit": 在 refit 数据上重拟合 scalers（更像“最终全量模型”口径）
        set_as_current:
        - True: self.models 指向最终模型，predict() 默认用最终模型
        - False: 只存 self.final_models，不动 self.models
        """
        self._check_fitted()
        if X is None: X = self.X_all
        if Y is None: Y = self.Y_all

        Xnp = np.asarray(X)
        Ynp = np.asarray(Y)

        scope = refit_scope.lower()
        if scope in ("train_val", "train+val", "trva", "tr_val"):
            refit_idx = self._pick_indices("train_val")
        elif scope in ("full", "all", "train+val+test"):
            refit_idx = self._pick_indices("all")
        else:
            raise ValueError("refit_scope must be 'train_val' or 'full'")

        out = self._train_models_on(Xnp[refit_idx], Ynp[refit_idx], scaler_mode=scaler_mode)
        if scaler_mode == "frozen":
            self.final_models = out
            if set_as_current:
                self.models = self.final_models
        else:
            self.final_models, new_scalers = out
            if set_as_current:
                self.models = self.final_models
                x_min, span, y_mu, y_sd = new_scalers
                self.x_min, self.x_span, self.y_mu, self.y_sd = x_min, span, y_mu, y_sd

        if verbose:
            print(f"[finalize] refit_scope={refit_scope}, N={len(refit_idx)}, scaler_mode={scaler_mode}, set_as_current={set_as_current}")
        return self.final_models

    # ====================== internal: kernel helpers ======================
    def _evaluate_with(self, models, X, Y, name="split") -> Dict[str, float]:
        # 临时切换 self.models 来复用 predict()
        old = self.models
        self.models = models
        MUo, _ = self.predict(X)
        self.models = old
        Yt = _as_np(_to_tensor(Y, self.C.device, self.C.dtype))
        return {"split": name, **_metrics_multi(MUo, Yt, self.target_names)}

    
    def _pick_indices(self, mode: str) -> np.ndarray:
        """基于 self.split 选 tr/va/te/all 的 index"""
        if not self.split:
            raise RuntimeError("No split found. Call fit() first.")
        mode = mode.lower()
        tr, va, te = self.split["tr"], self.split["va"], self.split["te"]
        if mode in ("train", "tr"):
            return tr
        if mode in ("val", "va"):
            return va
        if mode in ("test", "te"):
            return te
        if mode in ("train_val", "train+val", "trva", "tr_val"):
            return np.concatenate([tr, va])
        if mode in ("all", "full", "train_val_test", "train+val+test"):
            return np.concatenate([tr, va, te])
        raise ValueError(f"Unknown mode={mode}")

    def _scale_with_current_scalers(self, Xt: torch.Tensor, Yt: Optional[torch.Tensor] = None):
        """用 fit() 阶段保存的 x_min/x_span/y_mu/y_sd 缩放（frozen）"""
        if self.x_min is None or self.x_span is None:
            raise RuntimeError("Scalers are not set. Call fit() first.")
        Xs = (Xt - self.x_min) / self.x_span
        if Yt is None:
            return Xs
        if self.y_mu is None or self.y_sd is None:
            raise RuntimeError("Y scalers are not set. Call fit() first.")
        Ys = (Yt - self.y_mu) / self.y_sd
        return Xs, Ys

    def _train_models_on(self, X_np, Y_np, *, bags=None, scaler_mode="frozen"):
        """
        用 best_family/best_kernel 在给定数据上训练模型并返回 models。
        scaler_mode:
        - "frozen": 用 fit() 的 scalers 缩放（推荐，避免 evaluation / finalize 时引入额外漂移）
        - "refit": 在这份数据上重新拟合 scalers（更像“全量最终模型”口径）
        """
        self._check_fitted()
        if self.best_family is None or self.best_kernel is None:
            raise RuntimeError("best_family/kernel not set. Call fit() first.")
        if bags is None:
            bags = self.C.bags

        Xt = _to_tensor(X_np, self.C.device, self.C.dtype)
        Yt = _to_tensor(Y_np, self.C.device, self.C.dtype)

        if scaler_mode not in ("frozen", "refit"):
            raise ValueError("scaler_mode must be 'frozen' or 'refit'")

        if scaler_mode == "frozen":
            Xs, Ys = self._scale_with_current_scalers(Xt, Yt)
            # 注意：不更新 self.x_min/...，保持 fit 阶段 scalers
            if self.best_family == "ICM":
                models = self._train_icm_ensemble(Xs, Ys, kernel=self.best_kernel, bags=bags)
            else:
                models = self._train_ind_ensemble_per_target(Xs, Ys, kernels=self.best_kernel, bags=bags)
            return models

        # scaler_mode == "refit"
        with torch.no_grad():
            x_min = Xt.min(dim=0).values
            x_max = Xt.max(dim=0).values
            span  = (x_max - x_min).clamp_min(1e-12)
            y_mu  = Yt.mean(dim=0)
            y_sd  = Yt.std (dim=0, unbiased=False).clamp_min(1e-12)

        Xs = (Xt - x_min) / span
        Ys = (Yt - y_mu) / y_sd

        if self.best_family == "ICM":
            models = self._train_icm_ensemble(Xs, Ys, kernel=self.best_kernel, bags=bags)
        else:
            models = self._train_ind_ensemble_per_target(Xs, Ys, kernels=self.best_kernel, bags=bags)

        # 这里是否覆盖 scalers 由调用方决定（evaluate_final 一般不覆盖；finalize 可以覆盖）
        return models, (x_min, span, y_mu, y_sd)

    def _replace_data_kernel(self, model, kind="RBF", nu=1.5, is_multitask: bool = False):
        """兼容不同结构：替换数据核（ICM/IND 通用）"""
        # 维度（对于 ICM，最后 1 维是 task id）
        d_all = model.train_inputs[0].shape[-1]
        d = d_all - 1 if is_multitask else d_all

        cm = model.covar_module
        # SingleTaskGP 常见：ScaleKernel(base=RBF/Matern)
        # MultiTaskGP 常见：cm.data_covar_module.base_kernel
        if is_multitask and hasattr(cm, "data_covar_module") and hasattr(cm.data_covar_module, "base_kernel"):
            old_base = cm.data_covar_module.base_kernel
            holder = "data"
        elif hasattr(cm, "base_kernel") and isinstance(cm, ScaleKernel):
            old_base = cm.base_kernel
            holder = "base"
        else:
            # 兜底：若 cm 本身就是 RBF/Matern，则包一层 ScaleKernel
            if isinstance(cm, (RBFKernel, MaternKernel)):
                model.covar_module = ScaleKernel(cm)
                old_base = model.covar_module.base_kernel
                holder = "base"
            else:
                raise RuntimeError(f"Unrecognized covar_module structure: {type(cm)}")

        # 新核
        if kind.upper() == "RBF":
            new_base = RBFKernel(ard_num_dims=d)
        else:
            new_base = MaternKernel(nu=float(nu), ard_num_dims=d)

        # 继承长度尺度的约束/先验（若存在）
        if hasattr(old_base, "raw_lengthscale_constraint") and old_base.raw_lengthscale_constraint is not None:
            new_base.register_constraint("raw_lengthscale", old_base.raw_lengthscale_constraint)
        if hasattr(old_base, "priors") and isinstance(old_base.priors, dict) and ("lengthscale_prior" in old_base.priors):
            new_base.register_prior("lengthscale_prior", old_base.priors["lengthscale_prior"], "lengthscale")

        # 设置回去
        if holder == "data":
            cm.data_covar_module.base_kernel = new_base
        else:
            cm.base_kernel = new_base

        return model

    # ====================== internal: ICM path ======================
    def _stack_multitask(self, Xs, Ys):
        n, d = Xs.shape
        Xb, Yb = [], []
        for j in range(self.M):
            tid = torch.full((n, 1), float(j), dtype=self.C.dtype, device=self.C.device)
            Xj = torch.cat([Xs, tid], dim=1)
            Yj = Ys[:, [j]]
            Xb.append(Xj); Yb.append(Yj)
        return torch.cat(Xb, 0), torch.cat(Yb, 0), d

    def _build_icm(self, Xs, Ys, kernel=("RBF", None)):
        Xmt, Ymt, TASK_DIM = self._stack_multitask(Xs, Ys)
        model = MultiTaskGP(train_X=Xmt, train_Y=Ymt, task_feature=TASK_DIM, rank=self.C.icm_rank)
        model = self._replace_data_kernel(model, kind=kernel[0], nu=(1.5 if kernel[1] is None else kernel[1]), is_multitask=True)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(self.C.noise_floor))
        with torch.no_grad():
            model.likelihood.noise = torch.tensor(self.C.init_noise, dtype=self.C.dtype, device=self.C.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model.eval()

    def _predict_icm_single(self, model, Xs):
        n = Xs.shape[0]
        mus = []; sigs = []
        for j in range(self.M):
            tid = torch.full((n,1), float(j), dtype=self.C.dtype, device=self.C.device)
            Xj  = torch.cat([Xs, tid], dim=1)
            post = model.posterior(Xj)
            mu = post.mean.squeeze(-1)
            sd = post.variance.clamp_min(0).sqrt().squeeze(-1)
            mus.append(mu); sigs.append(sd)
        MU  = torch.stack(mus, dim=1)
        SIG = torch.stack(sigs, dim=1)
        return MU, SIG

    # def _predict_icm_ensemble(self, models, Xs):
    #     MU_list, VAR_list = [], []
    #     for m in models:
    #         mu, sd = self._predict_icm_single(m, Xs)
    #         MU_list.append(mu); VAR_list.append(sd**2)
    #     MU  = torch.stack(MU_list, dim=0).mean(0)
    #     SIG = torch.stack(VAR_list, dim=0).mean(0).sqrt()
    #     return MU, SIG

    @torch.inference_mode()
    def _predict_icm_ensemble(self, models, Xs, batch_size: int = 2048):
        """
        ICM 集成预测（分块 + 向量化 task 堆叠）
        返回：mu, std  in scaled space, shape [N, M]
        """
        N, d = Xs.shape
        M = self.M
        device, dtype = Xs.device, Xs.dtype

        sum_mu  = torch.zeros(N, M, device=device, dtype=dtype)
        sum_mu2 = torch.zeros_like(sum_mu)
        sum_var = torch.zeros_like(sum_mu)
        K = len(models)

        # 关键：fast_pred_var 通常能显著加速 variance
        # max_root_decomposition_size 可调：越大越准但越慢（比如 50~200）
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)
                Xc = Xs[s:e]          # [n, d]
                n = Xc.shape[0]

                # 构造 task stacking 的 Xmt: [M*n, d+1]
                # 注意：task id 只需要是“整数值”，dtype 跟 X 一致即可
                tids   = torch.arange(M, device=device, dtype=dtype)          # [M]
                tidcol = tids.repeat_interleave(n).unsqueeze(-1)              # [M*n, 1]

                # 这一句会真的复制内存（不可避免），但我们把 batch_size 降下来
                Xrep   = Xc.repeat(M, 1)                                      # [M*n, d]
                Xmt    = torch.cat([Xrep, tidcol], dim=1)                     # [M*n, d+1]

                # 每个模型预测并流式聚合
                for m in models:
                    # 建议确保 m 已经 eval()
                    post = m.posterior(Xmt, observation_noise=True)

                    mu_all  = post.mean.view(M, n).transpose(0, 1).contiguous()       # [n, M]
                    var_all = post.variance.view(M, n).transpose(0, 1).contiguous()   # [n, M]

                    sum_mu [s:e] += mu_all
                    sum_mu2[s:e] += mu_all * mu_all
                    sum_var[s:e] += var_all

        mu  = sum_mu / K
        var = (sum_var / K) + (sum_mu2 / K) - mu * mu
        var = var.clamp_min(0)
        return mu, var.sqrt()

    def _train_icm_ensemble(self, Xs, Ys, kernel, bags=0):
        models = [self._build_icm(Xs, Ys, kernel)]
        if bags > 0:
            rng = np.random.default_rng(self.C.seed)
            for _ in range(bags):
                idx = torch.tensor(rng.integers(0, Xs.shape[0], Xs.shape[0]), device=self.C.device, dtype=torch.long)
                models.append(self._build_icm(Xs[idx], Ys[idx], kernel))
        return models

    def _search_icm(self, Xtr_s, Ytr_s, Xva_s, Yva_s, verbose=True):
        best = (None, +np.inf, None)
        w = self.C.obj_weights if self.C.obj_weights is not None else (np.ones(self.M)/self.M)
        for kind, nu in self.C.kernel_grid:
            model = self._build_icm(Xtr_s, Ytr_s, kernel=(kind, nu))
            MUv, _ = self._predict_icm_single(model, Xva_s)
            MUo = MUv * self.y_sd + self.y_mu
            Yvo = Yva_s * self.y_sd + self.y_mu
            rmse_each = np.sqrt(((MUo.detach().cpu().numpy() - Yvo.detach().cpu().numpy())**2).mean(axis=0))
            score = float((rmse_each * w).sum())
            if verbose:
                tag = kind if nu is None else f"{kind}({nu})"
                print(f"[ICM] {tag:<10} | val-score={score:.6f} | per-target RMSE={rmse_each.round(4)}")
            if score < best[1]:
                best = ((kind, nu), score, model)
        return best

    # ====================== internal: IND path (per-target kernel) ======================
    def _build_ind(self, Xs, yjs, kernel=("RBF", None)):
        d = Xs.shape[1]
        if kernel[0].upper() == "RBF":
            base = RBFKernel(ard_num_dims=d)
        else:
            nu = 1.5 if kernel[1] is None else float(kernel[1])
            base = MaternKernel(nu=nu, ard_num_dims=d)
        covar = ScaleKernel(base)
        model = SingleTaskGP(train_X=Xs, train_Y=yjs, covar_module=covar)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(self.C.noise_floor))
        with torch.no_grad():
            model.likelihood.noise = torch.tensor(self.C.init_noise, dtype=self.C.dtype, device=self.C.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model.eval()


    def _predict_ind_models(self, models_per_target, Xs):
        mus = []; sigs = []
        for m in models_per_target:
            post = m.posterior(Xs)
            mus.append(post.mean.squeeze(-1))
            sigs.append(post.variance.clamp_min(0).sqrt().squeeze(-1))
        MU  = torch.stack(mus, dim=1)
        SIG = torch.stack(sigs, dim=1)
        return MU, SIG

    def _search_ind_per_target(self, Xtr_s, Ytr_s, Xva_s, Yva_s, verbose=True):
        """
        为每个目标单独做 kernel_grid 搜索，得到 per-target 最优 kernel。
        验证分数计算时，使用“每个目标各自的最优模型”联合起来评估（合成加权 RMSE）。
        """
        per_target_best = []
        # 逐目标搜索
        for j in range(self.M):
            best_j = (None, +np.inf, None)  # (kernel, score_j, model_j)
            ytr_j = Ytr_s[:, [j]]
            yva_j = Yva_s[:, [j]]
            for kind, nu in self.C.kernel_grid:
                m_j = self._build_ind(Xtr_s, ytr_j, kernel=(kind, nu))
                post = m_j.posterior(Xva_s)
                mu_j = post.mean.squeeze(-1)  # scaled
                mu_o = mu_j * self.y_sd[j] + self.y_mu[j]
                y_o  = yva_j.squeeze(-1) * self.y_sd[j] + self.y_mu[j]
                rmse_j = _rmse(mu_o.detach().cpu().numpy(), y_o.detach().cpu().numpy())
                if verbose:
                    tag = kind if nu is None else f"{kind}({nu})"
                    print(f"[IND:t{j}] {tag:<10} | RMSE={rmse_j:.6f}")
                if rmse_j < best_j[1]:
                    best_j = ((kind, nu), rmse_j, m_j)
            per_target_best.append(best_j)

        # 用“每目标最优”的模型打一个联合验证分
        MU_list = []
        for j in range(self.M):
            m_j = per_target_best[j][2]
            MU_list.append(m_j.posterior(Xva_s).mean.squeeze(-1))
        MUv = torch.stack(MU_list, dim=1)  # scaled
        MUo = MUv * self.y_sd + self.y_mu
        Yvo = Yva_s * self.y_sd + self.y_mu
        rmse_each = np.sqrt(((MUo.cpu().detach().numpy() - Yvo.cpu().detach().numpy())**2).mean(axis=0))
        w = self.C.obj_weights if self.C.obj_weights is not None else (np.ones(self.M)/self.M)
        score = float((rmse_each * w).sum())
        if verbose:
            print(f"[IND] per-target best kernels -> {[b[0] for b in per_target_best]}")
            print(f"[IND] overall val-score={score:.6f} | per-target RMSE={rmse_each.round(4)}")

        # 返回“每目标最优的 kernel 列表”，以及用最优核重训一套 models 以备后续集成
        best_kernels = [b[0] for b in per_target_best]  # list[(kind,nu)]
        base_models  = [b[2] for b in per_target_best]
        return best_kernels, score, base_models

    def _train_ind_ensemble_per_target(self, Xs, Ys, kernels, bags=0):
        """
        kernels: list[(kind,nu)]，长度 = M，每个目标自己的最优核
        返回：list of ensembles；每个“bag”是一组长度 M 的 SingleTaskGP
        """
        def _train_one_set(Xs_, Ys_):
            ms = []
            for j in range(self.M):
                ms.append(self._build_ind(Xs_, Ys_[:, [j]], kernel=kernels[j]))
            return ms

        models = [_train_one_set(Xs, Ys)]
        if bags > 0:
            rng = np.random.default_rng(self.C.seed)
            for _ in range(bags):
                idx = torch.tensor(rng.integers(0, Xs.shape[0], Xs.shape[0]), device=self.C.device, dtype=torch.long)
                models.append(_train_one_set(Xs[idx], Ys[idx]))
        return models  # list[bag][target]

    # def _predict_ind_ensemble(self, ensembles, Xs):
    #     MU_list, VAR_list = [], []
    #     for models_per_target in ensembles:
    #         mu, sd = self._predict_ind_models(models_per_target, Xs)
    #         MU_list.append(mu); VAR_list.append(sd**2)
    #     MU  = torch.stack(MU_list, dim=0).mean(0)
    #     SIG = torch.stack(VAR_list, dim=0).mean(0).sqrt()
    #     return MU, SIG

    @torch.no_grad()
    def _predict_ind_ensemble(self, ensembles, Xs, batch_size: int = 8192):
        """
        IND 集成预测（流式聚合 + 分块）
        ensembles: list[bag][target] 的 SingleTaskGP
        返回：MU, SIG in scaled space, shape [N, M]
        """
        N, d = Xs.shape
        M = self.M
        device = Xs.device
        dtype  = Xs.dtype

        K = len(ensembles)  # bag 数
        sum_mu  = torch.zeros(N, M, device=device, dtype=dtype)
        sum_mu2 = torch.zeros_like(sum_mu)
        sum_var = torch.zeros_like(sum_mu)

        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            Xc = Xs[s:e]   # [n, d]

            for models_per_target in ensembles:
                # 对每个目标独立预测，再拼
                mu_list = []
                var_list = []
                for j in range(M):
                    post = models_per_target[j].posterior(Xc, observation_noise=False)
                    mu_j  = post.mean.squeeze(-1)                  # [n]
                    var_j = post.variance.clamp_min(0).squeeze(-1) # [n]
                    mu_list.append(mu_j)
                    var_list.append(var_j)
                mu_all  = torch.stack(mu_list,  dim=1)  # [n, M]
                var_all = torch.stack(var_list, dim=1)  # [n, M]

                sum_mu [s:e]  += mu_all
                sum_mu2[s:e]  += mu_all * mu_all
                sum_var[s:e]  += var_all

        mu  = sum_mu / K
        var = (sum_var / K) + (sum_mu2 / K) - mu * mu
        var = var.clamp_min(0)
        return mu, var.sqrt()

    # ====================== utils ======================
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit(X, Y) first.")
        
    # def refit_on_full_data(self, X=None, Y=None, bags: Optional[int] = None, verbose: bool = True):
        # """
        # 用“已确定的 best_family / best_kernel”在 **全部数据** 上重训最终模型。
        # - 不做任何模型选择；只按既定 family+kernel 训练。
        # - 重新估计缩放器（X: min-max；Y: z-score）基于全量数据。
        # - 可指定 bags 覆写 config.bags，否则沿用 config。
        # """
        # self._check_fitted()  # 确保已经跑过 fit() 完成选参
        # if X is None: X = self.X_all
        # if Y is None: Y = self.Y_all
        # if bags is None: bags = self.C.bags

        # Xt = _to_tensor(X, self.C.device, self.C.dtype)
        # Yt = _to_tensor(Y, self.C.device, self.C.dtype)

        # # 1) 重新拟合缩放器（基于全量数据）
        # with torch.no_grad():
        #     x_min = Xt.min(dim=0).values
        #     x_max = Xt.max(dim=0).values
        #     span  = (x_max - x_min).clamp_min(1e-12)
        #     y_mu  = Yt.mean(dim=0)
        #     y_sd  = Yt.std (dim=0, unbiased=False).clamp_min(1e-12)

        # def _scaleX(Z): return (Z - x_min) / span
        # Xs = _scaleX(Xt)
        # Ys = (Yt - y_mu) / y_sd

        # # 2) 按已选 family+kernel 重训
        # if self.best_family == "ICM":
        #     self.models = self._train_icm_ensemble(Xs, Ys, kernel=self.best_kernel, bags=bags)
        # elif self.best_family == "IND":
        #     self.models = self._train_ind_ensemble_per_target(Xs, Ys, kernels=self.best_kernel, bags=bags)
        # else:
        #     raise RuntimeError("best_family is not set. Call fit() first.")

        # # 3) 覆盖缩放器与数据缓存（用于后续 predict）
        # self.x_min, self.x_span, self.y_mu, self.y_sd = x_min, span, y_mu, y_sd
        # self.X_all, self.Y_all = Xt, Yt

        # if verbose:
        #     print(f"[final] retrained on FULL data: N={Xt.shape[0]}, family={self.best_family}, kernel={self.best_kernel}, bags={bags}")
        # return self

    def refit_on_full_data(
        self, X=None, Y=None, bags: Optional[int] = None, verbose: bool = True,
        scaler_mode: str = "frozen"  # "frozen" | "refit"
    ):
        """
        用“已确定的 best_family / best_kernel”在 **全部数据** 上重训最终模型。
        scaler_mode:
        - "frozen": 使用 fit() 时的缩放器（推荐，保证与选参阶段一致）
        - "refit" : 在全量数据上重新拟合缩放器（均值/方差/范围更稳）
        """
        self._check_fitted()
        if X is None: X = self.X_all
        if Y is None: Y = self.Y_all
        if bags is None: bags = self.C.bags

        Xt = _to_tensor(X, self.C.device, self.C.dtype)
        Yt = _to_tensor(Y, self.C.device, self.C.dtype)

        if scaler_mode not in ("frozen", "refit"):
            raise ValueError("scaler_mode must be 'frozen' or 'refit'")

        # 1) 选择/生成缩放器
        if scaler_mode == "frozen":
            # 直接使用 fit 阶段的缩放器
            x_min = self.x_min; span = self.x_span
            y_mu  = self.y_mu;  y_sd = self.y_sd
        else:  # "refit"
            with torch.no_grad():
                x_min = Xt.min(dim=0).values
                x_max = Xt.max(dim=0).values
                span  = (x_max - x_min).clamp_min(1e-12)
                y_mu  = Yt.mean(dim=0)
                y_sd  = Yt.std (dim=0, unbiased=False).clamp_min(1e-12)
            # 若想做稳健缩放，可替换为分位数/median-MAD：
            # q1, q99 = torch.quantile(Xt, 0.01, dim=0), torch.quantile(Xt, 0.99, dim=0)
            # x_min, span = q1, (q99-q1).clamp_min(1e-12)
            # y_mu, y_sd  = torch.median(Yt, dim=0).values, (1.4826*(Yt - y_mu).abs().median(dim=0).values).clamp_min(1e-12)

        def _scaleX(Z): return (Z - x_min) / span
        Xs = _scaleX(Xt)
        Ys = (Yt - y_mu) / y_sd

        # 2) 按已选 family+kernel 重训
        if self.best_family == "ICM":
            self.models = self._train_icm_ensemble(Xs, Ys, kernel=self.best_kernel, bags=bags)
        elif self.best_family == "IND":
            self.models = self._train_ind_ensemble_per_target(Xs, Ys, kernels=self.best_kernel, bags=bags)
        else:
            raise RuntimeError("best_family is not set. Call fit() first.")

        # 3) 是否更新缩放器到“最终模型”
        if scaler_mode == "refit":
            # 最终模型基于全量缩放器；predict 时也会用新的缩放器
            self.x_min, self.x_span, self.y_mu, self.y_sd = x_min, span, y_mu, y_sd
        # 若是 "frozen"，保持原缩放器不变（与选参阶段一致）

        # self.X_all, self.Y_all = Xt, Yt

        if verbose:
            print(f"[final] retrained on FULL data: N={Xt.shape[0]}, family={self.best_family}, "
                f"kernel={self.best_kernel}, bags={bags}, scaler_mode={scaler_mode}")
        return self

# def train_val_test_then_finalize(auto_gp: AutoMultiOutputGP, X, Y, verbose=True):
#     """
#     1) auto_gp.fit(): 内部随机划分 train/val/test，选参并在 train∪val 重训；
#     2) evaluate()：在 test 上评估一次（记录指标，不再调参）；
#     3) refit_on_full_data()：将 test 也并回去，用已定参数在全量数据上重训最终模型。
#     返回：test_metrics, final_info
#     """
#     auto_gp.fit(X, Y, verbose=verbose)
#     # test 指标
#     te_idx = auto_gp.split["te"]
#     test_metrics = auto_gp.evaluate(X[te_idx], Y[te_idx], name="test")

#     if verbose:
#         print("[test] metrics:", test_metrics)

#     # 全量重训用于部署
#     auto_gp.refit_on_full_data(X, Y, verbose=verbose)
#     final_info = auto_gp.info()
#     return test_metrics, final_info

def train_val_test_then_finalize(
    auto_gp: AutoMultiOutputGP,
    X,
    Y,
    *,
    # 训练阶段仍然会做 train/val/test 划分：train/val 用来选参，test 默认不参与选参
    test_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
    # 训练完成后：你想“最后测试/汇报指标”用哪一部分数据
    # - "train": 只在训练集上报一次指标
    # - "train_val": 在 train∪val 上报一次指标
    # - "test": 在 test 上报一次指标（默认）
    # - "all": 在 train∪val∪test 上报一次指标
    final_eval_on: str = "test",
    # 是否在训练后把数据并回去重训“最终部署模型”
    # - "train_val": 只用 train∪val 重训（保留 test 作为真正外部测试集）
    # - "full": 用 train∪val∪test 全量数据重训（用于部署，但不再有独立 test）
    refit_scope: str = "full",
    scaler_mode: str = "frozen",  # 透传到 refit_on_full_data: "frozen" 推荐；或 "refit"
    verbose: bool = True,
):
    """
    完整流程（适合 notebook -> 可复现脚本）：
    1) auto_gp.fit(): 随机划分 train/val/test，做模型家族(ICM/IND)+核函数搜索，并在 train∪val 上重训
    2) final_eval_on: 选择你要用哪一部分数据做“最后汇报”的 evaluation
    3) refit_scope: （可选）用既定 best_family/best_kernel 在更大数据集上重训最终模型

    Returns
    -------
    eval_metrics: Dict[str, float]
        final_eval_on 对应数据集上的指标
    final_info: Dict[str, Any]
        auto_gp.info()
    indices: Dict[str, np.ndarray]
        {"tr":..., "va":..., "te":..., "eval":..., "refit":...}
    """
    auto_gp.fit(X, Y, test_frac=test_frac, val_frac=val_frac, verbose=verbose)

    tr_idx = auto_gp.split["tr"]
    va_idx = auto_gp.split["va"]
    te_idx = auto_gp.split["te"]

    def _pick(mode: str) -> np.ndarray:
        mode = mode.lower()
        if mode in ("train", "tr"):
            return tr_idx
        if mode in ("train_val", "train+val", "trva", "tr_val"):
            return np.concatenate([tr_idx, va_idx])
        if mode in ("test", "te"):
            return te_idx
        if mode in ("all", "train_val_test", "train+val+test", "full"):
            return np.concatenate([tr_idx, va_idx, te_idx])
        raise ValueError(f"final_eval_on must be one of: 'train'|'train_val'|'test'|'all' (got {mode!r})")

    eval_idx = _pick(final_eval_on)
    eval_metrics = auto_gp.evaluate(np.asarray(X)[eval_idx], np.asarray(Y)[eval_idx], name=str(final_eval_on))

    if verbose:
        print(f"[{final_eval_on}] metrics:", eval_metrics)

    # 重训最终模型（用于部署/后续采样/优化）
    refit_scope_l = refit_scope.lower()
    if refit_scope_l not in ("full", "train_val", "train+val", "trva"):
        raise ValueError("refit_scope must be 'full' or 'train_val'")

    if refit_scope_l == "train_val":
        refit_idx = np.concatenate([tr_idx, va_idx])
    else:
        refit_idx = np.concatenate([tr_idx, va_idx, te_idx])

    auto_gp.refit_on_full_data(
        np.asarray(X)[refit_idx],
        np.asarray(Y)[refit_idx],
        scaler_mode=scaler_mode,
        verbose=verbose,
    )
    final_info = auto_gp.info()

    return eval_metrics, final_info, {
        "tr": tr_idx, "va": va_idx, "te": te_idx,
        "eval": eval_idx, "refit": refit_idx
    }


def plot_pred_vs_actual(
    Y_tr, MU_tr,
    Y_va, MU_va,
    Y_te, MU_te,
    target_names=None,
    colors=None,
    sizes=None,
    add_pad=True,
    savepath_prefix=None,
    dpi=200,
):
    """
    Plot Predicted vs Actual for Train/Val/Test with a shared axis range per target.

    Args:
        Y_tr, MU_tr: (N_tr, M) arrays of true/pred for train.
        Y_va, MU_va: (N_va, M) arrays of true/pred for val.
        Y_te, MU_te: (N_te, M) arrays of true/pred for test.
        target_names: list of length M with target names (defaults to ["y1", ...]).
        colors: dict for split colors; defaults to {"Train":"#1f77b4","Val":"#ff7f0e","Test":"#2ca02c"}.
        sizes: dict for split point sizes; defaults to {"Train":22,"Val":28,"Test":34}.
        add_pad: whether to add a small padding to axis limits.
        savepath_prefix: if not None, saves each fig to f"{prefix}_{name}.png".
        dpi: save resolution.

    Returns:
        figs, axes, metrics: lists of figures, axes, and per-target metric dicts.
    """
    # ---- helpers ----
    def _to_np(a):
        if hasattr(a, "detach"): a = a.detach().cpu().numpy()
        return np.asarray(a)

    def _metrics(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        ss_res = float(np.sum((y_pred - y_true)**2))
        ss_tot = float(np.sum((y_true - y_true.mean())**2) + 1e-12)
        r2   = 1.0 - ss_res / ss_tot
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    # ---- numpy-ize ----
    Y_tr, MU_tr = _to_np(Y_tr), _to_np(MU_tr)
    Y_va, MU_va = _to_np(Y_va), _to_np(MU_va)
    Y_te, MU_te = _to_np(Y_te), _to_np(MU_te)

    # Ensure 2D
    if Y_tr.ndim == 1: Y_tr = Y_tr[:, None]
    if MU_tr.ndim == 1: MU_tr = MU_tr[:, None]
    if Y_va.ndim == 1: Y_va = Y_va[:, None]
    if MU_va.ndim == 1: MU_va = MU_va[:, None]
    if Y_te.ndim == 1: Y_te = Y_te[:, None]
    if MU_te.ndim == 1: MU_te = MU_te[:, None]

    M = Y_tr.shape[1]
    if target_names is None:
        target_names = [f"y{j+1}" for j in range(M)]

    if colors is None:
        colors = {"Train":"#1f77b4", "Val":"#ff7f0e", "Test":"#2ca02c"}
    if sizes is None:
        sizes  = {"Train":22, "Val":28, "Test":34}

    figs, axes, metrics_list = [], [], []

    for j, name in enumerate(target_names):
        all_actual = np.concatenate([Y_tr[:, j], Y_va[:, j], Y_te[:, j]])
        all_pred   = np.concatenate([MU_tr[:, j], MU_va[:, j], MU_te[:, j]])
        global_min = float(min(all_actual.min(), all_pred.min()))
        global_max = float(max(all_actual.max(), all_pred.max()))
        if add_pad:
            pad = 0.02 * (global_max - global_min + 1e-12)
            global_min -= pad
            global_max += pad

        fig, ax = plt.subplots(figsize=(5.6, 5.2))

        # 1) diagonal
        ax.plot([global_min, global_max], [global_min, global_max],
                'k--', lw=0.8, alpha=0.5, zorder=1)

        # 2) points
        ax.scatter(Y_tr[:, j], MU_tr[:, j], s=sizes["Train"], c=colors["Train"],
                   alpha=0.8, edgecolors='k', linewidths=0.4, label="Train", zorder=3)
        ax.scatter(Y_va[:, j], MU_va[:, j], s=sizes["Val"], c=colors["Val"],
                   alpha=0.85, edgecolors='k', linewidths=0.4, label="Val", zorder=3)
        ax.scatter(Y_te[:, j], MU_te[:, j], s=sizes["Test"], c=colors["Test"],
                   alpha=0.9, edgecolors='k', linewidths=0.4, label="Test", zorder=3)

        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name} — Predicted vs Actual")

        # test metrics
        m = _metrics(Y_te[:, j], MU_te[:, j])
        ax.text(0.02, 0.98,
                f"Test RMSE={m['RMSE']:.4f}\nMAE={m['MAE']:.4f}\nR²={m['R2']:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()

        if savepath_prefix is not None:
            fname = f"{savepath_prefix}_{name}.png"
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")

        figs.append(fig)
        axes.append(ax)
        metrics_list.append({"target": name, **m})

        # If you prefer to display immediately, keep plt.show(); else, comment out:
        plt.show()

    return figs, axes, metrics_list

