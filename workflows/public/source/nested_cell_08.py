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
    use_model_selection: bool = True

    fixed_family: str = "IND"  # "ICM" or "IND"
    fixed_kernel: Tuple[str, Optional[float]] = ("Matern", 2.5)


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
            rest, te_idx = train_test_split(
                all_idx, test_size=test_frac,
                random_state=self.C.seed, shuffle=True
            )
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

        if self.C.use_model_selection:
            # 原来的 validation-based model selection
            best_icm = self._search_icm(
                Xtr_s, Ytr_s, Xva_s, Yva_s, verbose=verbose
            )

            best_ind = self._search_ind_per_target(
                Xtr_s, Ytr_s, Xva_s, Yva_s, verbose=verbose
            )

            best_overall = (None, +np.inf, None, None)

            if best_icm and best_icm[1] < best_overall[1]:
                best_overall = (
                    "ICM",
                    best_icm[1],
                    best_icm[0],
                    best_icm[2],
                )

            if best_ind and best_ind[1] < best_overall[1]:
                best_overall = (
                    "IND",
                    best_ind[1],
                    best_ind[0],
                    best_ind[2],
                )

            if best_overall[0] is None:
                raise RuntimeError("No model family produced a valid fit.")

            self.best_family = best_overall[0]
            self.best_kernel = best_overall[2]

        else:
            # 不做 model selection，直接使用指定模型
            self.best_family = self.C.fixed_family.upper()

            if self.best_family == "ICM":
                self.best_kernel = self.C.fixed_kernel

            elif self.best_family == "IND":
                # 每个输出使用相同的固定 kernel
                self.best_kernel = [
                    self.C.fixed_kernel for _ in range(self.M)
                ]

            else:
                raise ValueError(
                    "fixed_family must be either 'ICM' or 'IND'."
                )
            
        if verbose:
            if self.best_family == "ICM":
                k = self.best_kernel
                ktag = (
                    k[0]
                    if k[1] is None
                    else f"{k[0]}({k[1]})"
                )
            else:
                ktag = "[" + ", ".join(
                    (
                        k[0]
                        if k[1] is None
                        else f"{k[0]}({k[1]})"
                    )
                    for k in self.best_kernel
                ) + "]"

            if self.C.use_model_selection:
                print(
                    f"\n==> Chosen family: {self.best_family}, "
                    f"kernel: {ktag}, "
                    f"val-score={best_overall[1]:.6f}"
                )
            else:
                print(
                    f"\n==> Fixed family: {self.best_family}, "
                    f"kernel: {ktag}, "
                    "model selection disabled"
                )

        # retrain final ensemble on train∪val
        X_ref_s = _scaleX(torch.cat([Xtr, Xva], 0))
        Y_ref_s = (torch.cat([Ytr, Yva], 0) - y_mu) / y_sd
        if self.best_family == "ICM":
            self.models = self._train_icm_ensemble(X_ref_s, Y_ref_s, kernel=self.best_kernel, bags=self.C.bags)
        else:
            self.models = self._train_ind_ensemble_per_target(X_ref_s, Y_ref_s, kernels=self.best_kernel, bags=self.C.bags)

        self._is_fitted = True
        return self

    def predict(self, X, batch_size: int=8192) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (mean, std) in ORIGINAL units, shape = (N, M)"""
        self._check_fitted()
        Xt = _to_tensor(X, self.C.device, self.C.dtype)
        Xt_s = (Xt - self.x_min) / self.x_span

        if self.best_family == "ICM":
            MU_s, SIG_s = self._predict_icm_ensemble(self.models, Xt_s, batch_size=batch_size)
        else:
            MU_s, SIG_s = self._predict_ind_ensemble(self.models, Xt_s, batch_size=batch_size)

        MUo = (MU_s * self.y_sd + self.y_mu).cpu().detach().numpy()
        SIGo = (SIG_s * self.y_sd).cpu().detach().numpy()
        return MUo, SIGo

    def evaluate(self, X, Y, name="split") -> Dict[str, float]:
        MUo, _ = self.predict(X)
        Yt = _as_np(_to_tensor(Y, self.C.device, self.C.dtype))
        return {"split": name, **_metrics_multi(MUo, Yt, self.target_names)}

    def info(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "family": self.best_family,
            "kernel": self.best_kernel,  # ICM: (kind,nu)；IND: list[(kind,nu)] per target
            "bags": self.C.bags,
            "features": self.feature_names,
            "targets": self.target_names,
        }

    # ====================== internal: kernel helpers ======================
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
        """Build one ICM GP with numerical fallbacks only if the original fit fails."""
        # Attempt 0 reproduces the original numerical settings exactly.
        # Later attempts are only used after BoTorch raises ModelFittingError.
        retry_settings = [
            (float(self.C.init_noise), 1e-8),
            (max(float(self.C.init_noise), 1e-2), 1e-6),
            (max(float(self.C.init_noise), 5e-2), 1e-4),
        ]
        last_error = None

        for init_noise, jitter in retry_settings:
            Xmt, Ymt, TASK_DIM = self._stack_multitask(Xs, Ys)
            model = MultiTaskGP(
                train_X=Xmt,
                train_Y=Ymt,
                task_feature=TASK_DIM,
                rank=self.C.icm_rank,
            )
            model = self._replace_data_kernel(
                model,
                kind=kernel[0],
                nu=(1.5 if kernel[1] is None else kernel[1]),
                is_multitask=True,
            )
            model.likelihood.noise_covar.register_constraint(
                "raw_noise",
                GreaterThan(self.C.noise_floor),
            )
            with torch.no_grad():
                model.likelihood.noise = torch.tensor(
                    max(init_noise, float(self.C.noise_floor) * 1.01),
                    dtype=self.C.dtype,
                    device=self.C.device,
                )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            try:
                with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                    fit_gpytorch_mll(mll)
                return model.eval()
            except Exception as exc:
                if exc.__class__.__name__ != "ModelFittingError":
                    raise
                last_error = exc
                del mll, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise last_error

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

    @torch.no_grad()
    def _predict_icm_ensemble(self, models, Xs, batch_size: int = 8192):
        """
        ICM 集成预测（流式聚合 + 分块 + 任务向量化）
        返回：MU, SIG in scaled space, shape [N, M]
        """
        N, d = Xs.shape
        M = self.M
        device = Xs.device
        dtype  = Xs.dtype

        # streaming accumulators
        sum_mu  = torch.zeros(N, M, device=device, dtype=dtype)
        sum_mu2 = torch.zeros_like(sum_mu)
        sum_var = torch.zeros_like(sum_mu)
        K = len(models)

        # chunk loop
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            Xc = Xs[s:e]                           # [n, d]
            n  = Xc.shape[0]

            # 构造所有任务的堆叠输入，一次 posterior
            # Xt_all: [M*n, d+1] ; 任务 id 列重复 n 次
            tids   = torch.arange(M, device=device, dtype=dtype)   # [M]
            tidcol = tids.repeat_interleave(n).unsqueeze(-1)       # [M*n, 1]
            Xrep   = Xc.repeat(M, 1)                               # [M*n, d]
            Xmt    = torch.cat([Xrep, tidcol], dim=1)              # [M*n, d+1]

            # 对每个模型做流式聚合
            # 为避免在 GPU/CPU 上重复构图，确保 eval/no_grad
            for m in models:
                post = m.posterior(Xmt, observation_noise=False)
                mu_all = post.mean.view(M, n).transpose(0, 1)      # [n, M]
                var_all = post.variance.view(M, n).transpose(0, 1) # [n, M]

                sum_mu [s:e]  += mu_all
                sum_mu2[s:e]  += mu_all * mu_all
                sum_var[s:e]  += var_all

        mu  = sum_mu / K
        # E[var] + Var[E]
        var = (sum_var / K) + (sum_mu2 / K) - mu * mu
        var = var.clamp_min(0)
        return mu, var.sqrt()

    def _train_icm_ensemble(self, Xs, Ys, kernel, bags=0):
        """Full-data base model + exactly `bags` successful bootstrap ICM models."""
        # The base model is not optional: if it cannot be fitted even after the
        # numerical fallbacks in _build_icm, propagate the error.
        models = [self._build_icm(Xs, Ys, kernel)]

        if bags > 0:
            rng = np.random.default_rng(self.C.seed)
            successful = 0
            failed_draws = 0
            max_draws = max(int(bags) * 10, int(bags) + 20)
            draws = 0

            while successful < int(bags) and draws < max_draws:
                draws += 1
                idx = torch.tensor(
                    rng.integers(0, Xs.shape[0], Xs.shape[0]),
                    device=self.C.device,
                    dtype=torch.long,
                )
                try:
                    models.append(
                        self._build_icm(Xs[idx], Ys[idx], kernel)
                    )
                    successful += 1
                except Exception as exc:
                    if exc.__class__.__name__ != "ModelFittingError":
                        raise
                    # A bootstrap sample with many repeated X values can make
                    # an exact GP numerically pathological. Discard only that
                    # bag and draw a new bootstrap sample.
                    failed_draws += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if successful < int(bags):
                raise RuntimeError(
                    f"ICM bagging produced only {successful}/{bags} successful "
                    f"bootstrap models after {draws} draws. "
                    "Consider inspecting duplicate/near-duplicate X values."
                )

            if failed_draws > 0:
                print(
                    f"[bagging:ICM] resampled {failed_draws} failed bootstrap "
                    f"draw(s); retained exactly {bags} successful bags."
                )

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
        """Build one independent GP with numerical fallbacks only after a failed original fit."""
        d = Xs.shape[1]
        retry_settings = [
            (float(self.C.init_noise), 1e-8),
            (max(float(self.C.init_noise), 1e-2), 1e-6),
            (max(float(self.C.init_noise), 5e-2), 1e-4),
        ]
        last_error = None

        for init_noise, jitter in retry_settings:
            if kernel[0].upper() == "RBF":
                base = RBFKernel(ard_num_dims=d)
            else:
                nu = 1.5 if kernel[1] is None else float(kernel[1])
                base = MaternKernel(nu=nu, ard_num_dims=d)

            covar = ScaleKernel(base)
            model = SingleTaskGP(
                train_X=Xs,
                train_Y=yjs,
                covar_module=covar,
            )
            model.likelihood.noise_covar.register_constraint(
                "raw_noise",
                GreaterThan(self.C.noise_floor),
            )
            with torch.no_grad():
                model.likelihood.noise = torch.tensor(
                    max(init_noise, float(self.C.noise_floor) * 1.01),
                    dtype=self.C.dtype,
                    device=self.C.device,
                )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            try:
                with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                    fit_gpytorch_mll(mll)
                return model.eval()
            except Exception as exc:
                if exc.__class__.__name__ != "ModelFittingError":
                    raise
                last_error = exc
                del mll, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise last_error


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

        Important for small nested-CV subsets:
        a failed bootstrap bag is discarded and resampled, rather than
        terminating the complete outer-CV experiment. The requested number
        of successful bags is still retained.
        """
        def _train_one_set(Xs_, Ys_):
            ms = []
            for j in range(self.M):
                ms.append(
                    self._build_ind(
                        Xs_,
                        Ys_[:, [j]],
                        kernel=kernels[j],
                    )
                )
            return ms

        # Base full-data model must succeed.
        models = [_train_one_set(Xs, Ys)]

        if bags > 0:
            rng = np.random.default_rng(self.C.seed)
            successful = 0
            failed_draws = 0
            max_draws = max(int(bags) * 10, int(bags) + 20)
            draws = 0

            while successful < int(bags) and draws < max_draws:
                draws += 1
                idx = torch.tensor(
                    rng.integers(0, Xs.shape[0], Xs.shape[0]),
                    device=self.C.device,
                    dtype=torch.long,
                )
                try:
                    models.append(
                        _train_one_set(Xs[idx], Ys[idx])
                    )
                    successful += 1
                except Exception as exc:
                    if exc.__class__.__name__ != "ModelFittingError":
                        raise
                    failed_draws += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if successful < int(bags):
                raise RuntimeError(
                    f"IND bagging produced only {successful}/{bags} successful "
                    f"bootstrap models after {draws} draws. "
                    "Consider inspecting duplicate/near-duplicate X values."
                )

            if failed_draws > 0:
                print(
                    f"[bagging:IND] resampled {failed_draws} failed bootstrap "
                    f"draw(s); retained exactly {bags} successful bags."
                )

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