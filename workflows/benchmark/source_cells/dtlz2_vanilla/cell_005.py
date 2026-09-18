# ============================================================
# True Vanilla GP
# Independent single-output GP per objective
# Fixed RBF kernel
# No model selection
# No ICM
# No bagging
# No conformal calibration
# ============================================================

class ExactRBFSingleOutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class VanillaIndependentRBFGP:
    def __init__(self, train_iters=120, lr=0.08, device=torch.device("cpu"), dtype=torch.double, verbose=False):
        self.train_iters = int(train_iters)
        self.lr = float(lr)
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.models = []
        self.likelihoods = []
        self.n_targets = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.n_targets = Y.shape[1]

        # X min-max normalisation
        self.x_min = X.min(axis=0)
        self.x_span = X.max(axis=0) - self.x_min
        self.x_span[self.x_span < 1e-12] = 1.0
        Xs = (X - self.x_min) / self.x_span

        # Y standardisation
        self.y_mean = Y.mean(axis=0)
        self.y_std = Y.std(axis=0)
        self.y_std[self.y_std < 1e-12] = 1.0
        Ys = (Y - self.y_mean) / self.y_std

        Xt = torch.tensor(Xs, device=self.device, dtype=self.dtype)

        self.models = []
        self.likelihoods = []

        for j in range(self.n_targets):
            yt = torch.tensor(Ys[:, j], device=self.device, dtype=self.dtype)

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=self.device, dtype=self.dtype)
            model = ExactRBFSingleOutputGP(Xt, yt, likelihood).to(device=self.device, dtype=self.dtype)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for it in range(self.train_iters):
                optimizer.zero_grad()
                output = model(Xt)
                loss = -mll(output, yt)
                loss.backward()
                optimizer.step()

                if self.verbose and (it == 0 or (it + 1) % 50 == 0):
                    print(f"target={j}, iter={it+1}, loss={float(loss):.4f}")

            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_min) / self.x_span
        Xt = torch.tensor(Xs, device=self.device, dtype=self.dtype)

        means = []
        stds = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for j, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
                pred = likelihood(model(Xt))
                mu_s = pred.mean.detach().cpu().numpy()
                var_s = pred.variance.detach().cpu().numpy()
                var_s = np.maximum(var_s, 1e-12)

                mu = mu_s * self.y_std[j] + self.y_mean[j]
                std = np.sqrt(var_s) * self.y_std[j]

                means.append(mu)
                stds.append(std)

        mu = np.stack(means, axis=1)
        std = np.stack(stds, axis=1)
        return mu, std


def surrogate_accuracy_and_calibration_metrics(y_true, mu, std, target_coverage=0.90):
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)

    err = mu - y_true
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)

    y_range = np.ptp(y_true, axis=0) + 1e-12
    nrmse = rmse / y_range
    nmae = mae / y_range

    ss_res = np.sum((y_true - mu) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    z = z_value_for_two_sided_interval(target_coverage)
    lo = mu - z * std
    hi = mu + z * std
    covered = (y_true >= lo) & (y_true <= hi)
    coverage = np.mean(covered, axis=0)
    coverage_error = np.abs(coverage - target_coverage)

    width = hi - lo
    mpiw = np.mean(width, axis=0)
    nmpiw = mpiw / y_range

    alpha = 1.0 - target_coverage
    interval_score = width.copy()
    interval_score += (2.0 / alpha) * (lo - y_true) * (y_true < lo)
    interval_score += (2.0 / alpha) * (y_true - hi) * (y_true > hi)
    interval_score_mean = np.mean(interval_score, axis=0)
    ninterval_score = interval_score_mean / y_range

    nll = 0.5 * np.log(2.0 * np.pi * std ** 2) + 0.5 * ((y_true - mu) / std) ** 2
    nll_mean = np.mean(nll, axis=0)

    # calibration ECE over central Gaussian intervals
    levels = np.linspace(0.1, 0.9, 9)
    ece = []
    for j in range(y_true.shape[1]):
        gaps = []
        for lev in levels:
            zlev = z_value_for_two_sided_interval(float(lev))
            lo_j = mu[:, j] - zlev * std[:, j]
            hi_j = mu[:, j] + zlev * std[:, j]
            cov_j = np.mean((y_true[:, j] >= lo_j) & (y_true[:, j] <= hi_j))
            gaps.append(abs(cov_j - lev))
        ece.append(np.mean(gaps))
    ece = np.asarray(ece)

    out = {}
    for j in range(y_true.shape[1]):
        name = f"f{j+1}"
        out[f"{name}_RMSE"] = float(rmse[j])
        out[f"{name}_nRMSE"] = float(nrmse[j])
        out[f"{name}_MAE"] = float(mae[j])
        out[f"{name}_nMAE"] = float(nmae[j])
        out[f"{name}_R2"] = float(r2[j])
        out[f"{name}_Coverage"] = float(coverage[j])
        out[f"{name}_Coverage_error"] = float(coverage_error[j])
        out[f"{name}_ECE"] = float(ece[j])
        out[f"{name}_MPIW"] = float(mpiw[j])
        out[f"{name}_nMPIW"] = float(nmpiw[j])
        out[f"{name}_IntervalScore"] = float(interval_score_mean[j])
        out[f"{name}_nIntervalScore"] = float(ninterval_score[j])
        out[f"{name}_NLL"] = float(nll_mean[j])

    out["RMSE_mean"] = float(np.mean(rmse))
    out["nRMSE_mean"] = float(np.mean(nrmse))
    out["MAE_mean"] = float(np.mean(mae))
    out["nMAE_mean"] = float(np.mean(nmae))
    out["R2_mean"] = float(np.mean(r2))
    out["Coverage_mean"] = float(np.mean(coverage))
    out["Coverage_error_mean"] = float(np.mean(coverage_error))
    out["ECE"] = float(np.mean(ece))
    out["MPIW_mean"] = float(np.mean(mpiw))
    out["nMPIW_mean"] = float(np.mean(nmpiw))
    out["IntervalScore_mean"] = float(np.mean(interval_score_mean))
    out["nIntervalScore_mean"] = float(np.mean(ninterval_score))
    out["NLL_mean"] = float(np.mean(nll_mean))

    return out