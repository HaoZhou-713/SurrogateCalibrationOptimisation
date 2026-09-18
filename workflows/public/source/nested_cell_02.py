# Optional installs if running in a fresh environment (uncomment to use)
# %pip install -U torch gpytorch botorch numpy pandas scikit-learn matplotlib
# 标准库
import os, random, math, json
from pathlib import Path

# 科学计算 / 绘图
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler  # 或 MinMaxScaler，二选一

# BoTorch / GPyTorch
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.transforms import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

# 仅当需要自定义核或先验时再保留
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.priors import LogNormalPrior, GammaPrior
import gpytorch

# ====================== Auto GP for Multi-output (ICM vs IND, per-target kernel) ======================
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Sequence, Union

from torch.quasirandom import SobolEngine