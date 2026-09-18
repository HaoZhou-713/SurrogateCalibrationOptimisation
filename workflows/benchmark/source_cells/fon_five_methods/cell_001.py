# ============================================================
# Imports
# ============================================================
import os
import gc
import copy
import time
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

import matplotlib.pyplot as plt

from scipy.stats import norm
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from surrogate_pipeline import AutoConfig, AutoMultiOutputGP
from surrogate_pareto_botorch import (
    surrogate_pareto_front_nsga2,
    calibrate_for_auto,
)

warnings.filterwarnings("ignore")

# Avoid freezing local machines
try:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device("cpu")
dtype = torch.double
