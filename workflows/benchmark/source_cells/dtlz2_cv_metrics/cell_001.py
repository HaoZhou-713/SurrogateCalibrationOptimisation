import os
import time
import gc
import traceback
import copy
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

import matplotlib.pyplot as plt

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

from surrogate_pipeline import AutoConfig, AutoMultiOutputGP
from surrogate_pareto_botorch import (
    surrogate_pareto_front,
    surrogate_pareto_front_nsga2,
    calibrate_from_auto_test,
    calibrate_for_auto,
    calibrate_from_cv_plus,
    pareto_front_botorch,
)

try:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except Exception:
    pass

device = torch.device("cpu")
dtype = torch.double
