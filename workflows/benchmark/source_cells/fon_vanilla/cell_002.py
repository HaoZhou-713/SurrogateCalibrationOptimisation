import os
import time
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
import gpytorch
import matplotlib.pyplot as plt

# Keep local machine responsive
try:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except Exception:
    pass