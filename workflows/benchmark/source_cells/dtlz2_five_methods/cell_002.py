# ============================================================
# Global experiment settings
# ============================================================
BENCHMARK_NAME = "DTLZ2"

d = 6
FEATURES = [f"x{i+1}" for i in range(d)]
TARGETS = ["f1", "f2"]
sense = ["min", "min"]

lb = np.zeros(d)
ub = np.ones(d)
bounds = np.column_stack([lb, ub])  # required shape: (d, 2)

# Main manuscript-style run
N_POOL = 30000
N_TRAIN = 50
SEEDS = range(20)

# External test points used for surrogate/calibration metrics
# Lower this to 1000 for a quick local check.
N_EVAL_METRIC = 3000

# GP / bagging / calibration settings
# If your computer is slow, set BAGS=2 first.
BAGS = 20
K_FOLDS = 5
SPLIT_CAL_FRAC = 0.20
TARGET_COVERAGE = 0.90

# NSGA-II surrogate Pareto search settings
# If your computer is slow, use pop_size=64, n_gen=40 first.
NSGA2_POP_SIZE = 96
NSGA2_N_GEN = 60

# Five methods for reviewer-requested ablation.
# The first three isolate surrogate-construction effects.
# The last two use the same selected + bagging base model and compare calibration strategies.
METHOD_CONFIGS = [
    {
        "name": "Vanilla GP",
        "use_model_selection": False,
        "fixed_family": "IND",
        "fixed_kernel": ("RBF", None),
        "bags": 0,
        "calibration_type": "none",
        "use_calibration": False,
    },
    {
        "name": "Selected GP/ICM",
        "use_model_selection": True,
        "fixed_family": "IND",
        "fixed_kernel": ("RBF", None),
        "bags": 0,
        "calibration_type": "none",
        "use_calibration": False,
    },
    {
        "name": "Selected + Bagging",
        "use_model_selection": True,
        "fixed_family": "IND",
        "fixed_kernel": ("RBF", None),
        "bags": BAGS,
        "calibration_type": "none",
        "use_calibration": False,
    },
    {
        "name": "Selected + Bagging + Split CP",
        "use_model_selection": True,
        "fixed_family": "IND",
        "fixed_kernel": ("RBF", None),
        "bags": BAGS,
        "calibration_type": "split",
        "use_calibration": True,
    },
    {
        "name": "Selected + Bagging + CV+",
        "use_model_selection": True,
        "fixed_family": "IND",
        "fixed_kernel": ("RBF", None),
        "bags": BAGS,
        "calibration_type": "cvplus",
        "use_calibration": True,
    },
]

OUTPUT_DIR = Path("results/benchmark_DTLZ2_5methods_10seeds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
