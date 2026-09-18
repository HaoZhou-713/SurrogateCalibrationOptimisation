

# Keep local machine responsive
try:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except Exception:
    pass

device = torch.device("cpu")
dtype = torch.double

RESULT_DIR = Path("results") / BENCHMARK_NAME / "vanilla_gp"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Benchmark:", BENCHMARK_NAME)
print("Result directory:", RESULT_DIR.resolve())