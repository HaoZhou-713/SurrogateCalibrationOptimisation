device = torch.device("cpu")
dtype = torch.double

RESULT_DIR = Path("results") / BENCHMARK_NAME / "vanilla_gp"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Benchmark:", BENCHMARK_NAME)
print("Result directory:", RESULT_DIR.resolve())