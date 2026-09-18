n, N = 6, 50                     # 维度与样本数（小数据集）
# lb, ub = 0, 1                 # 变量范围
# eng = SobolEngine(dimension=n, scramble=True, seed=7)
# U = eng.draw(N).numpy()
# X = lb + (ub - lb) * U
# np.random.seed(7)
# # noise = np.random.normal(0.0, 0.1, size=(N,2))
# Y = benchmark_dtlz2(X) # + noise

FEATURES = [f"x{i+1}" for i in range(n)]
TARGETS  = ["f1", "f2"]
# df = pd.DataFrame(np.column_stack([X, Y]), columns=FEATURES + TARGETS)