n, N = 4, 30                     # 维度与样本数（小数据集）

eng = SobolEngine(dimension=n, scramble=True, seed=7)
U = eng.draw(N).numpy()
X = lb + (ub - lb) * U
np.random.seed(7)
# noise = np.random.normal(0.0, 0.1, size=(N,2))
Y = fonseca_fleming(X) # + noise

FEATURES = [f"x{i+1}" for i in range(n)]
TARGETS  = ["f1", "f2"]
df = pd.DataFrame(np.column_stack([X, Y]), columns=FEATURES + TARGETS)