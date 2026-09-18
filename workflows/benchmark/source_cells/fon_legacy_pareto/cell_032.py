d = 4
sense = ["min", "min"]

lb = -np.ones(d)
ub = np.ones(d)

gt_dict = build_gt_dict(
    d=d,
    sense=sense,
    lb=lb,
    ub=ub,
    N_pool=30000,   # test 阶段先小
    seed=0,
)