pf_path = save_pf_store(
    pf_store,
    save_dir="pareto_results",
    filename="li_pf_store.pkl",
    metadata={
        "dataset": "Li",
        "frameworks": ["raw", "raw_uncertainty", "cal"],
        "beta": 1,
        "pop_size": 256,
        "n_gen": 150,
        "seeds": range(10),
        "sense": ["min", "min"],
        "use_hull": False,
    },
)