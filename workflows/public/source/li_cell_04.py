DATA_PATH = "../battery_data_Li.xlsx"
FEATURES = ["d1","d2","d3","d4", "v"]
TARGETS  = ["V", "TD"]   # add "TSD" if present

df = pd.read_excel(DATA_PATH)
expected = set(FEATURES + TARGETS)
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"Excel must contain columns: {missing}")