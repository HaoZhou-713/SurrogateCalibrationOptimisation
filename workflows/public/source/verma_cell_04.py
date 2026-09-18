DATA_PATH = "../battery_data_Verma_modify.xlsx"
FEATURES = ["V","T_in","AR","IA"]
TARGETS  = ["T", "P", "TD"]   # add "TSD" if present

df = pd.read_excel(DATA_PATH)
expected = set(FEATURES + TARGETS)
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"Excel must contain columns: {missing}")