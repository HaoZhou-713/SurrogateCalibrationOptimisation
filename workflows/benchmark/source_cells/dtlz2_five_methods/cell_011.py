# ============================================================
# Save CSV outputs
# ============================================================
results_path = OUTPUT_DIR / "dtlz2_5methods_10seeds_with_surrogate_calibration_metrics.csv"
summary_path = OUTPUT_DIR / "dtlz2_5methods_10seeds_summary_by_method.csv"

# Flatten MultiIndex columns for CSV summary
summary_to_save = summary_by_method.copy()
summary_to_save.columns = ["_".join([str(a), str(b)]).strip("_") for a, b in summary_to_save.columns]

# Save
df_all.to_csv(results_path, index=False)
summary_to_save.to_csv(summary_path)

print("Saved:")
print(results_path)
print(summary_path)
