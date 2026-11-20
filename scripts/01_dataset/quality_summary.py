import pandas as pd
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = r"D:\project1\bird_dataset"
REPORT_PATH = r"D:\project1\bird_dataset_split\audio_quality_report.csv"
METADATA_PATH = os.path.join(BASE_DIR, "selected_2000_metadata.csv")

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
metadata = pd.read_csv(METADATA_PATH)
quality = pd.read_csv(REPORT_PATH)

# Normalize filename
metadata["filename_only"] = metadata["filename"].apply(
    lambda x: os.path.basename(str(x))
)

quality["filename_only"] = quality["filename"].apply(
    lambda x: os.path.basename(str(x))
)

# --------------------------------------------------
# MERGE
# --------------------------------------------------
df = metadata.merge(
    quality[["filename_only", "status"]],
    on="filename_only",
    how="left"
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
summary = (
    df.groupby("common_name")["status"]
    .value_counts()
    .unstack(fill_value=0)
)

# Make sure both columns exist
if "GOOD" not in summary.columns:
    summary["GOOD"] = 0

if "REVIEW" not in summary.columns:
    summary["REVIEW"] = 0

summary["TOTAL"] = summary["GOOD"] + summary["REVIEW"]

summary = summary.sort_index()

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------
print("=" * 75)
print("QUALITY SUMMARY BY SPECIES")
print("=" * 75)

print(
    summary[["GOOD", "REVIEW", "TOTAL"]].to_string()
)

print("\n" + "=" * 75)
print("TOTAL")
print("=" * 75)

print(f"GOOD   : {summary['GOOD'].sum()}")
print(f"REVIEW : {summary['REVIEW'].sum()}")
print(f"TOTAL  : {summary['TOTAL'].sum()}")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
output_path = r"D:\project1\bird_dataset_split\quality_summary.csv"

summary[["GOOD", "REVIEW", "TOTAL"]].to_csv(output_path)

print("\nSaved to:")
print(output_path)