import os
import random
import numpy as np
import pandas as pd

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"D:\project1\bird_mels"
METADATA_PATH = os.path.join(BASE_DIR, "mel_metadata.csv")

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

print("=" * 65)
print("MEL DATASET VERIFICATION")
print("=" * 65)

print(f"\nMetadata records: {len(df)}")

# ============================================================
# 1. CHECK EXPECTED TOTAL
# ============================================================

print("\n[1] TOTAL RECORD CHECK")

actual_files = 0

for split in ["train", "validation", "test"]:

    split_dir = os.path.join(BASE_DIR, split)

    count = 0

    for root, dirs, files in os.walk(split_dir):
        count += sum(
            1 for f in files
            if f.lower().endswith(".npy")
        )

    print(f"{split:12}: {count}")
    actual_files += count

print(f"Total .npy files: {actual_files}")

if actual_files == len(df):
    print("PASS: File count matches metadata.")
else:
    print("WARNING: File count does NOT match metadata.")

# ============================================================
# 2. CHECK SPLIT DISTRIBUTION
# ============================================================

print("\n[2] SPLIT DISTRIBUTION")

print(
    df["split"]
    .value_counts()
    .sort_index()
)

# ============================================================
# 3. CHECK SPECIES
# ============================================================

print("\n[3] SPECIES CHECK")

species = sorted(df["species"].unique())

print(f"Number of species: {len(species)}")

for s in species:
    print(f"  {s}")

if len(species) == 20:
    print("PASS: 20 species found.")
else:
    print("WARNING: Expected 20 species.")

# ============================================================
# 4. SPECIES × SPLIT DISTRIBUTION
# ============================================================

print("\n[4] SPECIES × SPLIT")

distribution = pd.crosstab(
    df["species"],
    df["split"]
)

print(distribution)

# ============================================================
# 5. CHECK MEL SHAPES
# ============================================================

print("\n[5] MEL SHAPE CHECK")

print(
    df["shape"]
    .value_counts()
)

if (
    len(df["shape"].unique()) == 1
    and df["shape"].iloc[0] == "(128, 216)"
):
    print("PASS: All metadata shapes are 128 × 216.")
else:
    print("WARNING: Unexpected shapes found.")

# ============================================================
# 6. LOAD RANDOM FILES
# ============================================================

print("\n[6] RANDOM FILE CHECK")

sample = df.sample(
    n=min(10, len(df)),
    random_state=42
)

successful = 0

for _, row in sample.iterrows():

    path = os.path.join(
        BASE_DIR,
        row["split"],
        row["species"],
        row["mel_filename"]
    )

    if not os.path.exists(path):

        print(f"NOT FOUND: {path}")
        continue

    mel = np.load(path)

    print(
        f"{row['species']:<30} "
        f"{row['mel_filename']:<45} "
        f"shape={mel.shape} "
        f"range=({mel.min():.2f}, {mel.max():.2f})"
    )

    if mel.shape == (128, 216):
        successful += 1

print(
    f"\nValid random samples: "
    f"{successful}/{len(sample)}"
)

# ============================================================
# 7. CHECK DUPLICATE ORIGINAL RECORDINGS
# ============================================================

print("\n[7] DATA LEAKAGE CHECK")

# Extract original filename from chunk filename.
# Example:
# XC142466_chunk_0000.wav
# becomes XC142466
#
# We use the metadata's audio filename where available.

if "audio_filename" in df.columns:

    leakage = (
        df.groupby("audio_filename")["split"]
        .nunique()
    )

    leaked = leakage[leakage > 1]

    print(
        f"Original recordings appearing in multiple splits: "
        f"{len(leaked)}"
    )

    if len(leaked) == 0:
        print("PASS: No original recording crosses splits.")
    else:
        print("WARNING: DATA LEAKAGE DETECTED!")

else:

    print(
        "WARNING: audio_filename column not found."
    )

# ============================================================
# 8. CHECK MISSING FILES
# ============================================================

print("\n[8] MISSING FILE CHECK")

missing = 0

for _, row in df.iterrows():

    path = os.path.join(
        BASE_DIR,
        row["split"],
        row["species"],
        row["mel_filename"]
    )

    if not os.path.exists(path):
        missing += 1

print(f"Missing files: {missing}")

if missing == 0:
    print("PASS: No metadata files are missing.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VERIFICATION COMPLETE")
print("=" * 65)

if (
    actual_files == len(df)
    and len(species) == 20
    and missing == 0
    and successful == len(sample)
):
    print("DATASET BASIC CHECKS: PASSED")
else:
    print("DATASET CHECKS: REVIEW REQUIRED")

print("=" * 65)