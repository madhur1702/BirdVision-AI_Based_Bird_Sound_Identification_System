import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"D:\project1\bird_dataset"
OUTPUT_DIR = r"D:\project1\bird_dataset_split_all"

METADATA_PATH = os.path.join(
    SOURCE_DIR,
    "selected_2000_metadata.csv"
)

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("REBUILDING DATASET SPLIT")
print("=" * 60)

print(f"Total recordings in metadata: {len(df)}")
print(f"Number of species: {df['common_name'].nunique()}")

# ============================================================
# CLEAN OLD OUTPUT
# ============================================================

if os.path.exists(OUTPUT_DIR):
    print("\nRemoving previous split folder...")
    shutil.rmtree(OUTPUT_DIR)

# Create folders
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ============================================================
# STORE SPLIT METADATA
# ============================================================

train_records = []
val_records = []
test_records = []

# ============================================================
# SPLIT SPECIES-WISE
# ============================================================

for species in sorted(df["common_name"].unique()):

    species_df = df[df["common_name"] == species].copy()

    # Safety check
    if len(species_df) != 100:
        print(
            f"WARNING: {species} has {len(species_df)} recordings "
            f"instead of 100."
        )

    # --------------------------------------------------------
    # 70% TRAIN
    # 30% TEMP
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        species_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # --------------------------------------------------------
    # TEMP -> 15% VALIDATION + 15% TEST
    # --------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    for split_name, split_df, records_list in [
        ("train", train_df, train_records),
        ("validation", val_df, val_records),
        ("test", test_df, test_records)
    ]:

        species_folder = os.path.join(
            OUTPUT_DIR,
            split_name,
            species
        )

        os.makedirs(species_folder, exist_ok=True)

        for _, row in split_df.iterrows():

            filename = os.path.basename(
                str(row["filename"])
            )

            source_file = os.path.join(
                SOURCE_DIR,
                species,
                filename
            )

            destination_file = os.path.join(
                species_folder,
                filename
            )

            if not os.path.exists(source_file):
                print(
                    f"WARNING: File not found: {source_file}"
                )
                continue

            shutil.copy2(
                source_file,
                destination_file
            )

            record = row.to_dict()
            record["split"] = split_name
            record["filename_only"] = filename

            records_list.append(record)

    print(
        f"{species}: "
        f"Train={len(train_df)}, "
        f"Validation={len(val_df)}, "
        f"Test={len(test_df)}"
    )

# ============================================================
# SAVE METADATA
# ============================================================

train_metadata = pd.DataFrame(train_records)
val_metadata = pd.DataFrame(test_records if False else val_records)
test_metadata = pd.DataFrame(test_records)

train_metadata.to_csv(
    os.path.join(OUTPUT_DIR, "train_metadata.csv"),
    index=False
)

val_metadata.to_csv(
    os.path.join(OUTPUT_DIR, "validation_metadata.csv"),
    index=False
)

test_metadata.to_csv(
    os.path.join(OUTPUT_DIR, "test_metadata.csv"),
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print(f"Train:      {len(train_metadata)}")
print(f"Validation: {len(val_metadata)}")
print(f"Test:       {len(test_metadata)}")
print(
    f"Total:      "
    f"{len(train_metadata) + len(val_metadata) + len(test_metadata)}"
)

print("\nOutput:")
print(OUTPUT_DIR)

print("\nMetadata files:")
print("train_metadata.csv")
print("validation_metadata.csv")
print("test_metadata.csv")