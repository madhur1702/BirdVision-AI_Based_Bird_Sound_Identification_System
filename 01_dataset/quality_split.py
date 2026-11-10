import os
import shutil
import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"D:\project1\bird_dataset"

OUTPUT_DIR = r"D:\project1\bird_dataset_split"

METADATA_FILE = os.path.join(
    DATASET_DIR,
    "selected_2000_metadata.csv"
)

QUALITY_FILE = os.path.join(
    OUTPUT_DIR,
    "audio_quality_report.csv"
)

# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(METADATA_FILE)

quality_df = pd.read_csv(QUALITY_FILE)

print("Original recordings:", len(df))
print("Quality records:", len(quality_df))

# ============================================================
# NORMALIZE FILENAMES
# ============================================================

# The quality report contains only the basename.
# Do the same for the original metadata.

df["filename_only"] = (
    df["filename"]
    .astype(str)
    .apply(os.path.basename)
)

quality_df["filename_only"] = (
    quality_df["filename"]
    .astype(str)
    .apply(os.path.basename)
)

# ============================================================
# KEEP ONLY GOOD RECORDINGS
# ============================================================

good_quality = quality_df[
    quality_df["status"] == "GOOD"
].copy()

print(
    "\nGood recordings:",
    len(good_quality)
)

print(
    "Review recordings:",
    len(
        quality_df[
            quality_df["status"] == "REVIEW"
        ]
    )
)

# ============================================================
# MERGE CORRECTLY
# ============================================================

# Merge using:
# filename + common_name

good_df = df.merge(
    good_quality[
        [
            "filename_only",
            "common_name",
            "status",
            "duration",
            "rms_db"
        ]
    ],
    on=[
        "filename_only",
        "common_name"
    ],
    how="inner"
)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

good_df = good_df.drop_duplicates(
    subset=[
        "common_name",
        "filename_only"
    ]
)

print(
    "Good recordings matched to metadata:",
    len(good_df)
)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRAIN_DIR = os.path.join(
    OUTPUT_DIR,
    "train"
)

VAL_DIR = os.path.join(
    OUTPUT_DIR,
    "validation"
)

TEST_DIR = os.path.join(
    OUTPUT_DIR,
    "test"
)

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:
    os.makedirs(
        folder,
        exist_ok=True
    )

# ============================================================
# SPLIT DATA
# ============================================================

train_records = []
val_records = []
test_records = []

print("\n")
print("=" * 70)
print("CREATING SPLIT")
print("=" * 70)

for species in sorted(
    good_df["common_name"].unique()
):

    species_df = good_df[
        good_df["common_name"] == species
    ].copy()

    # Shuffle
    species_df = species_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    total = len(species_df)

    # --------------------------------------------------------
    # We originally selected 100 recordings per species.
    #
    # We use:
    # 70 Train
    # 15 Validation
    # 15 Test
    # --------------------------------------------------------

    train_count = int(total * 0.70)

    val_count = int(total * 0.15)

    train = species_df[
        :train_count
    ]

    validation = species_df[
        train_count:
        train_count + val_count
    ]

    test = species_df[
        train_count + val_count:
    ]

    # --------------------------------------------------------
    # Copy function
    # --------------------------------------------------------

    def copy_recordings(
        records,
        destination_root,
        record_list
    ):

        folder_name = (
            species
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )

        destination_species = os.path.join(
            destination_root,
            folder_name
        )

        os.makedirs(
            destination_species,
            exist_ok=True
        )

        for _, row in records.iterrows():

            filename = row["filename_only"]

            source_file = os.path.join(
                DATASET_DIR,
                folder_name,
                filename
            )

            destination_file = os.path.join(
                destination_species,
                filename
            )

            if not os.path.isfile(source_file):

                print(
                    "WARNING - File missing:",
                    source_file
                )

                continue

            shutil.copy2(
                source_file,
                destination_file
            )

            record_list.append(row)

    # --------------------------------------------------------
    # Copy train
    # --------------------------------------------------------

    copy_recordings(
        train,
        TRAIN_DIR,
        train_records
    )

    # --------------------------------------------------------
    # Copy validation
    # --------------------------------------------------------

    copy_recordings(
        validation,
        VAL_DIR,
        val_records
    )

    # --------------------------------------------------------
    # Copy test
    # --------------------------------------------------------

    copy_recordings(
        test,
        TEST_DIR,
        test_records
    )

    print(
        f"{species}: "
        f"Train={len(train)}, "
        f"Validation={len(validation)}, "
        f"Test={len(test)}"
    )

# ============================================================
# SAVE METADATA
# ============================================================

train_df = pd.DataFrame(train_records)

val_df = pd.DataFrame(val_records)

test_df = pd.DataFrame(test_records)

train_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_metadata.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_metadata.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_metadata.csv"
    ),
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)

print(
    "Total:",
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("\nOutput:")
print(OUTPUT_DIR)