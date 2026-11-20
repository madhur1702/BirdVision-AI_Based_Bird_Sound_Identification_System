import os
import shutil
import pandas as pd

# ============================================================
# PATHS
# ============================================================

CSV_PATH = r"D:\BirdClef\train_metadata.csv"

SOURCE_DIR = r"D:\BirdClef\train_audio"

OUTPUT_DIR = r"D:\project1\bird_dataset"

# ============================================================
# SETTINGS
# ============================================================

RATING_THRESHOLD = 4.0
RECORDINGS_PER_SPECIES = 100
RANDOM_SEED = 42

# ============================================================
# 20 SPECIES
# ============================================================

SPECIES_LIST = [
    "Black-winged Stilt",
    "Common Greenshank",
    "Blyth's Reed Warbler",
    "Wood Sandpiper",
    "Gray Heron",
    "Eurasian Coot",
    "Black-crowned Night-Heron",
    "Barn Swallow",
    "Common Kingfisher",
    "Little Egret",
    "Western Yellow Wagtail",
    "Zitting Cisticola",
    "Green Sandpiper",
    "Little Grebe",
    "Common Sandpiper",
    "Common Rosefinch",
    "Little Ringed Plover",
    "Eurasian Moorhen",
    "Greenish Warbler",
    "Rose-ringed Parakeet"
]

# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("LOADING METADATA")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

print("Total recordings:", len(df))
print("Total species:", df["common_name"].nunique())

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SELECT RECORDINGS
# ============================================================

selected_records = []

total_copied = 0
total_missing = 0

for species in SPECIES_LIST:

    print("\n" + "=" * 70)
    print("SPECIES:", species)
    print("=" * 70)

    # --------------------------------------------------------
    # Filter species + rating >= 4
    # --------------------------------------------------------

    species_df = df[
        (df["common_name"] == species) &
        (df["rating"] >= RATING_THRESHOLD)
    ].copy()

    print(
        "High-rated recordings available:",
        len(species_df)
    )

    if len(species_df) < RECORDINGS_PER_SPECIES:

        print(
            "WARNING: Not enough recordings!"
        )

        continue

    # --------------------------------------------------------
    # Randomly select exactly 100
    # --------------------------------------------------------

    selected = species_df.sample(
        n=RECORDINGS_PER_SPECIES,
        random_state=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Get primary label
    # --------------------------------------------------------

    primary_label = selected.iloc[0]["primary_label"]

    print(
        "Primary label:",
        primary_label
    )

    # --------------------------------------------------------
    # Source folder
    # --------------------------------------------------------

    source_species_folder = os.path.join(
        SOURCE_DIR,
        primary_label
    )

    if not os.path.isdir(source_species_folder):

        print(
            "ERROR: Source folder not found:",
            source_species_folder
        )

        continue

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    # Replace problematic characters
    output_species_name = (
        species
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )

    output_species_folder = os.path.join(
        OUTPUT_DIR,
        output_species_name
    )

    os.makedirs(
        output_species_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    copied = 0

    for _, row in selected.iterrows():

        filename = str(row["filename"])

        # The CSV may contain:
        #
        # asbfly/XC123456.ogg
        #
        # or simply:
        #
        # XC123456.ogg
        #
        # We only need the actual filename.

        filename_only = os.path.basename(filename)

        source_file = os.path.join(
            source_species_folder,
            filename_only
        )

        destination_file = os.path.join(
            output_species_folder,
            filename_only
        )

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if not os.path.isfile(source_file):

            print(
                "MISSING:",
                filename_only
            )

            total_missing += 1

            continue

        # ----------------------------------------------------
        # Copy
        # ----------------------------------------------------

        shutil.copy2(
            source_file,
            destination_file
        )

        copied += 1
        total_copied += 1

        # Save metadata
        selected_records.append(row)

    print(
        f"Copied: {copied}/{RECORDINGS_PER_SPECIES}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

if selected_records:

    selected_df = pd.DataFrame(
        selected_records
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "selected_2000_metadata.csv"
    )

    selected_df.to_csv(
        metadata_path,
        index=False
    )

else:

    print("No recordings were selected.")
    exit()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DATASET CREATION FINISHED")
print("=" * 70)

print(
    "Total files copied:",
    total_copied
)

print(
    "Missing files:",
    total_missing
)

print(
    "Metadata:",
    metadata_path
)

print(
    "Dataset:",
    OUTPUT_DIR
)

print("\nExpected:")
print("20 species × 100 recordings = 2000 recordings")