import os
import shutil
import pandas as pd

# ============================================================
# PATHS
# ============================================================

CSV_PATH = r"D:\BirdClef\train_metadata(1).csv"
SOURCE_DIR = r"D:\BirdClef\train_audio"
OUTPUT_DIR = r"D:\project1\bird_dataset"

# ============================================================
# SETTINGS
# ============================================================

RATING_THRESHOLD = 4.0
RECORDINGS_PER_SPECIES = 100
RANDOM_SEED = 42

# ============================================================
# 20 SELECTED SPECIES
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
# LOAD CSV
# ============================================================

print("Loading metadata...")

df = pd.read_csv(CSV_PATH)

print(f"Total recordings in CSV: {len(df)}")
print(f"Total species in CSV: {df['common_name'].nunique()}")

# Make rating numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SELECT AND COPY RECORDINGS
# ============================================================

all_selected = []

total_copied = 0
total_missing = 0

for species in SPECIES_LIST:

    print("\n" + "=" * 60)
    print(f"Processing: {species}")

    # --------------------------------------------------------
    # Filter species + rating
    # --------------------------------------------------------

    species_df = df[
        (df["common_name"] == species) &
        (df["rating"] >= RATING_THRESHOLD)
    ].copy()

    print(
        f"High-quality recordings available: "
        f"{len(species_df)}"
    )

    if len(species_df) < RECORDINGS_PER_SPECIES:
        print(
            f"ERROR: Not enough recordings for {species}"
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
    # Create species folder
    # --------------------------------------------------------

    # Replace characters that may cause folder problems
    folder_name = species.replace("/", "-")

    species_output_dir = os.path.join(
        OUTPUT_DIR,
        folder_name
    )

    os.makedirs(species_output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Copy audio files
    # --------------------------------------------------------

    copied_for_species = 0

    for _, row in selected.iterrows():

        filename = str(row["filename"])

        # Filename in BirdCLEF normally looks like:
        # species_code/audio_file.ogg
        #
        # Example:
        # abcd123/XC123456.ogg

        source_file = os.path.join(
            SOURCE_DIR,
            filename
        )

        # If metadata filename doesn't directly exist,
        # try only the basename.
        if not os.path.isfile(source_file):

            source_file = os.path.join(
                SOURCE_DIR,
                os.path.basename(filename)
            )

        if not os.path.isfile(source_file):

            print(
                f"  MISSING: {filename}"
            )

            total_missing += 1
            continue

        destination_file = os.path.join(
            species_output_dir,
            os.path.basename(filename)
        )

        shutil.copy2(
            source_file,
            destination_file
        )

        copied_for_species += 1
        total_copied += 1

        # Keep selected metadata
        all_selected.append(row)

    print(
        f"Copied: {copied_for_species}/"
        f"{RECORDINGS_PER_SPECIES}"
    )

# ============================================================
# SAVE SELECTED METADATA
# ============================================================

if all_selected:

    selected_df = pd.DataFrame(all_selected)

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "selected_2000_metadata.csv"
    )

    selected_df.to_csv(
        metadata_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)

    print(
        f"Total audio files copied: {total_copied}"
    )

    print(
        f"Missing files: {total_missing}"
    )

    print(
        f"Metadata saved at:\n{metadata_path}"
    )

    print(
        f"\nDataset location:\n{OUTPUT_DIR}"
    )

else:

    print(
        "\nNo files were copied. "
        "Please check your paths and filenames."
    )
