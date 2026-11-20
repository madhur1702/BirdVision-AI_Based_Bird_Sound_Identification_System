import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\project1\bird_dataset_split_all"
OUTPUT_DIR = r"D:\project1\bird_chunks"

# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 22050

CHUNK_DURATION = 5.0
OVERLAP = 2.5

CHUNK_LENGTH = int(SAMPLE_RATE * CHUNK_DURATION)
HOP_LENGTH = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP))

# Chunks below this RMS level are treated as silent
SILENCE_DB = -50

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_rms_db(audio):

    rms = np.sqrt(np.mean(audio ** 2))

    if rms <= 0:
        return -100.0

    return 20 * np.log10(rms)


def process_recording(input_path, output_species_dir,
                      species, split, original_filename):

    try:

        audio, sr = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        original_duration = len(audio) / SAMPLE_RATE

        # ----------------------------------------------------
        # Handle empty audio
        # ----------------------------------------------------

        if len(audio) == 0:
            return []

        # ----------------------------------------------------
        # If recording is shorter than 5 sec,
        # pad it to 5 sec
        # ----------------------------------------------------

        if len(audio) < CHUNK_LENGTH:

            audio = np.pad(
                audio,
                (0, CHUNK_LENGTH - len(audio)),
                mode="constant"
            )

        # ----------------------------------------------------
        # Generate chunks
        # ----------------------------------------------------

        chunks = []

        start = 0
        chunk_id = 0

        while start < len(audio):

            end = start + CHUNK_LENGTH

            chunk = audio[start:end]

            # Pad final chunk
            if len(chunk) < CHUNK_LENGTH:

                chunk = np.pad(
                    chunk,
                    (0, CHUNK_LENGTH - len(chunk)),
                    mode="constant"
                )

            # ------------------------------------------------
            # Silence check
            # ------------------------------------------------

            rms_db = calculate_rms_db(chunk)

            if rms_db > SILENCE_DB:

                output_filename = (
                    f"{os.path.splitext(original_filename)[0]}"
                    f"_chunk_{chunk_id:04d}.wav"
                )

                output_path = os.path.join(
                    output_species_dir,
                    output_filename
                )

                sf.write(
                    output_path,
                    chunk,
                    SAMPLE_RATE
                )

                chunks.append({
                    "split": split,
                    "species": species,
                    "original_filename": original_filename,
                    "chunk_filename": output_filename,
                    "chunk_id": chunk_id,
                    "start_sec": round(start / SAMPLE_RATE, 3),
                    "end_sec": round(
                        min(end, len(audio)) / SAMPLE_RATE,
                        3
                    ),
                    "original_duration_sec": round(
                        original_duration,
                        3
                    ),
                    "rms_db": round(rms_db, 2)
                })

            chunk_id += 1
            start += HOP_LENGTH

            # Stop if we've reached the end
            if start >= len(audio):
                break

        return chunks

    except Exception as e:

        print(
            f"\nERROR processing {input_path}: {e}"
        )

        return []


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("5-SECOND AUDIO CHUNKING")
print("=" * 65)

print(f"Source: {BASE_DIR}")
print(f"Output: {OUTPUT_DIR}")

print("\nSettings:")
print(f"Sample rate : {SAMPLE_RATE}")
print(f"Chunk size  : {CHUNK_DURATION} sec")
print(f"Overlap     : {OVERLAP} sec")
print(f"Silence     : {SILENCE_DB} dB")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_metadata = []

total_recordings = 0
total_chunks = 0
silent_chunks = 0

# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in ["train", "validation", "test"]:

    split_dir = os.path.join(
        BASE_DIR,
        split
    )

    output_split_dir = os.path.join(
        OUTPUT_DIR,
        split
    )

    os.makedirs(
        output_split_dir,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 65)

    if not os.path.exists(split_dir):

        print(f"Split folder not found: {split_dir}")
        continue

    # --------------------------------------------------------
    # Find species folders
    # --------------------------------------------------------

    species_list = sorted([
        x for x in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, x))
    ])

    # --------------------------------------------------------
    # Process species
    # --------------------------------------------------------

    for species in species_list:

        species_dir = os.path.join(
            split_dir,
            species
        )

        output_species_dir = os.path.join(
            output_split_dir,
            species
        )

        os.makedirs(
            output_species_dir,
            exist_ok=True
        )

        audio_files = [
            f for f in os.listdir(species_dir)
            if f.lower().endswith(".ogg")
        ]

        print(
            f"\n{species}: {len(audio_files)} recordings"
        )

        for filename in tqdm(
            audio_files,
            desc=species,
            unit="file"
        ):

            input_path = os.path.join(
                species_dir,
                filename
            )

            chunks = process_recording(
                input_path,
                output_species_dir,
                species,
                split,
                filename
            )

            all_metadata.extend(chunks)

            total_recordings += 1
            total_chunks += len(chunks)

# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(all_metadata)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "chunk_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CHUNKING COMPLETE")
print("=" * 65)

print(f"Original recordings processed : {total_recordings}")
print(f"Valid chunks created          : {total_chunks}")

print("\nChunks by split:")

if len(metadata_df) > 0:

    print(
        metadata_df["split"]
        .value_counts()
        .sort_index()
    )

    print("\nChunks by species:")

    species_counts = (
        metadata_df["species"]
        .value_counts()
        .sort_index()
    )

    print(species_counts)

print("\nMetadata:")
print(metadata_path)

print("\nOutput:")
print(OUTPUT_DIR)

print("\nDone.")

