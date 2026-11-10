import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"D:\project1\bird_chunks"
OUTPUT_DIR = r"D:\project1\bird_mels"

# ============================================================
# MEL SPECTROGRAM SETTINGS
# ============================================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FIND ALL AUDIO FILES
# ============================================================

audio_files = []

for split in ["train", "validation", "test"]:

    split_dir = os.path.join(INPUT_DIR, split)

    for species in os.listdir(split_dir):

        species_dir = os.path.join(split_dir, species)

        if not os.path.isdir(species_dir):
            continue

        for filename in os.listdir(species_dir):

            if filename.lower().endswith(".wav"):

                audio_files.append(
                    (
                        split,
                        species,
                        filename,
                        os.path.join(species_dir, filename)
                    )
                )

print("=" * 65)
print("MEL SPECTROGRAM GENERATION")
print("=" * 65)

print(f"Audio chunks found: {len(audio_files)}")

# ============================================================
# PROCESS
# ============================================================

metadata = []

failed = 0

for index, (split, species, filename, audio_path) in enumerate(
    tqdm(audio_files, desc="Generating Mel spectrograms", unit="file")
):

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # Generate Mel spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # ----------------------------------------------------
        # Convert to dB
        # ----------------------------------------------------

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        output_species_dir = os.path.join(
            OUTPUT_DIR,
            split,
            species
        )

        os.makedirs(
            output_species_dir,
            exist_ok=True
        )

        output_filename = (
            os.path.splitext(filename)[0] + ".npy"
        )

        output_path = os.path.join(
            output_species_dir,
            output_filename
        )

        # ----------------------------------------------------
        # Save Mel spectrogram
        # ----------------------------------------------------

        np.save(
            output_path,
            mel_db.astype(np.float32)
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata.append({
            "split": split,
            "species": species,
            "audio_filename": filename,
            "mel_filename": output_filename,
            "shape": str(mel_db.shape)
        })

    except Exception as e:

        failed += 1

        print(
            f"\nERROR: {audio_path}"
        )

        print(f"Reason: {e}")

# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "mel_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MEL GENERATION COMPLETE")
print("=" * 65)

print(f"Successful: {len(metadata_df)}")
print(f"Failed:     {failed}")
print(f"Total:      {len(audio_files)}")

if len(metadata_df) > 0:

    print("\nBy split:")

    print(
        metadata_df["split"]
        .value_counts()
        .sort_index()
    )

    print("\nMel shape:")

    print(
        metadata_df["shape"]
        .value_counts()
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nMetadata:")
print(metadata_path)

print("\nDone.")