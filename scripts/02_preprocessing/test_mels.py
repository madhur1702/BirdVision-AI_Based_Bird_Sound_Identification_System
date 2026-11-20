import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"D:\project1\bird_chunks"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# Number of chunks to test
NUM_TEST_FILES = 5

# ============================================================
# FIND TEST AUDIO FILES
# ============================================================

audio_files = []

for split in ["train", "validation", "test"]:

    split_dir = os.path.join(BASE_DIR, split)

    for root, dirs, files in os.walk(split_dir):

        for file in files:

            if file.lower().endswith(".wav"):

                audio_files.append(
                    os.path.join(root, file)
                )

            if len(audio_files) >= NUM_TEST_FILES:
                break

        if len(audio_files) >= NUM_TEST_FILES:
            break

    if len(audio_files) >= NUM_TEST_FILES:
        break

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 60)
print("MEL SPECTROGRAM TEST")
print("=" * 60)

print(f"Files selected: {len(audio_files)}")

# ============================================================
# PROCESS
# ============================================================

for i, audio_path in enumerate(audio_files):

    print("\n" + "-" * 60)
    print(f"File {i + 1}")
    print(os.path.basename(audio_path))

    # Load audio
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    print(f"Sample rate : {sr}")
    print(f"Samples     : {len(audio)}")
    print(
        f"Duration    : {len(audio) / sr:.2f} sec"
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert power spectrogram to dB
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    print(f"Mel shape   : {mel_db.shape}")
    print(
        f"Min value   : {mel_db.min():.2f} dB"
    )
    print(
        f"Max value   : {mel_db.max():.2f} dB"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(
        f"Mel Spectrogram - "
        f"{os.path.basename(audio_path)}"
    )

    plt.tight_layout()
    plt.show()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)