import pandas as pd
import librosa
import numpy as np
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = r"D:\project1\bird_dataset"
QUALITY_PATH = r"D:\project1\bird_dataset_split\audio_quality_report.csv"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
SAMPLE_RATE = 22050
MIN_DURATION = 1.0
SILENCE_DB = -50

# --------------------------------------------------
# LOAD QUALITY REPORT
# --------------------------------------------------
quality = pd.read_csv(QUALITY_PATH)

# Keep only REVIEW recordings
review = quality[quality["status"] == "REVIEW"].copy()

print("=" * 75)
print("ANALYZING REVIEW RECORDINGS")
print("=" * 75)
print(f"Review recordings: {len(review)}")

# --------------------------------------------------
# ANALYZE
# --------------------------------------------------
results = []

for _, row in review.iterrows():

    filename = os.path.basename(str(row["filename"]))
    species = str(row["common_name"])

    audio_path = os.path.join(
        BASE_DIR,
        species,
        filename
    )

    duration = 0
    rms_db = -100

    try:
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        duration = len(audio) / sr

        if len(audio) > 0:
            rms = np.sqrt(np.mean(audio ** 2))

            if rms > 0:
                rms_db = 20 * np.log10(rms)

        reasons = []

        if duration < MIN_DURATION:
            reasons.append("SHORT")

        if rms_db < SILENCE_DB:
            reasons.append("QUIET")

        if not reasons:
            reasons.append("PASSED")

        reason = "+".join(reasons)

    except Exception as e:
        reason = "ERROR"
        print(f"Error: {filename} -> {e}")

    results.append({
        "common_name": species,
        "filename": filename,
        "duration_sec": round(duration, 2),
        "rms_db": round(rms_db, 2),
        "reason": reason
    })

    if len(results) % 25 == 0:
        print(f"Processed {len(results)}/{len(review)}")

# --------------------------------------------------
# SAVE
# --------------------------------------------------
result_df = pd.DataFrame(results)

output_path = r"D:\project1\bird_dataset_split\review_details.csv"

result_df.to_csv(output_path, index=False)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "=" * 75)
print("REVIEW REASON SUMMARY")
print("=" * 75)

print(
    result_df["reason"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 75)
print("SAVED")
print("=" * 75)

print(output_path)