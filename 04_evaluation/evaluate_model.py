import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# PATHS
# ============================================================

# MODEL_PATH = r"D:\project1\models\mobilenetv2_exp2_finetuned_best.keras"
# MODEL_PATH = r"D:\project1\models\mobilenetv2_exp3_finetuned_best.keras"
# MODEL_PATH = r"D:\project1\models\mobilenetv2_exp4_finetuned_best.keras"
MODEL_PATH = r"D:\project1\models\mobilenetv2_exp5_finetuned_best.keras"
MEL_DIR = r"D:\project1\bird_mels"
METADATA_PATH = r"D:\project1\bird_mels\mel_metadata.csv"

# OUTPUT_DIR = r"D:\project1\evaluation_exp2"
# OUTPUT_DIR = r"D:\project1\evaluation_exp3"
OUTPUT_DIR = r"D:\project1\evaluation_exp5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32

# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(METADATA_PATH)

classes = sorted(df["species"].unique())
class_to_idx = {
    name: i for i, name in enumerate(classes)
}

test_df = df[df["split"] == "test"].copy()

print("Total chunks:", len(df))
print("Test chunks:", len(test_df))

print("\nClasses:")
for i, name in enumerate(classes):
    print(f"{i:2d} -> {name}")

# ============================================================
# CREATE MEL PATH
# ============================================================

test_df["mel_path"] = test_df.apply(
    lambda row: os.path.join(
        MEL_DIR,
        row["split"],
        row["species"],
        row["mel_filename"]
    ),
    axis=1
)

missing = test_df[
    ~test_df["mel_path"].apply(os.path.exists)
]

print("\nMissing Mel files:", len(missing))

if len(missing) > 0:
    raise FileNotFoundError(
        "Some Mel files are missing."
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD DATA
# ============================================================

X = []
y = []

print("\nLoading test Mel spectrograms...")

for _, row in test_df.iterrows():

    mel = np.load(row["mel_path"]).astype(
        np.float32
    )

    if mel.shape != (128, 216):
        raise ValueError(
            f"Unexpected shape {mel.shape}"
        )

    X.append(
        mel[..., np.newaxis]
    )

    y.append(
        class_to_idx[row["species"]]
    )

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# CHUNK-LEVEL PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("CHUNK-LEVEL EVALUATION")
print("=" * 60)

predictions = model.predict(
    X,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

chunk_accuracy = accuracy_score(
    y,
    y_pred
)

print(
    f"\nChunk-level Accuracy: "
    f"{chunk_accuracy * 100:.2f}%"
)

# ============================================================
# CHUNK CLASSIFICATION REPORT
# ============================================================

print("\nChunk-level Classification Report:\n")

print(
    classification_report(
        y,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)

chunk_report = classification_report(
    y,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(chunk_report).transpose().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "chunk_classification_report.csv"
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    y_pred,
    labels=list(range(len(classes)))
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "chunk_confusion_matrix.csv"
    )
)

# ============================================================
# CREATE ORIGINAL RECORDING ID
# ============================================================

test_df = test_df.reset_index(drop=True)

test_df["original_recording"] = (
    test_df["audio_filename"]
    .str.split("_chunk_")
    .str[0]
)

# ============================================================
# ADD MODEL PROBABILITIES
# ============================================================

for i, class_name in enumerate(classes):
    test_df[f"prob_{i}"] = predictions[:, i]

# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("RECORDING-LEVEL EVALUATION")
print("=" * 60)

recording_results = []

for recording_id, group in test_df.groupby(
    "original_recording"
):

    # True species
    true_species = group[
        "species"
    ].iloc[0]

    # Average probabilities across all chunks
    mean_probabilities = group[
        [f"prob_{i}" for i in range(len(classes))]
    ].mean()

    # Final prediction
    predicted_idx = int(
        np.argmax(mean_probabilities.values)
    )

    predicted_species = classes[
        predicted_idx
    ]

    recording_results.append({
        "original_recording": recording_id,
        "true_species": true_species,
        "predicted_species": predicted_species,
        "num_chunks": len(group),
        "correct": (
            true_species == predicted_species
        )
    })

recording_df = pd.DataFrame(
    recording_results
)

# ============================================================
# RECORDING ACCURACY
# ============================================================

recording_accuracy = accuracy_score(
    recording_df["true_species"],
    recording_df["predicted_species"]
)

print(
    "\nOriginal recordings evaluated:",
    len(recording_df)
)

print(
    f"Recording-level Accuracy: "
    f"{recording_accuracy * 100:.2f}%"
)

# ============================================================
# RECORDING CLASSIFICATION REPORT
# ============================================================

recording_y_true = [
    class_to_idx[x]
    for x in recording_df["true_species"]
]

recording_y_pred = [
    class_to_idx[x]
    for x in recording_df["predicted_species"]
]

print(
    "\nRecording-level Classification Report:\n"
)

print(
    classification_report(
        recording_y_true,
        recording_y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)

recording_report = classification_report(
    recording_y_true,
    recording_y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(
    recording_report
).transpose().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recording_classification_report.csv"
    )
)

# ============================================================
# RECORDING CONFUSION MATRIX
# ============================================================

recording_cm = confusion_matrix(
    recording_y_true,
    recording_y_pred,
    labels=list(range(len(classes)))
)

pd.DataFrame(
    recording_cm,
    index=classes,
    columns=classes
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recording_confusion_matrix.csv"
    )
)

# ============================================================
# SAVE RECORDING PREDICTIONS
# ============================================================

recording_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recording_predictions.csv"
    ),
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)

print(
    f"Test chunks             : {len(test_df)}"
)

print(
    f"Original recordings     : {len(recording_df)}"
)

print(
    f"Chunk-level accuracy    : "
    f"{chunk_accuracy * 100:.2f}%"
)

print(
    f"Recording-level accuracy: "
    f"{recording_accuracy * 100:.2f}%"
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nEvaluation complete.")