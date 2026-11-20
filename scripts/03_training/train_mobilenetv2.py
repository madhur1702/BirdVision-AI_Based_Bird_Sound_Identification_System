import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ============================================================
# EXPERIMENT 5
# Recording-balanced sampling + targeted hard-class sampling
# ============================================================

# -----------------------------
# Paths
# -----------------------------
MEL_DIR = r"D:\project1\bird_mels"
METADATA_FILE = r"D:\project1\bird_mels\mel_metadata.csv"
MODEL_DIR = r"D:\project1\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Configuration
# -----------------------------
IMG_H = 128
IMG_W = 216
NUM_CLASSES = 20
BATCH_SIZE = 32

STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 20

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Difficult classes identified from Experiment 4
HARD_CLASSES = {
    "Eurasian Moorhen",
    "Eurasian Coot",
    "Green Sandpiper",
    "Little Grebe",
    "Black-crowned Night-Heron"
}

# ============================================================
# Load metadata
# ============================================================

print("=" * 70)
print("EXPERIMENT 5")
print("Recording-balanced + targeted hard-class sampling")
print("=" * 70)

df = pd.read_csv(METADATA_FILE)

print("\nMetadata loaded:")
print(df.shape)
print(df.columns.tolist())

# ============================================================
# Create recursive Mel-file index
# ============================================================

print("\nIndexing Mel files...")

mel_file_index = {}

for root, dirs, files in os.walk(MEL_DIR):
    for file in files:
        if file.lower().endswith(".npy"):
            mel_file_index[file] = os.path.join(root, file)

print("Mel files indexed:", len(mel_file_index))

# Resolve paths
df["mel_path"] = df["mel_filename"].map(mel_file_index)

missing = df["mel_path"].isna().sum()

print("Missing Mel files:", missing)

if missing > 0:
    print("\nExample missing files:")
    print(df[df["mel_path"].isna()]["mel_filename"].head(10))
    raise FileNotFoundError("Some Mel files are missing.")

# ============================================================
# Encode species
# ============================================================

classes = sorted(df["species"].unique())

if len(classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes but found {len(classes)}"
    )

class_to_idx = {name: i for i, name in enumerate(classes)}

df["label"] = df["species"].map(class_to_idx)

print("\nClasses:")
for i, name in enumerate(classes):
    print(i, name)

# ============================================================
# Original recording ID
# ============================================================

df["original_recording"] = (
    df["audio_filename"]
    .str.split("_chunk_")
    .str[0]
)

# ============================================================
# Split information
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nSplit sizes:")
print("Train chunks      :", len(train_df))
print("Validation chunks :", len(val_df))
print("Test chunks       :", len(test_df))

# ============================================================
# Recording-level statistics
# ============================================================

print("\nTraining recordings per class:")
print(
    train_df.groupby("species")["original_recording"]
    .nunique()
    .sort_index()
)

# ============================================================
# Create recording -> chunks mapping
# ============================================================

recording_groups = {}

for recording_id, group in train_df.groupby("original_recording"):
    species = group["species"].iloc[0]

    recording_groups[recording_id] = {
        "species": species,
        "chunks": group.index.tolist()
    }

print("\nTraining recordings:", len(recording_groups))

# ============================================================
# Recording-balanced sampler
#
# Every training recording gets approximately equal chance.
# Hard classes receive extra probability.
# ============================================================

recording_ids = list(recording_groups.keys())

recording_weights = []

for recording_id in recording_ids:

    species = recording_groups[recording_id]["species"]

    # Base weight
    weight = 1.0

    # Targeted oversampling
    if species in HARD_CLASSES:
        weight *= 2.0

    recording_weights.append(weight)

recording_weights = np.array(recording_weights, dtype=np.float64)
recording_weights /= recording_weights.sum()

print("\nRecording sampler ready.")

print("\nHard classes receive 2x recording probability:")

for species in sorted(HARD_CLASSES):
    count = sum(
        1
        for r in recording_ids
        if recording_groups[r]["species"] == species
    )
    print(f"{species:35} recordings={count}")

# ============================================================
# Sample one chunk from a selected recording
# ============================================================

def create_balanced_epoch():

    selected_indices = []

    # Same number of samples as original training chunks
    num_samples = len(train_df)

    selected_recordings = np.random.choice(
        recording_ids,
        size=num_samples,
        replace=True,
        p=recording_weights
    )

    for recording_id in selected_recordings:

        chunks = recording_groups[recording_id]["chunks"]

        # Random chunk from that recording
        selected_idx = random.choice(chunks)

        selected_indices.append(selected_idx)

    return train_df.loc[selected_indices].reset_index(drop=True)


# ============================================================
# Mel loading
# ============================================================

def load_mel(path):
    mel = np.load(path).astype(np.float32)

    if mel.shape != (IMG_H, IMG_W):
        raise ValueError(
            f"Unexpected Mel shape {mel.shape} for {path}"
        )

    return mel


# ============================================================
# Augmentation
# ============================================================

def augment_mel(mel):

    mel = mel.copy()

    # -----------------------------
    # Frequency masking
    # -----------------------------
    if random.random() < 0.5:

        max_width = 17

        width = random.randint(0, max_width)

        if width > 0:

            start = random.randint(
                0,
                max(0, IMG_H - width)
            )

            mel[start:start + width, :] = -80.0

    # -----------------------------
    # Time masking
    # -----------------------------
    if random.random() < 0.5:

        max_width = 25

        width = random.randint(0, max_width)

        if width > 0:

            start = random.randint(
                0,
                max(0, IMG_W - width)
            )

            mel[:, start:start + width] = -80.0

    # -----------------------------
    # Small time shift
    # -----------------------------
    if random.random() < 0.5:

        shift = random.randint(-10, 10)

        if shift != 0:

            mel = np.roll(
                mel,
                shift,
                axis=1
            )

    # -----------------------------
    # Small Gaussian noise
    # -----------------------------
    if random.random() < 0.3:

        noise = np.random.normal(
            0,
            0.005,
            mel.shape
        ).astype(np.float32)

        mel = mel + noise

    # Keep expected dB range
    mel = np.clip(
        mel,
        -80.0,
        0.0
    )

    return mel


# ============================================================
# TensorFlow datasets
# ============================================================

def train_generator():

    while True:

        epoch_df = create_balanced_epoch()

        for _, row in epoch_df.iterrows():

            mel = load_mel(row["mel_path"])

            mel = augment_mel(mel)

            mel = np.expand_dims(
                mel,
                axis=-1
            )

            label = int(row["label"])

            yield mel, label


def validation_generator():

    for _, row in val_df.iterrows():

        mel = load_mel(row["mel_path"])

        mel = np.expand_dims(
            mel,
            axis=-1
        )

        label = int(row["label"])

        yield mel, label


# ============================================================
# Dataset objects
# ============================================================

output_signature = (
    tf.TensorSpec(
        shape=(IMG_H, IMG_W, 1),
        dtype=tf.float32
    ),
    tf.TensorSpec(
        shape=(),
        dtype=tf.int32
    )
)

train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=output_signature
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    validation_generator,
    output_signature=output_signature
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = int(
    np.ceil(len(val_df) / BATCH_SIZE)
)

print("\nDataset ready.")

print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)

# ============================================================
# Build MobileNetV2 model
# ============================================================

print("\nBuilding MobileNetV2...")

input_layer = layers.Input(
    shape=(IMG_H, IMG_W, 1)
)

# Convert grayscale -> 3 channels
x = layers.Concatenate(
    axis=-1
)([
    input_layer,
    input_layer,
    input_layer
])

# Mel values [-80, 0] -> [-1, 1]
x = layers.Rescaling(
    scale=1 / 40.0,
    offset=1
)(x)

base_model = MobileNetV2(
    input_shape=(IMG_H, IMG_W, 3),
    include_top=False,
    weights="imagenet"
)

# Stage 1: freeze backbone
base_model.trainable = False

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

output_layer = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs=input_layer,
    outputs=output_layer
)

model.summary()

# ============================================================
# Stage 1
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 - FROZEN MOBILENETV2")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

stage1_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_exp5_stage1_best.keras"
)

callbacks_stage1 = [

    ModelCheckpoint(
        stage1_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_dataset,
    epochs=STAGE1_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks_stage1
)

# ============================================================
# Stage 2 - Fine tuning
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 - FINE TUNING LAST 30 LAYERS")
print("=" * 70)

base_model.trainable = True

# Freeze everything except last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNorm frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print(
    "Trainable MobileNetV2 layers:",
    trainable_count
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

stage2_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_exp5_finetuned_best.keras"
)

callbacks_stage2 = [

    ModelCheckpoint(
        stage2_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_dataset,
    epochs=STAGE2_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks_stage2
)

# ============================================================
# Final test evaluation
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

# Load best fine-tuned model
best_model = tf.keras.models.load_model(
    stage2_path
)

# Build test dataset
def test_generator():

    for _, row in test_df.iterrows():

        mel = load_mel(row["mel_path"])

        mel = np.expand_dims(
            mel,
            axis=-1
        )

        label = int(row["label"])

        yield mel, label


test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_signature=output_signature
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

test_steps = int(
    np.ceil(len(test_df) / BATCH_SIZE)
)

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    steps=test_steps,
    verbose=1
)

print("\nTest Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)

# ============================================================
# Save final model
# ============================================================

final_path = os.path.join(
    MODEL_DIR,
    "bird_mobilenetv2_exp5_final.keras"
)

best_model.save(final_path)

print("\n")
print("=" * 70)
print("EXPERIMENT 5 COMPLETE")
print("=" * 70)

print("\nSaved models:")

print(stage1_path)
print(stage2_path)
print(final_path)

print("\nBest model for evaluation:")
print(stage2_path)