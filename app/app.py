# import os
# import tempfile

# import numpy as np
# import librosa
# import tensorflow as tf
# import streamlit as st
# import matplotlib.pyplot as plt


# # ============================================================
# # CONFIGURATION
# # ============================================================

# MODEL_PATH = r"D:\project1\models\bird_mobilenetv2_exp5_final.keras"

# SAMPLE_RATE = 22050

# CHUNK_SECONDS = 5.0
# OVERLAP_SECONDS = 2.5

# N_MELS = 128
# N_FFT = 1024
# HOP_LENGTH = 512

# IMG_H = 128
# IMG_W = 216

# BATCH_SIZE = 32


# # ============================================================
# # CLASS NAMES
# # ============================================================

# CLASSES = [
#     "Barn Swallow",
#     "Black-crowned Night-Heron",
#     "Black-winged Stilt",
#     "Blyth's Reed Warbler",
#     "Common Greenshank",
#     "Common Kingfisher",
#     "Common Rosefinch",
#     "Common Sandpiper",
#     "Eurasian Coot",
#     "Eurasian Moorhen",
#     "Gray Heron",
#     "Green Sandpiper",
#     "Greenish Warbler",
#     "Little Egret",
#     "Little Grebe",
#     "Little Ringed Plover",
#     "Rose-ringed Parakeet",
#     "Western Yellow Wagtail",
#     "Wood Sandpiper",
#     "Zitting Cisticola"
# ]


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Svara | AI Bird Sound Identification",
#     page_icon="🐦",
#     layout="centered"
# )


# # ============================================================
# # CUSTOM CSS
# # ============================================================

# st.markdown(
#     """
#     <style>

#     .title {
#         text-align: center;
#         font-size: 52px;
#         font-weight: 700;
#         margin-top: 20px;
#         margin-bottom: 0px;
#     }

#     .subtitle {
#         text-align: center;
#         font-size: 19px;
#         color: #666666;
#         margin-top: 4px;
#         margin-bottom: 45px;
#     }

#     .section-title {
#         font-size: 25px;
#         font-weight: 650;
#         margin-top: 20px;
#         margin-bottom: 12px;
#     }

#     .prediction-card {
#         padding: 25px;
#         border-radius: 15px;
#         border: 1px solid #dddddd;
#         margin-top: 15px;
#         text-align: center;
#     }

#     .prediction-species {
#         font-size: 31px;
#         font-weight: 700;
#         margin-top: 10px;
#     }

#     .prediction-confidence {
#         font-size: 19px;
#         color: #555555;
#         margin-top: 7px;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # ============================================================
# # LOAD MODEL
# # ============================================================

# @st.cache_resource
# def load_model():

#     loaded_model = tf.keras.models.load_model(
#         MODEL_PATH
#     )

#     return loaded_model


# try:

#     model = load_model()

# except Exception as e:

#     st.error(
#         f"Unable to load model: {e}"
#     )

#     st.stop()


# # ============================================================
# # AUDIO → MEL-SPECTROGRAM
# # ============================================================

# def audio_to_mel(audio):

#     mel = librosa.feature.melspectrogram(
#         y=audio,
#         sr=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     mel_db = librosa.power_to_db(
#         mel,
#         ref=np.max
#     )

#     return mel_db.astype(
#         np.float32
#     )


# # ============================================================
# # CREATE 5-SECOND AUDIO CHUNKS
# # ============================================================

# def create_chunks(audio):

#     chunk_size = int(
#         CHUNK_SECONDS * SAMPLE_RATE
#     )

#     hop_size = int(
#         (CHUNK_SECONDS - OVERLAP_SECONDS)
#         * SAMPLE_RATE
#     )

#     chunks = []

#     start = 0

#     while start < len(audio):

#         end = start + chunk_size

#         chunk = audio[start:end]

#         # Pad final chunk
#         if len(chunk) < chunk_size:

#             chunk = np.pad(
#                 chunk,
#                 (
#                     0,
#                     chunk_size - len(chunk)
#                 ),
#                 mode="constant"
#             )

#         chunks.append(chunk)

#         if end >= len(audio):

#             break

#         start += hop_size

#     return chunks


# # ============================================================
# # PREDICTION
# # ============================================================

# def predict_bird(audio):

#     chunks = create_chunks(
#         audio
#     )

#     mel_images = []
#     valid_mels = []

#     for chunk in chunks:

#         mel = audio_to_mel(
#             chunk
#         )

#         if mel.shape != (
#             IMG_H,
#             IMG_W
#         ):

#             continue

#         valid_mels.append(
#             mel.copy()
#         )

#         mel_input = np.expand_dims(
#             mel,
#             axis=-1
#         )

#         mel_images.append(
#             mel_input
#         )

#     if len(mel_images) == 0:

#         raise ValueError(
#             "No valid Mel-spectrograms "
#             "were generated."
#         )

#     X = np.array(
#         mel_images,
#         dtype=np.float32
#     )

#     # --------------------------------------------------------
#     # Predict all chunks
#     # --------------------------------------------------------

#     predictions = model.predict(
#         X,
#         batch_size=BATCH_SIZE,
#         verbose=0
#     )

#     # --------------------------------------------------------
#     # Average chunk probabilities
#     # --------------------------------------------------------

#     final_probabilities = np.mean(
#         predictions,
#         axis=0
#     )

#     predicted_index = int(
#         np.argmax(
#             final_probabilities
#         )
#     )

#     predicted_species = CLASSES[
#         predicted_index
#     ]

#     confidence = float(
#         final_probabilities[
#             predicted_index
#         ] * 100
#     )

#     # --------------------------------------------------------
#     # Select strongest chunk
#     # --------------------------------------------------------

#     chunk_scores = predictions[
#         :,
#         predicted_index
#     ]

#     best_chunk_index = int(
#         np.argmax(
#             chunk_scores
#         )
#     )

#     best_mel = valid_mels[
#         best_chunk_index
#     ]

#     # --------------------------------------------------------
#     # Top 5 predictions
#     # --------------------------------------------------------

#     top_indices = np.argsort(
#         final_probabilities
#     )[::-1][:5]

#     top_predictions = []

#     for index in top_indices:

#         top_predictions.append(
#             {
#                 "species": CLASSES[index],
#                 "confidence": float(
#                     final_probabilities[index]
#                     * 100
#                 )
#             }
#         )

#     return (
#         predicted_species,
#         confidence,
#         top_predictions,
#         len(chunks),
#         best_mel,
#         predicted_index,
#         best_chunk_index
#     )


# # ============================================================
# # FIND MOBILENETV2 BASE MODEL
# # ============================================================

# def get_mobilenet_base_model():

#     for layer in model.layers:

#         if isinstance(
#             layer,
#             tf.keras.Model
#         ):

#             if "mobilenet" in layer.name.lower():

#                 return layer

#     raise ValueError(
#         "MobileNetV2 base model was not found "
#         "inside the saved model."
#     )


# # ============================================================
# # FIND LAST CONVOLUTIONAL LAYER
# # ============================================================

# def get_last_conv_layer(
#     base_model
# ):

#     conv_layers = []

#     for layer in base_model.layers:

#         if isinstance(
#             layer,
#             (
#                 tf.keras.layers.Conv2D,
#                 tf.keras.layers.DepthwiseConv2D
#             )
#         ):

#             conv_layers.append(
#                 layer
#             )

#     if not conv_layers:

#         raise ValueError(
#             "No convolutional layer found "
#             "inside MobileNetV2."
#         )

#     return conv_layers[-1]


# # ============================================================
# # PREPARE MOBILENETV2 INPUT
# # ============================================================

# def prepare_mobilenet_input(
#     mel_input
# ):

#     """
#     Reproduce Exp5 preprocessing.

#     Mel values:
#         approximately [-80, 0]

#     Rescaling:
#         x / 40 + 1

#     Result:
#         approximately [-1, 1]

#     Then:
#         1 channel → 3 channels
#     """

#     x = tf.convert_to_tensor(
#         mel_input,
#         dtype=tf.float32
#     )

#     # Same Rescaling used during Exp5
#     x = x / 40.0 + 1.0

#     # Convert grayscale to 3 channels
#     x = tf.concat(
#         [x, x, x],
#         axis=-1
#     )

#     return x


# # ============================================================
# # GET CLASSIFICATION HEAD
# # ============================================================

# def get_head_layers(
#     base_model
# ):

#     base_index = None

#     for i, layer in enumerate(
#         model.layers
#     ):

#         if layer is base_model:

#             base_index = i

#             break

#     if base_index is None:

#         raise ValueError(
#             "Could not locate MobileNetV2 "
#             "inside the outer model."
#         )

#     return model.layers[
#         base_index + 1:
#     ]


# # ============================================================
# # GRAD-CAM
# # ============================================================

# def generate_gradcam(
#     mel_input,
#     class_index
# ):

#     # --------------------------------------------------------
#     # Get MobileNetV2
#     # --------------------------------------------------------

#     base_model = (
#         get_mobilenet_base_model()
#     )

#     # --------------------------------------------------------
#     # Get final convolutional layer
#     # --------------------------------------------------------

#     target_layer = (
#         get_last_conv_layer(
#             base_model
#         )
#     )

#     # --------------------------------------------------------
#     # Feature model
#     #
#     # Outputs:
#     #   1. Target convolutional feature maps
#     #   2. MobileNetV2 output
#     # --------------------------------------------------------

#     feature_model = tf.keras.models.Model(
#         inputs=base_model.input,
#         outputs=[
#             target_layer.output,
#             base_model.output
#         ]
#     )

#     # --------------------------------------------------------
#     # Prepare input
#     # --------------------------------------------------------

#     x = prepare_mobilenet_input(
#         mel_input
#     )

#     # Add batch dimension
#     x = tf.expand_dims(
#         x,
#         axis=0
#     )

#     # --------------------------------------------------------
#     # Forward pass
#     # --------------------------------------------------------

#     with tf.GradientTape() as tape:

#         conv_outputs, base_output = (
#             feature_model(
#                 x,
#                 training=False
#             )
#         )

#         # ----------------------------------------------------
#         # Apply classification head
#         # ----------------------------------------------------

#         predictions = base_output

#         head_layers = (
#             get_head_layers(
#                 base_model
#             )
#         )

#         for layer in head_layers:

#             predictions = layer(
#                 predictions,
#                 training=False
#             )

#         # Score of selected class
#         class_score = predictions[
#             :,
#             class_index
#         ]

#     # --------------------------------------------------------
#     # Gradient
#     # --------------------------------------------------------

#     gradients = tape.gradient(
#         class_score,
#         conv_outputs
#     )

#     if gradients is None:

#         raise ValueError(
#             "Gradients are None. "
#             "Grad-CAM could not be calculated."
#         )

#     # --------------------------------------------------------
#     # Global average pooling
#     # --------------------------------------------------------

#     pooled_gradients = tf.reduce_mean(
#         gradients,
#         axis=(1, 2)
#     )

#     # Remove batch dimension
#     conv_outputs = conv_outputs[0]

#     pooled_gradients = (
#         pooled_gradients[0]
#     )

#     # --------------------------------------------------------
#     # Weighted feature maps
#     # --------------------------------------------------------

#     heatmap = tf.reduce_sum(
#         conv_outputs
#         * pooled_gradients,
#         axis=-1
#     )

#     # --------------------------------------------------------
#     # ReLU
#     # --------------------------------------------------------

#     heatmap = tf.maximum(
#         heatmap,
#         0
#     )

#     # --------------------------------------------------------
#     # Normalize to 0–1
#     # --------------------------------------------------------

#     max_value = tf.reduce_max(
#         heatmap
#     )

#     heatmap = heatmap / (
#         max_value
#         + tf.keras.backend.epsilon()
#     )

#     return heatmap.numpy()


# # ============================================================
# # CREATE TWO-PANEL GRAD-CAM FIGURE
# # ============================================================

# def create_gradcam_figure(
#     mel,
#     heatmap,
#     predicted_species,
#     confidence
# ):

#     # ========================================================
#     # NORMALIZE MEL-SPECTROGRAM
#     # ========================================================

#     mel_display = mel.copy()

#     mel_min = mel_display.min()
#     mel_max = mel_display.max()

#     if mel_max > mel_min:

#         mel_display = (
#             (mel_display - mel_min)
#             / (mel_max - mel_min)
#         )

#     # ========================================================
#     # RESIZE GRAD-CAM HEATMAP
#     # ========================================================

#     heatmap_resized = (
#         tf.image.resize(
#             heatmap[..., np.newaxis],
#             [
#                 mel.shape[0],
#                 mel.shape[1]
#             ]
#         )
#         .numpy()
#         .squeeze()
#     )

#     # ========================================================
#     # CREATE TWO PANELS
#     # ========================================================

#     fig, axes = plt.subplots(
#         1,
#         2,
#         figsize=(14, 5)
#     )

#     # ========================================================
#     # LEFT PANEL — ORIGINAL MEL
#     # ========================================================

#     axes[0].imshow(
#         mel_display,
#         aspect="auto",
#         origin="lower"
#     )

#     axes[0].set_title(
#         "Original Mel-Spectrogram",
#         fontsize=13,
#         fontweight="bold"
#     )

#     axes[0].set_xlabel(
#         "Time"
#     )

#     axes[0].set_ylabel(
#         "Mel Frequency"
#     )

#     # ========================================================
#     # RIGHT PANEL — GRAD-CAM
#     # ========================================================

#     axes[1].imshow(
#         mel_display,
#         aspect="auto",
#         origin="lower"
#     )

#     gradcam_image = axes[1].imshow(
#         heatmap_resized,
#         cmap="jet",
#         alpha=0.50,
#         aspect="auto",
#         origin="lower",
#         vmin=0,
#         vmax=1
#     )

#     axes[1].set_title(
#         f"Grad-CAM — {predicted_species}\n"
#         f"Confidence: {confidence:.2f}%",
#         fontsize=13,
#         fontweight="bold"
#     )

#     axes[1].set_xlabel(
#         "Time"
#     )

#     axes[1].set_ylabel(
#         "Mel Frequency"
#     )

#     # ========================================================
#     # COLORBAR
#     # ========================================================

#     colorbar = fig.colorbar(
#         gradcam_image,
#         ax=axes[1],
#         fraction=0.046,
#         pad=0.04
#     )

#     colorbar.set_label(
#         "Grad-CAM Importance",
#         rotation=270,
#         labelpad=15
#     )

#     # ========================================================
#     # MAIN TITLE
#     # ========================================================

#     fig.suptitle(
#         "Explainable AI — "
#         "Acoustic Regions Influencing Prediction",
#         fontsize=15,
#         fontweight="bold"
#     )

#     plt.tight_layout(
#         rect=[0, 0, 1, 0.94]
#     )

#     return fig


# # ============================================================
# # HEADER
# # ============================================================

# st.markdown(
#     '<div class="title">Svara</div>',
#     unsafe_allow_html=True
# )

# st.markdown(
#     '<div class="subtitle">'
#     'AI-Based Bird Sound Identification'
#     '</div>',
#     unsafe_allow_html=True
# )


# # ============================================================
# # UPLOAD BIRD AUDIO
# # ============================================================

# st.markdown(
#     '<div class="section-title">'
#     '🎵 Upload Bird Audio'
#     '</div>',
#     unsafe_allow_html=True
# )

# uploaded_file = st.file_uploader(
#     "Choose an audio file",
#     type=[
#         "wav",
#         "mp3",
#         "ogg",
#         "flac",
#         "m4a"
#     ],
#     label_visibility="collapsed"
# )


# # ============================================================
# # AUDIO PROCESSING
# # ============================================================

# if uploaded_file is not None:

#     # --------------------------------------------------------
#     # Audio player
#     # --------------------------------------------------------

#     st.audio(
#         uploaded_file,
#         format=uploaded_file.type
#     )

#     st.markdown("---")

#     # --------------------------------------------------------
#     # Analyze button
#     # --------------------------------------------------------

#     analyze = st.button(
#         "🔍 Analyze Audio",
#         use_container_width=True
#     )

#     if analyze:

#         temp_path = None

#         try:

#             # ====================================================
#             # ANALYSIS
#             # ====================================================

#             with st.spinner(
#                 "Analyzing bird sound..."
#             ):

#                 # ------------------------------------------------
#                 # Save temporary file
#                 # ------------------------------------------------

#                 suffix = os.path.splitext(
#                     uploaded_file.name
#                 )[1]

#                 with tempfile.NamedTemporaryFile(
#                     delete=False,
#                     suffix=suffix
#                 ) as temp_file:

#                     temp_file.write(
#                         uploaded_file.getbuffer()
#                     )

#                     temp_path = (
#                         temp_file.name
#                     )

#                 # ------------------------------------------------
#                 # Load audio
#                 # ------------------------------------------------

#                 audio, sr = librosa.load(
#                     temp_path,
#                     sr=SAMPLE_RATE,
#                     mono=True
#                 )

#                 # ------------------------------------------------
#                 # Run prediction
#                 # ------------------------------------------------

#                 (
#                     species,
#                     confidence,
#                     top_predictions,
#                     num_chunks,
#                     best_mel,
#                     predicted_index,
#                     best_chunk_index
#                 ) = predict_bird(
#                     audio
#                 )

#             # ----------------------------------------------------
#             # Delete temporary file
#             # ----------------------------------------------------

#             if os.path.exists(
#                 temp_path
#             ):

#                 os.remove(
#                     temp_path
#                 )

#             temp_path = None

#             # ====================================================
#             # PREDICTION RESULT
#             # ====================================================

#             st.markdown(
#                 '<div class="section-title">'
#                 '🐦 Prediction'
#                 '</div>',
#                 unsafe_allow_html=True
#             )

#             st.markdown(
#                 '<div class="prediction-card">',
#                 unsafe_allow_html=True
#             )

#             st.markdown(
#                 f'<div class="prediction-species">'
#                 f'{species}'
#                 f'</div>',
#                 unsafe_allow_html=True
#             )

#             st.markdown(
#                 f'<div class="prediction-confidence">'
#                 f'Confidence: {confidence:.2f}%'
#                 f'</div>',
#                 unsafe_allow_html=True
#             )

#             st.markdown(
#                 '</div>',
#                 unsafe_allow_html=True
#             )

#             # ====================================================
#             # TOP 5 PREDICTIONS
#             # ====================================================

#             st.markdown(
#                 '<div class="section-title">'
#                 'Top 5 Predictions'
#                 '</div>',
#                 unsafe_allow_html=True
#             )

#             for i, item in enumerate(
#                 top_predictions,
#                 start=1
#             ):

#                 col1, col2 = st.columns(
#                     [4, 1]
#                 )

#                 with col1:

#                     st.write(
#                         f"**{i}. "
#                         f"{item['species']}**"
#                     )

#                 with col2:

#                     st.write(
#                         f"{item['confidence']:.2f}%"
#                     )

#             st.info(
#                 f"Analyzed using "
#                 f"{num_chunks} audio chunks."
#             )

#             # ====================================================
#             # GRAD-CAM
#             # ====================================================

#             st.markdown("---")

#             st.markdown(
#                 '<div class="section-title">'
#                 '🔥 Grad-CAM Explanation'
#                 '</div>',
#                 unsafe_allow_html=True
#             )

#             with st.spinner(
#                 "Generating Grad-CAM..."
#             ):

#                 # ------------------------------------------------
#                 # Prepare representative Mel
#                 # ------------------------------------------------

#                 gradcam_input = (
#                     np.expand_dims(
#                         best_mel,
#                         axis=-1
#                     )
#                 )

#                 # ------------------------------------------------
#                 # Generate Grad-CAM
#                 # ------------------------------------------------

#                 heatmap = (
#                     generate_gradcam(
#                         gradcam_input,
#                         predicted_index
#                     )
#                 )

#                 # ------------------------------------------------
#                 # Create two-panel figure
#                 # ------------------------------------------------

#                 fig = (
#                     create_gradcam_figure(
#                         best_mel,
#                         heatmap,
#                         species,
#                         confidence
#                     )
#                 )

#                 st.pyplot(
#                     fig,
#                     use_container_width=True
#                 )

#                 plt.close(fig)

#             st.caption(
#                 f"Grad-CAM generated from "
#                 f"representative chunk "
#                 f"{best_chunk_index + 1} "
#                 f"of {num_chunks}."
#             )

#             st.info(
#                 "Grad-CAM highlights the "
#                 "time-frequency regions that "
#                 "contributed most strongly to "
#                 "the predicted class."
#             )

#         except Exception as e:

#             # ----------------------------------------------------
#             # Cleanup
#             # ----------------------------------------------------

#             if (
#                 temp_path is not None
#                 and os.path.exists(
#                     temp_path
#                 )
#             ):

#                 os.remove(
#                     temp_path
#                 )

#             # ----------------------------------------------------
#             # Error
#             # ----------------------------------------------------

#             st.error(
#                 "An error occurred during analysis."
#             )

#             st.exception(e)
import os
import tempfile

import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\project1\models\bird_mobilenetv2_exp5_final.keras"

SAMPLE_RATE = 22050

CHUNK_SECONDS = 5.0
OVERLAP_SECONDS = 2.5

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

IMG_H = 128
IMG_W = 216

BATCH_SIZE = 32


# ============================================================
# CLASS NAMES
# ============================================================

CLASSES = [
    "Barn Swallow",
    "Black-crowned Night-Heron",
    "Black-winged Stilt",
    "Blyth's Reed Warbler",
    "Common Greenshank",
    "Common Kingfisher",
    "Common Rosefinch",
    "Common Sandpiper",
    "Eurasian Coot",
    "Eurasian Moorhen",
    "Gray Heron",
    "Green Sandpiper",
    "Greenish Warbler",
    "Little Egret",
    "Little Grebe",
    "Little Ringed Plover",
    "Rose-ringed Parakeet",
    "Western Yellow Wagtail",
    "Wood Sandpiper",
    "Zitting Cisticola"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BirdVision | AI Bird Sound Identification",
    page_icon="🐦",
    layout="centered"
)


# ============================================================
# CUSTOM CSS (Modern Nature-Inspired Theme)
# ============================================================

st.markdown(
    """
    <style>
    /* Global Styles */
    .main {
        background-color: #fafdfa;
    }

    .title-container {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }

    .title {
        font-size: 48px;
        font-weight: 800;
        color: #1b4332;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #52796f;
        margin-top: 4px;
        margin-bottom: 30px;
        font-weight: 400;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #2d6a4f;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 5px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #f0f7f4 0%, #e8f5e9 100%);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-top: 15px;
        text-align: center;
    }

    .prediction-species {
        font-size: 32px;
        font-weight: 800;
        color: #1b4332;
        margin-top: 5px;
    }

    .prediction-confidence {
        font-size: 18px;
        color: #40916c;
        font-weight: 600;
        margin-top: 8px;
    }

    /* Styler overrides for cleaner widgets */
    .stButton>button {
        background-color: #2d6a4f;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b4332;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    return loaded_model


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ============================================================
# AUDIO → MEL-SPECTROGRAM
# ============================================================

def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


# ============================================================
# CREATE 5-SECOND AUDIO CHUNKS
# ============================================================

def create_chunks(audio):
    chunk_size = int(CHUNK_SECONDS * SAMPLE_RATE)
    hop_size = int((CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE)

    chunks = []
    start = 0

    while start < len(audio):
        end = start + chunk_size
        chunk = audio[start:end]

        # Pad final chunk
        if len(chunk) < chunk_size:
            chunk = np.pad(
                chunk,
                (0, chunk_size - len(chunk)),
                mode="constant"
            )

        chunks.append(chunk)

        if end >= len(audio):
            break

        start += hop_size

    return chunks


# ============================================================
# PREDICTION
# ============================================================

def predict_bird(audio):
    chunks = create_chunks(audio)

    mel_images = []
    valid_mels = []

    for chunk in chunks:
        mel = audio_to_mel(chunk)
        if mel.shape != (IMG_H, IMG_W):
            continue

        valid_mels.append(mel.copy())
        mel_input = np.expand_dims(mel, axis=-1)
        mel_images.append(mel_input)

    if len(mel_images) == 0:
        raise ValueError("No valid Mel-spectrograms were generated.")

    X = np.array(mel_images, dtype=np.float32)

    predictions = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    final_probabilities = np.mean(predictions, axis=0)

    predicted_index = int(np.argmax(final_probabilities))
    predicted_species = CLASSES[predicted_index]
    confidence = float(final_probabilities[predicted_index] * 100)

    chunk_scores = predictions[:, predicted_index]
    best_chunk_index = int(np.argmax(chunk_scores))
    best_mel = valid_mels[best_chunk_index]

    top_indices = np.argsort(final_probabilities)[::-1][:5]
    top_predictions = []

    for index in top_indices:
        top_predictions.append(
            {
                "species": CLASSES[index],
                "confidence": float(final_probabilities[index] * 100)
            }
        )

    return (
        predicted_species,
        confidence,
        top_predictions,
        len(chunks),
        best_mel,
        predicted_index,
        best_chunk_index
    )


# ============================================================
# MOBILENETV2 & GRAD-CAM UTILITIES
# ============================================================

def get_mobilenet_base_model():
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "mobilenet" in layer.name.lower():
                return layer
    raise ValueError("MobileNetV2 base model was not found inside the saved model.")

def get_last_conv_layer(base_model):
    conv_layers = []
    for layer in base_model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            conv_layers.append(layer)
    if not conv_layers:
        raise ValueError("No convolutional layer found inside MobileNetV2.")
    return conv_layers[-1]

def prepare_mobilenet_input(mel_input):
    x = tf.convert_to_tensor(mel_input, dtype=tf.float32)
    x = x / 40.0 + 1.0
    x = tf.concat([x, x, x], axis=-1)
    return x

def get_head_layers(base_model):
    base_index = None
    for i, layer in enumerate(model.layers):
        if layer is base_model:
            base_index = i
            break
    if base_index is None:
        raise ValueError("Could not locate MobileNetV2 inside the outer model.")
    return model.layers[base_index + 1:]

def generate_gradcam(mel_input, class_index):
    base_model = get_mobilenet_base_model()
    target_layer = get_last_conv_layer(base_model)

    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output]
    )

    x = prepare_mobilenet_input(mel_input)
    x = tf.expand_dims(x, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = feature_model(x, training=False)
        predictions = base_output
        head_layers = get_head_layers(base_model)

        for layer in head_layers:
            predictions = layer(predictions, training=False)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise ValueError("Gradients are None. Grad-CAM could not be calculated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())

    return heatmap.numpy()


def create_gradcam_figure(mel, heatmap, predicted_species, confidence):
    mel_display = mel.copy()
    mel_min = mel_display.min()
    mel_max = mel_display.max()

    if mel_max > mel_min:
        mel_display = (mel_display - mel_min) / (mel_max - mel_min)

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        [mel.shape[0], mel.shape[1]]
    ).numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left Panel
    axes[0].imshow(mel_display, aspect="auto", origin="lower", cmap="inferno")
    axes[0].set_title("Original Mel-Spectrogram", fontsize=13, fontweight="bold", color="#1b4332")
    axes[0].set_xlabel("Time Frames")
    axes[0].set_ylabel("Mel Frequency Bins")

    # Right Panel
    axes[1].imshow(mel_display, aspect="auto", origin="lower", cmap="inferno")
    gradcam_image = axes[1].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.55,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        f"Grad-CAM — {predicted_species}\nConfidence: {confidence:.2f}%",
        fontsize=13,
        fontweight="bold",
        color="#1b4332"
    )
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Frequency Bins")

    colorbar = fig.colorbar(gradcam_image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Importance Score", rotation=270, labelpad=15)

    fig.suptitle(
        "Explainable AI — Acoustic Regions Influencing Prediction",
        fontsize=15,
        fontweight="bold",
        color="#1b4332"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=11306&format=png&color=2D6A4F", width=60)
    st.markdown("### About BirdVision")
    st.markdown(
        "BirdVision is an advanced deep-learning audio classification tool "
        "trained to recognize common bird species by their calls and songs using "
        "Mel-spectrogram analysis and MobileNetV2."
    )
    st.markdown("---")
    st.markdown("**Supported Formats:**")
    st.caption("WAV, MP3, OGG, FLAC, M4A")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title-container">
        <div class="title">🐦 BirdVision</div>
        <div class="subtitle">AI-Powered Bird Sound Identification & Explainable Audio Analysis</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD BIRD AUDIO
# ============================================================

st.markdown('<div class="section-title">🎵 Upload Audio Recording</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    label_visibility="collapsed"
)


# ============================================================
# AUDIO PROCESSING & PREDICTION
# ============================================================

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)
    st.markdown("---")

    analyze = st.button("🔍 Analyze Audio", use_container_width=True)

    if analyze:
        temp_path = None
        try:
            with st.spinner("Analyzing bird vocalizations..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)

                (
                    species,
                    confidence,
                    top_predictions,
                    num_chunks,
                    best_mel,
                    predicted_index,
                    best_chunk_index
                ) = predict_bird(audio)

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                temp_path = None

            # --------------------------------------------------------
            # RESULT CARD
            # --------------------------------------------------------
            st.markdown('<div class="section-title">🏆 Top Prediction</div>', unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #52796f; font-weight: 700;">Identified Species</div>
                    <div class="prediction-species">{species}</div>
                    <div class="prediction-confidence">Confidence Score: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --------------------------------------------------------
            # TOP 5 PREDICTIONS
            # --------------------------------------------------------
            st.markdown('<div class="section-title">📊 Top 5 Probabilities</div>', unsafe_allow_html=True)

            for i, item in enumerate(top_predictions, start=1):
                col1, col2, col3 = st.columns([3, 5, 1])
                with col1:
                    st.write(f"**{i}. {item['species']}**")
                with col2:
                    st.progress(int(item['confidence']))
                with col3:
                    st.write(f"**{item['confidence']:.1f}%**")

            st.info(f"💡 Audio processed successfully across **{num_chunks}** individual time chunks.")

            # --------------------------------------------------------
            # GRAD-CAM EXPLANATION
            # --------------------------------------------------------
            st.markdown("---")
            st.markdown('<div class="section-title">🔥 Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

            with st.spinner("Generating Grad-CAM visualization..."):
                gradcam_input = np.expand_dims(best_mel, axis=-1)
                heatmap = generate_gradcam(gradcam_input, predicted_index)
                fig = create_gradcam_figure(best_mel, heatmap, species, confidence)

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.caption(f"Grad-CAM activation highlights generated from representative chunk #{best_chunk_index + 1} of {num_chunks}.")

        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            st.error("An error occurred during audio processing or model inference.")
            st.exception(e)