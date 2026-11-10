# Data

Audio files and mel-spectrograms are **not** included in this repository because of their size. This page describes the data and how to rebuild it.

## Source

- **Dataset:** [BirdCLEF 2024](https://www.kaggle.com/competitions/birdclef-2024) (Kaggle), bird recordings from the Western Ghats of India, originally from Xeno-canto
- **Original size:** 24,459 recordings, 182 species
- **Used here:** 20 species
- **License:** each recording has its own Creative Commons license, given in the competition metadata. Check the competition rules and the licenses before redistributing any audio.

## Species (20)

Barn Swallow, Black-crowned Night-Heron, Black-winged Stilt, Blyth's Reed Warbler, Common Greenshank, Common Kingfisher, Common Rosefinch, Common Sandpiper, Eurasian Coot, Eurasian Moorhen, Gray Heron, Green Sandpiper, Greenish Warbler, Little Egret, Little Grebe, Little Ringed Plover, Rose-ringed Parakeet, Western Yellow Wagtail, Wood Sandpiper, Zitting Cisticola

## Processing

| Step | Setting |
|---|---|
| Split | 70 / 15 / 15 (train / validation / test), done **per recording before chunking** so that chunks of one recording never appear in two splits |
| Chunk length | 5 s, with 2.5 s overlap |
| Sample rate | 22,050 Hz |
| Mel-spectrogram | 128 x 216, N_FFT 1024, hop length 512, dB scale in [-80, 0], saved as `.npy` |

## Expected layout

```
data/
└── bird_mels/
    ├── mel_metadata.csv
    └── <split>/<species>/<chunk>.npy
```

`mel_metadata.csv` columns used by the training script:

| Column | Meaning |
|---|---|
| `mel_filename` | name of the `.npy` file |
| `audio_filename` | source chunk name (`<recording>_chunk_<n>`) |
| `species` | species name |
| `split` | `train`, `validation` or `test` |

## Rebuild

1. Download BirdCLEF 2024 from Kaggle.
2. Run the scripts in `scripts/01_dataset/` (species selection and split).
3. Run `scripts/02_preprocessing/` in this order: `chunk_audio.py`, `generate_mels.py`, `verify_mels.py`.
4. If your mels are stored elsewhere, set the `BIRD_MEL_DIR` environment variable to that folder.
