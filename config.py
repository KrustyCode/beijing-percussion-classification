"""Project-wide settings for the percussion classification project.

Every path is resolved relative to this file, so scripts and notebooks work
no matter which directory they are started from.
"""
from pathlib import Path
from typing import List

N_FFT: int = 2_048
WINDOW_LENGTH: int = 2_048
HOP_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Audio and dataset
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 44_100
N_CHANNEL: int = 1
SEED: int = 42

TRAIN_SIZE: float = 0.8
TEST_SIZE: float = 0.2

LABEL: List[str] = ["daluo", "bangu", "xiaoluo", "naobo"]

# ---------------------------------------------------------------------------
# Augmentation settings
# ---------------------------------------------------------------------------
# Number of noise / MP3 copies made from every training clip. Each copy uses
# a different value drawn from the TRAIN_* range below.
N_AUG_COPIES: int = 3

# Target sample rates for down-sampling (one copy per rate, train and test).
DOWN_SR: List[int] = [
    22_050,
    16_000,
    8_000
]

# LAME VBR quality, 0 = best / highest bitrate, 9 = worst / lowest bitrate.
# TRAIN_COMPRESSION is an inclusive [low, high] range; TEST_COMPRESSION are
# the exact levels used for the test set.
TRAIN_COMPRESSION: List[int] = [2, 9]
TEST_COMPRESSION: List[int] = [2, 5, 9]

# Signal-to-noise ratio in dB, lower = noisier. TRAIN_SNR is an inclusive
# [low, high] range; TEST_SNR are the exact levels used for the test set.
TRAIN_SNR: List[int] = [20, 40]
TEST_SNR: List[int] = [30, 20, 10, 5, 0]

# ---------------------------------------------------------------------------
# Model hyperparameter grid (searched with Grid Search)
# ---------------------------------------------------------------------------
# SVM: every combination of KERNEL, C and GAMMA is tried, except that the
# "linear" kernel is not combined with GAMMA, because the linear kernel
# function does not use gamma.
KERNEL: List[str] = ["linear", "poly", "rbf"]
C: List[float] = [0.1, 1, 10, 100]
GAMMA: List[str | float] = ["scale", 0.001, 0.01, 0.1]

# KNN: number of neighbours that vote for the prediction.
K: List[int] = [1, 3, 5, 7, 9, 11]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Original (untouched) dataset.
FOLDER_PATH: Path = PROJECT_ROOT / "dataset"

# Saved train / test split (clip name, label, split). Created on the first
# run and reused afterwards, so the split never changes unless asked to.
SPLIT_PATH: Path = PROJECT_ROOT / "split.csv"

# Augmented audio lives outside the original dataset, one folder per
# augmentation. Each folder has a `train/` and a `test/` subfolder.
AUGMENTED_FOLDER_PATH: Path = PROJECT_ROOT / "augmented_dataset"
NOISE_FOLDER_PATH: Path = AUGMENTED_FOLDER_PATH / "noise"
MP3_FOLDER_PATH: Path = AUGMENTED_FOLDER_PATH / "mp3"
DOWNSAMPLE_FOLDER_PATH: Path = AUGMENTED_FOLDER_PATH / "downsample"

# CSV listing every original and augmented file with its label and split.
# Paths in it are relative to PROJECT_ROOT.
MANIFEST_PATH: Path = AUGMENTED_FOLDER_PATH / "manifest.csv"

# Mean / std of the training features used for z-score standardization.
SCALER_PATH: Path = PROJECT_ROOT / "scaler_params.json"

# Trained models saved with `model.save_model` (one .pkl file per model).
MODEL_FOLDER_PATH: Path = PROJECT_ROOT / "models"
