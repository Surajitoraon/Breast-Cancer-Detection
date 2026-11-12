"""
config.py
----------
Central configuration for the Breast Cancer Detection pipeline.
Edit values here rather than hardcoding them across scripts.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
CLASS_NAMES = ["benign", "malignant"]  # index 0 = benign, index 1 = malignant

# ---------------------------------------------------------------------------
# Image / preprocessing settings
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)      # (height, width) fed to the CNN
CHANNELS = 3                # 3 for RGB (also works if source images are grayscale)
APPLY_CLAHE = True          # contrast-limited adaptive histogram equalization

# ---------------------------------------------------------------------------
# Data augmentation settings (applied only to training data)
# ---------------------------------------------------------------------------
AUGMENTATION = dict(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.85, 1.15),
    fill_mode="nearest",
)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15      # used only if a separate val/ folder isn't provided
RANDOM_SEED = 42

# Handle class imbalance (malignant cases are usually the minority class)
USE_CLASS_WEIGHTS = True

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# If True, uses a pretrained EfficientNetB0 backbone (recommended for small
# datasets). If False, uses the custom CNN defined from scratch.
USE_TRANSFER_LEARNING = True
BACKBONE = "EfficientNetB0"   # options: "EfficientNetB0", "ResNet50"
FREEZE_BACKBONE_EPOCHS = 10   # epochs to keep backbone frozen before fine-tuning

# Classification threshold on the sigmoid probability for "malignant"
DECISION_THRESHOLD = 0.5
