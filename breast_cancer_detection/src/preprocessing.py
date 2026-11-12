"""
preprocessing.py
-----------------
Implements the preprocessing stage of the pipeline:
    Resize -> Normalize -> Contrast Enhancement (CLAHE) -> Data Augmentation

Provides:
    - preprocess_image(path): loads + preprocesses a single image (for inference)
    - build_data_generators(): returns Keras ImageDataGenerator flows for
      train / val / test, wired to use the same preprocessing function.
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[import-not-found]

import config


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to boost the
    visibility of masses/calcifications typical in mammograms."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def load_and_preprocess(path: str) -> np.ndarray:
    """Load an image from disk and apply resize + (optional) CLAHE.
    Returns a float32 array in [0, 1], RGB, shape (H, W, 3).
    Normalization to [0,1] is left as the final step so it composes cleanly
    with Keras' ImageDataGenerator (which also rescales)."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    image = cv2.resize(image, config.IMG_SIZE, interpolation=cv2.INTER_AREA)

    if config.APPLY_CLAHE:
        image = apply_clahe(image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32)


def preprocess_image(path: str) -> np.ndarray:
    """Full single-image preprocessing for inference: resize, CLAHE, normalize.
    Returns a batch-ready array of shape (1, H, W, 3) scaled to [0, 1]."""
    image = load_and_preprocess(path)
    image = image / 255.0
    return np.expand_dims(image, axis=0)


def _clahe_preprocessing_fn(image: np.ndarray) -> np.ndarray:
    """Preprocessing function hook for ImageDataGenerator. Keras passes
    images already resized to `target_size`; we just apply CLAHE here.
    Normalization (rescale=1/255) is handled by the generator itself."""
    image_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if config.APPLY_CLAHE:
        image_bgr = apply_clahe(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32)


def build_data_generators():
    """Builds train/val/test Keras generators implementing:
    Resize (via target_size) -> CLAHE (via preprocessing_function) ->
    Normalize (via rescale) -> Augmentation (train only).
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        preprocessing_function=_clahe_preprocessing_fn,
        **config.AUGMENTATION,
    )

    eval_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        preprocessing_function=_clahe_preprocessing_fn,
    )

    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        classes=config.CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
