"""
model.py
--------
Defines the CNN / Deep Learning model stage of the pipeline:
    Convolution -> Pooling -> Feature Extraction -> Fully Connected Layers

Two architectures are provided:
    1. build_custom_cnn()   - a CNN built from scratch
    2. build_transfer_model() - EfficientNetB0 / ResNet50 backbone + custom head

Both output a single sigmoid unit: P(malignant).
"""

from tensorflow.keras import layers, models, regularizers  # type: ignore[import-not-found]
from tensorflow.keras.applications import EfficientNetB0, ResNet50  # type: ignore[import-not-found]
from tensorflow.keras.optimizers import Adam  # type: ignore[import-not-found]

import config


def build_custom_cnn(input_shape=(224, 224, 3)) -> models.Model:
    """A CNN built from scratch, following the classic
    Conv -> BatchNorm -> ReLU -> MaxPool pattern, repeated with
    increasing filter depth, then a global-pooled fully-connected head."""

    inputs = layers.Input(shape=input_shape)

    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, (3, 3), padding="same",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    # Feature extraction summary
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="malignant_probability")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    return model


def build_transfer_model(input_shape=(224, 224, 3),
                          backbone_name: str = "EfficientNetB0",
                          freeze_backbone: bool = True) -> models.Model:
    """Transfer-learning model: a pretrained ImageNet backbone with a new
    classification head fine-tuned for benign/malignant mammogram classification."""

    if backbone_name == "EfficientNetB0":
        backbone = EfficientNetB0(include_top=False, weights="imagenet",
                                   input_shape=input_shape)
    elif backbone_name == "ResNet50":
        backbone = ResNet50(include_top=False, weights="imagenet",
                             input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone.trainable = not freeze_backbone

    inputs = backbone.input
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="malignant_probability")(x)

    model = models.Model(inputs, outputs, name=f"{backbone_name}_transfer")
    return model


def compile_model(model: models.Model, learning_rate: float = 1e-4) -> models.Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            "AUC",
            "Precision",
            "Recall",
        ],
    )
    return model


def get_model() -> models.Model:
    """Factory function that returns a compiled model according to config.py."""
    input_shape = (*config.IMG_SIZE, config.CHANNELS)

    if config.USE_TRANSFER_LEARNING:
        model = build_transfer_model(
            input_shape=input_shape,
            backbone_name=config.BACKBONE,
            freeze_backbone=True,
        )
    else:
        model = build_custom_cnn(input_shape=input_shape)

    return compile_model(model, learning_rate=config.LEARNING_RATE)
