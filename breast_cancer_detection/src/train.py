"""
train.py
--------
End-to-end training pipeline:
    1. Build data generators (preprocessing + augmentation)
    2. Build the CNN model
    3. Train with callbacks (checkpoint, early stopping, LR scheduling)
    4. Optionally fine-tune the backbone (if using transfer learning)
    5. Save training curves to outputs/
"""

# pyright: reportMissingImports=false

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow.keras.callbacks import (  # type: ignore[import-not-found]
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    )
else:
    from tensorflow.keras.callbacks import (  # type: ignore[import-not-found]
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    )

# pyright: reportMissingImports=false

import config
from model import get_model
from preprocessing import build_data_generators


def get_class_weights(train_gen) -> dict:
    """Compute class weights to counter class imbalance
    (malignant cases are typically under-represented)."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    print("Building data generators (resize -> CLAHE -> normalize -> augment)...")
    train_gen, val_gen, _ = build_data_generators()

    print(f"Found {train_gen.samples} training images, "
          f"{val_gen.samples} validation images.")
    print(f"Class indices: {train_gen.class_indices}")

    print("Building model...")
    model = get_model()
    model.summary()

    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = get_class_weights(train_gen)
        print(f"Using class weights: {class_weights}")

    callback_list = [
        ModelCheckpoint(config.BEST_MODEL_PATH, monitor="val_auc",
                         mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_auc", mode="max", patience=8,
                       restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                           min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(config.OUTPUTS_DIR, "training_log.csv")),
    ]

    # Phase 1: train with frozen backbone (if transfer learning) / full custom CNN
    initial_epochs = (config.FREEZE_BACKBONE_EPOCHS
                       if config.USE_TRANSFER_LEARNING else config.EPOCHS)

    print(f"\n--- Phase 1 training ({initial_epochs} epochs) ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=initial_epochs,
        class_weight=class_weights,
        callbacks=callback_list,
    )

    # Phase 2: fine-tune the backbone for transfer learning models
    if config.USE_TRANSFER_LEARNING and config.EPOCHS > initial_epochs:
        print("\n--- Phase 2: fine-tuning backbone ---")
        base_model = model.layers[1] if hasattr(model.layers[1], "trainable") else None
        for layer in model.layers:
            layer.trainable = True

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "AUC", "Precision", "Recall"],
        )
        fine_tune_history = model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=initial_epochs,
            epochs=config.EPOCHS,
            class_weight=class_weights,
            callbacks=callback_list,
        )
        for key in history.history:
            history.history[key].extend(fine_tune_history.history.get(key, []))

    print(f"\nSaving final model to {config.BEST_MODEL_PATH}")
    model.save(config.BEST_MODEL_PATH)

    history_path = os.path.join(config.OUTPUTS_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    plot_history(history, os.path.join(config.OUTPUTS_DIR, "training_curves.png"))
    print("Training complete. Curves saved to outputs/training_curves.png")


if __name__ == "__main__":
    main()
