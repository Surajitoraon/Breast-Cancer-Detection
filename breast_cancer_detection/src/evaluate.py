"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set:
    - Accuracy, Precision, Recall, F1-score (classification report)
    - Confusion matrix (saved as image)
    - ROC curve + AUC (saved as image)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from tensorflow.keras.models import load_model  # type: ignore[import-not-found]

import config
from preprocessing import build_data_generators


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return roc_auc


def main():
    print(f"Loading model from {config.BEST_MODEL_PATH}...")
    model = load_model(config.BEST_MODEL_PATH)

    _, _, test_gen = build_data_generators()
    print(f"Evaluating on {test_gen.samples} test images...")

    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_true = test_gen.classes
    y_pred = (y_prob >= config.DECISION_THRESHOLD).astype(int)

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    report_text = str(report)
    print("\nClassification Report:\n", report_text)

    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, config.CLASS_NAMES,
                           os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png"))

    roc_auc = plot_roc_curve(y_true, y_prob,
                              os.path.join(config.OUTPUTS_DIR, "roc_curve.png"))
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"Reports and plots saved to {config.OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
