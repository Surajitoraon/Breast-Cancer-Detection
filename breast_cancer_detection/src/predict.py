"""
predict.py
----------
Runs inference on a single mammography image and reports:
    - Predicted class (Benign / Malignant)
    - Probability / confidence score

Usage:
    python predict.py --image path/to/mammogram.png
    python predict.py --image path/to/mammogram.png --model models/best_model.h5
"""

import argparse
import importlib

import config
from preprocessing import preprocess_image


def _load_model_function():
    for module_name in ("tensorflow.keras.models", "keras.models"):
        try:
            module = importlib.import_module(module_name)
            return module.load_model
        except ImportError:
            continue

    raise ImportError(
        "TensorFlow or Keras must be installed to load the model. "
        "Install the required dependency and try again."
    )


def predict_image(image_path: str, model_path: str = config.BEST_MODEL_PATH) -> dict:
    model = _load_model_function()(model_path)
    image_batch = preprocess_image(image_path)

    probability_malignant = float(model.predict(image_batch, verbose=0)[0][0])

    if probability_malignant >= config.DECISION_THRESHOLD:
        predicted_class = "Malignant"
        confidence = probability_malignant
    else:
        predicted_class = "Benign"
        confidence = 1.0 - probability_malignant

    return {
        "predicted_class": predicted_class,
        "probability_malignant": probability_malignant,
        "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict Benign/Malignant from a mammography image."
    )
    parser.add_argument("--image", required=True, help="Path to the mammogram image.")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH,
                         help="Path to the trained model (.h5).")
    args = parser.parse_args()

    result = predict_image(args.image, args.model)

    print(f"\nImage: {args.image}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Probability (malignant): {result['probability_malignant']:.4f}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nNote: This is a decision-support prediction only. "
          "Please consult a qualified radiologist for diagnosis.")


if __name__ == "__main__":
    main()
