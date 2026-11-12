"""
app.py
------
Flask web server for the Breast Cancer Detection project.

Routes:
  GET  /          → Serves the main web UI.
  POST /predict   → Accepts a mammography image upload and returns a JSON
                    prediction result from the trained model.
  GET  /health    → Simple health check endpoint.

Usage:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import io
import os
import sys
import uuid
import tempfile
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Add src/ to path so we can import project modules
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload
app.config["UPLOAD_FOLDER"] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif", "avif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Lazy model loading – the model is loaded once on first request
# ---------------------------------------------------------------------------
_model = None
_model_error = None


def get_model():
    """Load model lazily and cache it. Returns (model, error_string)."""
    global _model, _model_error
    if _model is not None:
        return _model, None
    if _model_error is not None:
        return None, _model_error
    try:
        import config
        from predict import predict_image  # noqa: F401 – test import works

        # Import keras load to pre-warm
        from tensorflow.keras.models import load_model  # type: ignore

        if not os.path.exists(config.BEST_MODEL_PATH):
            _model_error = (
                f"Model file not found at '{config.BEST_MODEL_PATH}'. "
                "Please run 'python src/train.py' first."
            )
            return None, _model_error

        print(f"Loading model from {config.BEST_MODEL_PATH} …", flush=True)
        _model = load_model(config.BEST_MODEL_PATH)
        print("Model loaded successfully.", flush=True)
        return _model, None
    except Exception as exc:
        _model_error = str(exc)
        return None, _model_error


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    model, err = get_model()
    return jsonify({
        "status": "ok" if model is not None else "model_unavailable",
        "model_error": err,
    })


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Validate file presence
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # 2. Ensure model is ready
    model, err = get_model()
    if err:
        return jsonify({"error": err}), 503

    # 3. Save to a temp file and run inference
    ext = secure_filename(file.filename).rsplit(".", 1)[-1]
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}.{ext}")
    try:
        file.save(tmp_path)
        from predict import predict_image
        import config
        result = predict_image(tmp_path, config.BEST_MODEL_PATH)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify({
        "predicted_class": result["predicted_class"],
        "probability_malignant": round(result["probability_malignant"] * 100, 2),
        "confidence": round(result["confidence"] * 100, 2),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pre-load the model at startup so the first request isn't slow
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
