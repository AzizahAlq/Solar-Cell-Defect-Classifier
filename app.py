#!/usr/bin/env python3
import os, json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Configuration
# -----------------------------
app = Flask(__name__)
app_root = Path(__file__).resolve().parent
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me")

# Model path: point to classifier .keras OR AE .keras
MODEL_PATH = os.environ.get("MODEL_PATH", str(app_root / "models" / "ae_model.keras"))

# Threshold file for AE (created by unsupervised training script)
THRESH_PATH = app_root / "models" / "ae_threshold.json"

# Class names for classifier fallback
CLASS_NAMES_ENV = os.environ.get("CLASS_NAMES_JSON")
CLASS_NAMES = list(json.loads(CLASS_NAMES_ENV)) if CLASS_NAMES_ENV else ["good", "defect"]

# Image preprocessing
IMG_SIZE = tuple(map(int, os.environ.get("IMG_SIZE", "224,224").split(",")))
RESCALE = float(os.environ.get("RESCALE", "255.0"))  # set 1.0 if your model already rescales internally

# Upload settings
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024)))
uploads_dir = app_root / "uploads"
uploads_dir.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(uploads_dir)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Optional plots produced by your training script
TSNE_PNG = app_root / "models" / "ae_tsne.png"
LOSS_PNG = app_root / "models" / "ae_loss.png"

# -----------------------------
# Model loading + helpers
# -----------------------------
_model = None
_model_load_error = None
INPUT_CHANNELS = 3  # default; corrected after load
_AE_THRESHOLD = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global _model, _model_load_error, INPUT_CHANNELS
    if _model is not None:
        return _model
    try:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        try:
            INPUT_CHANNELS = int(_model.input_shape[-1])
        except Exception:
            INPUT_CHANNELS = 3
        return _model
    except Exception as e:
        _model_load_error = str(e)
        return None

def get_ae_threshold() -> float:
    """Load and cache AE threshold from JSON."""
    global _AE_THRESHOLD
    if _AE_THRESHOLD is not None:
        return _AE_THRESHOLD
    if not THRESH_PATH.exists():
        raise RuntimeError(f"Threshold file not found: {THRESH_PATH}")
    with open(THRESH_PATH) as f:
        data = json.load(f)
    _AE_THRESHOLD = float(data["threshold"])
    return _AE_THRESHOLD

def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.
    Auto-grayscale if the model expects 1 channel.
    Returns shape (1, H, W, C) with values in [0,1] if RESCALE=255.0.
    """
    mode = "L" if INPUT_CHANNELS == 1 else "RGB"
    img = Image.open(img_path).convert(mode).resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    if INPUT_CHANNELS == 1 and x.ndim == 2:
        x = np.expand_dims(x, -1)  # (H, W, 1)
    if RESCALE != 1.0:
        x = x / RESCALE
    x = np.expand_dims(x, 0)       # (1, H, W, C)
    return x

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        classes=CLASS_NAMES,
        model_path=MODEL_PATH,
        error=_model_load_error,
        has_loss=LOSS_PNG.exists(),
        has_tsne=TSNE_PNG.exists()
    )

@app.route("/classify", methods=["POST"])
def classify():
    mdl = load_model()
    if mdl is None:
        flash(f"Model failed to load from '{MODEL_PATH}'. Error: {_model_load_error}", "error")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("No file part", "error")
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload an image.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = uploads_dir / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}_{filename}"
    file.save(str(save_path))

    # Preprocess & predict
    x = preprocess_image(str(save_path))
    preds = mdl.predict(x)

    # ---- Branch by prediction shape ----
    # A) Classifier: (1, K) with sigmoid or softmax
    if preds.ndim == 2:
        if preds.shape[1] == 1:
            p1 = float(preds[0, 0])
            probs = [1.0 - p1, p1]  # [good, defect]
            pred_idx = int(np.argmax(probs))
        else:
            probs = preds[0].tolist()
            pred_idx = int(np.argmax(probs))

        pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        topk = min(3, len(CLASS_NAMES))
        top_indices = np.argsort(probs)[::-1][:topk]
        top_items = [{"label": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}",
                      "prob": float(probs[i])} for i in top_indices]

        return render_template(
            "index.html",
            classes=CLASS_NAMES,
            model_path=MODEL_PATH,
            filename=save_path.name,
            pred_label=pred_label,
            top_items=top_items,
            has_loss=LOSS_PNG.exists(),
            has_tsne=TSNE_PNG.exists()
        )

    # B) Autoencoder: (1, H, W, C) reconstruction
    if preds.ndim == 4 and preds.shape[1:3] == (IMG_SIZE[0], IMG_SIZE[1]):
        try:
            thr = get_ae_threshold()
        except Exception as e:
            flash(f"AE threshold missing: {e}", "error")
            return redirect(url_for("index"))

        recon = preds
        err = float(np.mean((x - recon) ** 2))
        anomaly = int(err >= thr)
        # pseudo-probability for display
        norm = err / (thr + 1e-12)
        top_items = [
            {"label": "good",   "prob": float(max(0.0, 1.0 - norm))},
            {"label": "defect", "prob": float(min(1.0, norm))}
        ]
        pred_label = "defect" if anomaly else "good"

        return render_template(
            "index.html",
            classes=CLASS_NAMES,
            model_path=MODEL_PATH,
            filename=save_path.name,
            pred_label=pred_label,
            top_items=top_items,
            ae_error=err,
            ae_threshold=thr,
            has_loss=LOSS_PNG.exists(),
            has_tsne=TSNE_PNG.exists()
        )

    # Unknown model output
    flash(f"Unexpected prediction shape: {preds.shape}", "error")
    return redirect(url_for("index"))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Optional: serve training plots if they exist
@app.route("/plots/loss")
def plot_loss():
    if not LOSS_PNG.exists():
        return "Loss plot not found.", 404
    return send_file(str(LOSS_PNG), mimetype="image/png")

@app.route("/plots/tsne")
def plot_tsne():
    if not TSNE_PNG.exists():
        return "t-SNE plot not found.", 404
    return send_file(str(TSNE_PNG), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
