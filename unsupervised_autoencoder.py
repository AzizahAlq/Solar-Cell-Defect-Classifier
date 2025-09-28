#!/usr/bin/env python3
# Unsupervised anomaly detection (autoencoder) — FIXED to provide targets

import os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

APP_ROOT   = Path(__file__).resolve().parent
DATA_ROOT  = APP_ROOT / "solar_dataset"
IMAGES_DIR = DATA_ROOT / "images"

IMG_SIZE   = (224, 224)
BATCH      = 32
EPOCHS     = 10
SEED       = 1337
ANOMALY_PERCENTILE = 95.0

MODEL_DIR   = APP_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
AE_PATH     = MODEL_DIR / "ae_model.keras"
THRESH_PATH = MODEL_DIR / "ae_threshold.json"
SCORES_CSV  = DATA_ROOT / "ae_scores.csv"

def list_all_images(root: Path):
    if not root.exists():
        raise SystemExit(f"[ERR] images folder not found: {root}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        raise SystemExit(f"[ERR] No image files found under: {root}")
    return sorted(paths)

def make_ds(paths):
    """Return (img, img) pairs so Keras has targets for MSE."""
    str_paths = [str(p) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices(str_paths)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)  # (H,W,1)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)  # << key fix
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_autoencoder(input_shape):
    inputs = keras.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)   # 112x112
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)        # 56x56
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)       # 28x28
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)       # 14x14
    latent = layers.Conv2D(256, 3, padding="same", activation="relu")(x)             # 14x14
    # Decoder
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(latent)  # 28x28
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)       # 56x56
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)       # 112x112
    x = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)       # 224x224
    outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)  # [0,1]
    model = keras.Model(inputs, outputs, name="autoencoder")
    # Use legacy Adam for better speed on Apple Silicon
    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss="mse")
    return model

def recon_errors(model, ds):
    errs = []
    for x, _ in ds:  # ds now yields (x, x)
        recon = model.predict(x, verbose=0)
        b = tf.reshape(x - recon, [tf.shape(x)[0], -1])
        mse = tf.reduce_mean(tf.square(b), axis=1).numpy()
        errs.extend(mse.tolist())
    return np.array(errs, dtype=np.float32)

def main():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.keras.utils.set_random_seed(SEED)

    paths = list_all_images(IMAGES_DIR)
    print(f"[OK] Found {len(paths)} images.")
    ds = make_ds(paths)

    ae = build_autoencoder(IMG_SIZE + (1,))
    ae.summary()
    ae.fit(ds, epochs=EPOCHS)

    errs = recon_errors(ae, ds)
    thr = float(np.percentile(errs, ANOMALY_PERCENTILE))
    print(f"[OK] Threshold at P{ANOMALY_PERCENTILE:.1f}: {thr:.6f}")

    ae.save(AE_PATH)
    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": thr, "percentile": ANOMALY_PERCENTILE}, f)
    print(f"[OK] Saved model → {AE_PATH}")
    print(f"[OK] Saved threshold → {THRESH_PATH}")

    df = pd.DataFrame({
        "image": [str(p) for p in paths],
        "recon_error": errs,
        "anomaly": (errs >= thr).astype(int)
    })
    df.to_csv(SCORES_CSV, index=False)
    print(f"[OK] Scores CSV → {SCORES_CSV}")
    print("Anomaly counts:", df["anomaly"].value_counts().to_dict())

    # Quick demo
    sample_path = str(paths[0])
    img = Image.open(sample_path).convert("L").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x[None, ..., None]
    recon = ae.predict(x, verbose=0)
    mse = float(np.mean((x - recon)**2))
    print(f"Demo on: {sample_path}")
    print(f"Reconstruction error: {mse:.6f}  | anomaly: {int(mse >= thr)}")

if __name__ == "__main__":
    main()
