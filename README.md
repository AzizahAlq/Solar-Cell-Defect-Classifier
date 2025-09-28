# Solar-Cell-Defect-Classifier
Solar Cell Defect Classifier 
Solar Cell Defect Detection (Supervised & Unsupervised) + Flask App

A small end-to-end project for detecting defects in solar cell images.
It includes:

Supervised classifier (binary: good vs defect)

Unsupervised anomaly detector using a convolutional autoencoder (AE)

t-SNE embedding plot of AE latent features

A simple Flask web app to upload an image and get a prediction

Robust utilities to parse messy labels.csv files and resolve image paths

Project Structure
solar_flask_app/
├─ models/                       # saved models & plots
│  ├─ ae_model.keras             # autoencoder (unsupervised)
│  ├─ ae_threshold.json          # AE threshold (percentile of recon error)
│  ├─ ae_loss.png                # training loss curve
│  └─ ae_tsne.png                # t-SNE 2D embedding of latents
├─ solar_dataset/
│  ├─ images/                    # all images live here
│  ├─ labels.csv                 # raw labels (any format: comma or whitespace)
│  └─ labels_standard.csv        # canonical image,label CSV (created by script)
├─ uploads/                      # uploaded images from Flask
├─ templates/
│  └─ index.html                 # UI for Flask
├─ static/
│  └─ style.css                  # basic styling
├─ solar_app.py                  # Flask app (supports classifier or AE)
├─ train_all.py                  # supervised: train on 100% images (binary CNN)
├─ unsupervised_autoencoder_tsne.py # AE training + threshold + t-SNE
└─ requirements.txt

Dataset Assumptions

Images are under solar_dataset/images/ (any nested folders are fine).

For supervised training, solar_dataset/labels.csv has two columns:

image – path or filename (absolute or relative; script resolves)

label – 0 for good, 1 for defect (floats 0.0/1.0 also OK)

The scripts auto-detect comma CSV with/without header or whitespace-separated.
They write a clean labels_standard.csv for consistency.

Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt


Apple Silicon (M-series): using tensorflow-macos + tensorflow-metal is recommended; the scripts already use the legacy Adam optimizer for speed on macOS.

Training
1) Unsupervised (Autoencoder + t-SNE)

Learns to reconstruct normal patterns; reconstruction error is used as an anomaly score.

python unsupervised_autoencoder_tsne.py


Outputs:

models/ae_model.keras – the trained AE

models/ae_threshold.json – anomaly threshold (95th percentile by default)

models/ae_loss.png – loss curve

models/ae_tsne.png – 2D t-SNE plot (colored by anomaly)

solar_dataset/ae_scores.csv – per-image reconstruction error & anomaly flag

Tune: change ANOMALY_PERCENTILE to 99 for stricter detection or 90 for more sensitive.

2) Supervised (Binary Classifier)

Trains a small grayscale CNN on all labeled images (no hold-out).
Edit train_all.py if you want a validation split.

python train_all.py


Outputs:

models/solar_defect_model.keras

models/class_names.json (["good","defect"])

Web App (Flask)

The Flask app supports both model types:

Classifier: outputs class probabilities.

Autoencoder: outputs reconstruction error; the app compares it to the saved threshold and returns good/defect.

Run:

# choose which model to serve (AE shown here)
export MODEL_PATH="$PWD/models/ae_model.keras"
# If your model already rescales internally (e.g., layers.Rescaling(1./255)), do:
# export RESCALE=1.0

python solar_app.py
# open http://127.0.0.1:5000


UI shows:

Uploaded image

Predicted label + top scores

For AE: reconstruction error and threshold

Links to training plots (loss & t-SNE) if present

How It Works
Supervised

Grayscale input (H, W, 1) resized to 224×224

Small CNN → Dense(1, sigmoid) → P(defect)

Labels parsed and normalized from labels.csv, paths auto-resolved

Unsupervised (AE)

Grayscale AE trained with MSE reconstruction loss

Per-image reconstruction error is computed

Threshold = percentile of errors (default P95) ⇒ anomaly flag

Latents from the bottleneck are aggregated & reduced via t-SNE for visualization

Troubleshooting

“No image files found…”
Your image paths in labels.csv don’t match files on disk. The scripts print dropped rows and write a cleaned CSV.

Unexpected prediction shape (1, 224, 224, 1)
You’re loading an AE into the Flask classifier route. This repo’s solar_app.py handles both; it computes AE error and uses ae_threshold.json. Make sure that file exists (run the AE training first).

Images look too dark/bright
Check preprocessing scale. If your model applies layers.Rescaling(1./255) internally, set RESCALE=1.0 in the environment for inference.

To-Do Ideas

Calibrate AE threshold using a small labeled subset (maximize F1 on a dev set)

Add ROC-AUC/F1 metric logging per epoch if labels are available

Swap t-SNE for UMAP or add PCA(50) → t-SNE for speed

Export ONNX / CoreML for mobile/edge inference





Acknowledgments

Dataset inspired by ELPV defective solar cells (Kaggle).
This project is educational and minimal by design—feel free to adapt and extend.
