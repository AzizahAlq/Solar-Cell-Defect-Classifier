Solar Cell Defect Detection: Project Description
Executive Summary

This project detects defects in solar cell images and provides results through a simple web application. It supports two approaches:

Supervised classifier when labels are available (“good” vs “defect”).

Unsupervised anomaly detection with a convolutional autoencoder (AE) when labels are limited.

A t-SNE plot is included to visualize how images group in feature space, helping to understand separation between normal and defective samples.

Objectives

Identify defective solar cells from images.

Offer both supervised and unsupervised options.

Provide a lightweight web interface for quick testing.

Visualize model behavior with a 2D embedding (t-SNE).

Keep data handling robust and simple.

Dataset and Preprocessing

Images are stored under a single folder.

An optional labels file (two columns: image, label) can be used for supervised training.

The pipeline accepts comma-separated or whitespace-separated files, with or without headers.

Paths are cleaned and validated; missing files are removed from training.

All images are converted to grayscale, resized to 224 × 224, and normalized.

Methods
1) Supervised Classification

A compact convolutional neural network predicts good or defect.

Output is a single probability for the “defect” class.

Suitable when reliable labels exist.

Outputs

Saved classifier model file.

Class names file (e.g., “good”, “defect”).

2) Unsupervised Anomaly Detection (Autoencoder)

A convolutional autoencoder reconstructs input images.

Reconstruction error (difference between input and output) is the anomaly score.

A threshold (e.g., 95th percentile of errors) marks images as normal or defective.

Suitable when labels are scarce or noisy.

Outputs

Saved autoencoder model.

Threshold file for decision making.

Training loss image.

CSV with each image’s reconstruction error and anomaly flag.

Visualization: t-SNE Embedding

Latent features from the autoencoder’s bottleneck are reduced to 2D using t-SNE.

Points are colored by anomaly decision (normal vs defect).

This plot reveals clusters, data structure, and potential mislabeled or unusual samples.

Outputs

t-SNE scatter plot image.

CSV with 2D coordinates and per-image metadata.

Web Application

A small Flask web app accepts image uploads and returns a decision.

Supports both model types:

Classifier: shows class scores and the top label.

Autoencoder: computes reconstruction error, compares it to the stored threshold, and returns good or defect.

The interface also links to the training loss plot and the t-SNE plot when available.

How to Use (High Level)

Place images in the dataset folder.

(Optional) Provide a two-column labels file (image, label) for supervised training.

Train either:

the autoencoder for unsupervised anomaly detection, or

the classifier for supervised detection.

Launch the web app and upload an image to see the result.

Review the loss curve and t-SNE plot to understand model behavior.

Results Summary

Supervised approach performs well with clean labels and provides straightforward probabilities.

Unsupervised approach highlights unusual images without labels and is effective for triage.

t-SNE improves understanding by showing how images cluster in feature space.

Strengths

Works with or without labels.

Simple, consistent preprocessing.

Clear artifacts for review (loss plot, t-SNE plot, per-image scores).

Easy to test through a web interface.

Limitations

Autoencoder threshold is heuristic and may require tuning.

Image quality and lighting variations can affect results.

The simple CNN and AE favor speed and clarity over maximum accuracy.

Future Improvements

Calibrate the autoencoder threshold using a small labeled set to maximize F1 score.

Try UMAP as an alternative to t-SNE for faster, stable embeddings.

Add self-supervised pretraining to improve features with few labels.

Export models to ONNX or Core ML for edge deployment.

Add basic audit tools (e.g., sample browser for false positives/negatives).

Ethical and Practical Notes

Use images responsibly and respect any data licenses.

Validate the system with domain experts before operational decisions.

Monitor for data drift and re-train when the image distribution changes.

Project Structure (Overview)

models/
Saved models, threshold file, training loss image, and t-SNE image.

solar_dataset/
Image folder and optional labels file; a cleaned labels file is generated.

uploads/
Temporary storage for images uploaded via the web app.

templates/ and static/
Web interface templates and styles.

web app file
Starts the Flask server and serves predictions.

training scripts
One for the classifier and one for the autoencoder with t-SNE.

Acknowledgments

Dataset inspiration: defective solar cells imagery from public sources such as ELPV.

Built for clarity and learning; designed to be easy to adapt and extend.
