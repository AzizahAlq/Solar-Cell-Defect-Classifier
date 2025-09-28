# Solar Cell Defect Classifier (Flask)

Upload an EL/IR image of a solar cell and get a defect prediction using your trained Keras model.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then visit http://127.0.0.1:5000

## Configure

- **Model file**: Put your model under `models/` and set `MODEL_PATH` in `app.py` (or env var `MODEL_PATH`).
- **Class names**: Update `CLASS_NAMES` in `app.py`, or set env var `CLASS_NAMES_JSON`, e.g.:
  ```bash
  export CLASS_NAMES_JSON='["good","crack","finger","inactive","scratch"]'
  ```
- **Image size**: Set `IMG_SIZE` in `app.py` (or env var: `IMG_SIZE=224,224`).
- **Rescale**: If you used `rescale=1./255` during training, keep the default `RESCALE=255.0`.
  If not, set `RESCALE=1.0` (no scaling).

## Exporting from Kaggle

In your Kaggle notebook (based on Belal Safy’s “Defective Solar Cells Detection”), after training:

```python
model.save("/kaggle/working/solar_defect_model.h5")
```

Download the `.h5` file and place it into `models/` here, then run the app.

## Notes

- Supports both binary (sigmoid) and multi-class (softmax) outputs.
- Secure file handling using `werkzeug.utils.secure_filename` and type checks.
- If you prefer **PyTorch**, load your `.pt` in place of TensorFlow and adapt preprocessing.
