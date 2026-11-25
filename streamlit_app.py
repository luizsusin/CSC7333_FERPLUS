# Streamlit demo of facial expression analysis using either an image upload or a webcam picture. A minimum of one
# trained model (hog_svm.joblib, cnn_scratch.pt, resnet18.pt or efficientnet_b0.pt) must be available in the models
# directory. This script uses a copy of the model, preprocessing and corruption code from the notebooks, so it runs
# on its own.
# Run it from the repository root with: streamlit run streamlit_app.py

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from skimage.feature import hog
from torch import nn
from torchvision import models

MODELS_DIR = Path(__file__).resolve().parent / "models"
CLASSES = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Fear"]
SEED = 42
PRETTY_NAMES = {"hog_svm": "HOG + SVM", "cnn_scratch": "CNN (scratch)", "resnet18": "ResNet-18",
                "efficientnet_b0": "EfficientNet-B0"}
MODEL_FILES = {"hog_svm": "hog_svm.joblib", "cnn_scratch": "cnn_scratch.pt", "resnet18": "resnet18.pt",
               "efficientnet_b0": "efficientnet_b0.pt"}

# Same corruptions as in notebook 07, with the same 5 levels.
SEVERITIES = {
    "gaussian_noise": [4, 8, 16, 24, 32],
    "occlusion": [8, 12, 16, 20, 24],
    "low_light": [0.6, 0.4, 0.25, 0.15, 0.08],
}
CORRUPTION_NAMES = {"gaussian_noise": "Gaussian noise", "occlusion": "Occlusion", "low_light": "Low light"}
SEVERITY_LABELS = {"gaussian_noise": "noise std (gray levels)", "occlusion": "patch side (px of 48)",
                   "low_light": "brightness factor"}
REGIONS = {"eyes": (12, 24), "mouth": (34, 46)}
REGION_NAMES = {"eyes": "Eyes hidden", "mouth": "Mouth hidden"}


class ScratchCNN(nn.Module):
    # The network from notebook 03. Its structure must match the saved weights exactly.
    def __init__(self, base_filters=64, dropout=0.4, block_dropout=0.25):
        super().__init__()
        blocks, in_ch = [], 1
        for out_ch in [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(block_dropout)))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(256, len(CLASSES)))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


@st.cache_resource
def load_model(name):
    # Either the HOG + SVM bundle or a network restored from its checkpoint. Streamlit keeps it in memory after it
    # is first loaded.
    if name == "hog_svm":
        return joblib.load(MODELS_DIR / "hog_svm.joblib")
    checkpoint = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    if name == "cnn_scratch":
        model = ScratchCNN(**checkpoint["model_kwargs"])
    elif name == "resnet18":
        model = models.resnet18()
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(CLASSES)))
    else:
        model = models.efficientnet_b0()
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def prepare(face, name):
    # Change the dimensions of the face crop to fit the model input. Reducing the size of an image involves
    # averaging pixels, whereas increasing in size is done using bicubic interpolation. This is the method used to
    # convert 48x48 data to 224x224 for training.
    size = 224 if name in ("resnet18", "efficientnet_b0") else 48
    interpolation = cv2.INTER_CUBIC if face.shape[0] < size else cv2.INTER_AREA
    return cv2.resize(face, (size, size), interpolation=interpolation)


def to_tensor(face, name):
    # Normalize a single face in the same way as for training.
    x = torch.from_numpy(face)[None, None].float() * (1 / 255)
    if name == "cnn_scratch":
        return (x - 0.5) / 0.5
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x.repeat(1, 3, 1, 1) - mean) / std


def predict(name, face):
    # Class scores for one grayscale face crop. As the SVM does not output probabilities, its decision values are
    # passed through a softmax function. This provides an ordering of classes but these are not calibrated.
    model = load_model(name)
    face = prepare(face, name)
    if name == "hog_svm":
        p = model["hog_params"]
        features = hog(face, orientations=p["orientations"], pixels_per_cell=(p["cell_size"], p["cell_size"]),
                       cells_per_block=(p["block_size"], p["block_size"]), block_norm="L2-Hys")
        scores = model["pipeline"].decision_function(features[None].astype(np.float32))[0]
        scores = np.exp(scores - scores.max())
        scores = scores / scores.sum()
    else:
        with torch.no_grad():
            scores = torch.softmax(model(to_tensor(face, name)), dim=1)[0].numpy()
    return dict(zip(CLASSES, scores.astype(float)))


def grad_cam(name, face):
    # Grad-CAM overlay for the output class (see notebook 06). The derivative of the output class score is used to
    # produce a weighting for each feature map in the final convolutional block. The positive part of the weighted
    # sum of the feature maps is then overlaid on the face.
    model = load_model(name)
    face = prepare(face, name)
    layer = model.layer4[-1] if name == "resnet18" else model.features[-1]
    saved = {}
    hook = layer.register_forward_hook(lambda module, inputs, output: saved.update(maps=output))
    logits = model(to_tensor(face, name))
    hook.remove()
    pred = int(logits.argmax(1))
    grads = torch.autograd.grad(logits[0, pred], saved["maps"])[0]
    cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * saved["maps"]).sum(dim=1))[0].detach().numpy()
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, face.shape[::-1])), cv2.COLORMAP_JET)
    return np.uint8(0.55 * cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) + 0.45 * cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def find_face(gray):
    # Locate the largest frontal face using the Haar cascade included in OpenCV. A 15% border is added around the
    # box, which is then made square, as the FER2013 faces are loosely cropped to include both the forehead and
    # chin. The crop and box, or the middle square of the image and None if no face is detected, are output.
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        h, w = gray.shape
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        return gray[top:top + side, left:left + side], None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    side = min(min(gray.shape[1], x + w + pad) - x0, min(gray.shape[0], y + h + pad) - y0)
    return gray[y0:y0 + side, x0:x0 + side], (int(x0), int(y0), int(side), int(side))


def corrupt(face, kind, level):
    # One 48x48 face, corrupted as in notebook 07 at level 1-5. The same random numbers are used.
    severity = SEVERITIES[kind][level - 1]
    rng = np.random.default_rng([SEED, list(SEVERITIES).index(kind), level])
    if kind == "gaussian_noise":
        noisy = face.astype(np.float32) + rng.normal(0, severity, face.shape).astype(np.float32)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    if kind == "occlusion":
        out = face.copy()
        y, x = rng.integers(0, 49 - severity, 2)
        out[y:y + severity, x:x + severity] = 0
        return out
    return np.rint(face.astype(np.float32) * severity).astype(np.uint8)


def occlude_region(face, region):
    # Cover the eye area (as with sunglasses) or mouth area (as with a face mask) of a 48x48 face.
    top, bottom = REGIONS[region]
    out = face.copy()
    out[top:bottom] = 0
    return out


st.set_page_config(page_title="FERPlus Project - Group 6", page_icon="", layout="wide")
st.title("Facial Expression Recognition")
st.caption("CSC 7333 Machine Learning - Group 6")

available = [name for name, file in MODEL_FILES.items() if (MODELS_DIR / file).exists()]
if not available:
    st.error(f"No trained models found in `{MODELS_DIR}`. Run the notebooks first "
             "or copy the weights from Google Drive into `models/`.")
    st.stop()

with st.sidebar:
    model_name = st.selectbox("Model", available, format_func=PRETTY_NAMES.get, index=len(available) - 1)
    show_cam = st.checkbox("Show Grad-CAM", value=True,
                           help="Highlights the regions the network used for its prediction.")
    st.markdown("---")
    st.subheader("Robustness test")
    corruption = st.selectbox(
        "Corruption", ["none", *SEVERITIES, *REGION_NAMES],
        format_func=lambda c: {"none": "None", **CORRUPTION_NAMES, **REGION_NAMES}[c],
        help="Corrupts the 48×48 face to see how the prediction changes.")
    level = None
    if corruption in SEVERITIES:
        level = st.slider("Level", 1, len(SEVERITIES[corruption]), 3)
        st.caption(f"{SEVERITY_LABELS[corruption].capitalize()}: {SEVERITIES[corruption][level - 1]:g}")
    st.markdown("---")
    st.markdown("Classes: Neutral, Happiness, Surprise, Sadness, Anger, Fear")

source = st.radio("Input", ["Upload a photo", "Use the webcam"], horizontal=True)
if source == "Upload a photo":
    file = st.file_uploader("Photo with a face", type=["jpg", "jpeg", "png", "bmp", "webp"])
else:
    file = st.camera_input("Take a picture")

if file is None:
    st.info("Upload a photo or take a picture to get a prediction.")
    st.stop()

image = np.array(Image.open(file).convert("RGB"))
face, box = find_face(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
if box is None:
    st.warning("No face detected, so the center of the image is used instead.")

clean_scores = None
if corruption != "none":
    # As in notebook 07, the face is corrupted at 48x48, the same resolution as the training data.
    face48 = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    clean_scores = predict(model_name, face48)
    face = corrupt(face48, corruption, level) if level else occlude_region(face48, corruption)
scores = predict(model_name, face)
label = max(scores, key=scores.get)

col_img, col_face, col_scores = st.columns([2, 1, 2])
with col_img:
    shown = image.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 200, 0), max(2, image.shape[1] // 200))
    st.image(shown, caption="Input", width="stretch")

with col_face:
    model_input = prepare(face, model_name)
    caption = f"Model input ({len(model_input)}×{len(model_input)})"
    if corruption != "none":
        name = {**CORRUPTION_NAMES, **REGION_NAMES}[corruption]
        caption += f", {name.lower()}" + (f" level {level}" if level else "")
    st.image(model_input, caption=caption, width="stretch", clamp=True)
    if show_cam and model_name != "hog_svm":
        st.image(grad_cam(model_name, face), caption="Grad-CAM", width="stretch")
    elif show_cam:
        st.caption("Grad-CAM is only available for the neural networks.")

with col_scores:
    st.subheader(f"Prediction: {label}")
    if clean_scores is not None:
        clean_label = max(clean_scores, key=clean_scores.get)
        st.caption(f"Without the corruption: {clean_label} ({clean_scores[clean_label]:.0%}).")
    st.bar_chart(pd.Series(scores, name="score"), horizontal=True)
    if model_name == "hog_svm":
        st.caption("SVM scores are softmax-scaled decision values (not calibrated probabilities).")
