# Sign Language MNIST Streamlit Dashboard
# Uses baseline RandomForest model from baseline_model/baseline_model.pkl.
# Recreates StandardScaler from training CSV to match training pipeline.

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "baseline_model", "baseline_model.pkl")
TRAIN_CSV = os.path.join(
    BASE_DIR, "data", "sign-language-mnist",
    "sign_mnist_train", "sign_mnist_train.csv",
)

IMAGE_SIZE = 28

# Label map: 0-24 skipping 9 (J) and 25 (Z) -- they require motion
LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    23: "X", 24: "Y",
}


# Model and scaler loading (cached once per session)

@st.cache_resource(show_spinner="Loading baseline model...")
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found.\n\n"
            f"Expected: `{MODEL_PATH}`\n\n"
            "Run baseline_model/model.ipynb or model.py first."
        )
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if not os.path.exists(TRAIN_CSV):
        st.error(
            "Training CSV not found.\n\n"
            f"Expected: `{TRAIN_CSV}`\n\n"
            "Place the Sign-Language MNIST training CSV in data/."
        )
        st.stop()

    # Recreate the same StandardScaler used during training.
    df = pd.read_csv(TRAIN_CSV)
    X_train = df.drop("label", axis=1).values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train)

    return model, scaler


# Image preprocessing -- must match training pipeline exactly

def preprocess_image(image: Image.Image, scaler: StandardScaler) -> np.ndarray:
    # Grayscale -> 28x28 -> flatten to 784 -> StandardScaler transform.
    img = image.convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(img, dtype=np.float32).flatten()
    features = pixels.reshape(1, -1)
    features_scaled = scaler.transform(features)
    return features_scaled


# Prediction

def predict(features: np.ndarray, model):
    proba = model.predict_proba(features)[0]
    classes = model.classes_
    top_idx = int(np.argmax(proba))
    pred_label = int(classes[top_idx])
    confidence = float(proba[top_idx])
    letter = LABEL_TO_LETTER.get(pred_label, "?")
    return letter, confidence, classes, proba


# UI

st.set_page_config(page_title="ASL Sign Classifier", layout="wide")

st.title("ASL Sign Language Detector")
st.write("Upload a hand-sign photo to predict the American Sign Language letter.")

# Sidebar
with st.sidebar:
    model, scaler = load_model_and_scaler()
    st.success("Random Forest baseline loaded")

# Main layout
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Drop a PNG / JPG hand-sign image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        example_dir = os.path.join(BASE_DIR, "data", "sign-language-mnist")
        for img_name in ["amer_sign2.png", "amer_sign3.png", "american_sign_language.PNG"]:
            img_path = os.path.join(example_dir, img_name)
            if os.path.exists(img_path):
                st.subheader("ASL Alphabet Reference")
                st.image(img_path, use_container_width=True)
                break

with right_col:
    if uploaded_file is not None:
        # Preprocess: grayscale -> 28x28 -> 784 features -> StandardScaler
        features = preprocess_image(image, scaler)
        letter, confidence, classes, proba = predict(features, model)

        # Prediction result
        st.subheader("Prediction")
        st.metric(label="Predicted Letter", value=letter)
        st.metric(label="Confidence", value=f"{confidence:.1%}")

        # Top-5 candidates
        st.subheader("Top-5 Candidates")
        top5_idx = np.argsort(proba)[::-1][:5]
        top5_data = {
            "Letter": [LABEL_TO_LETTER.get(int(classes[i]), "?") for i in top5_idx],
            "Confidence": [f"{proba[i]:.1%}" for i in top5_idx],
        }
        st.table(top5_data)

    else:
        st.info("Upload an image to see the prediction.")
