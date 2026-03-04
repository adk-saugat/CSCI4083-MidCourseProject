import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("multilayer_model/multilayer_model.keras")

model = load_model()

# Labels 0-8 = A-I, label 9 = K (J is skipped), 10-23 = L-Y (Z is skipped)
label_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",  
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "X",
    23: "Y", 
}

# Streamlit UI
st.title("Sign Language Letter Predictor")
st.write("Upload a hand sign image to predict the letter")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])


def preprocess_image(pil_img):
    """
    Preprocess to MATCH the training data format exactly.

    """
    
    # Convert PIL image to numpy array (RGB)
    img = np.array(pil_img)

    # Convert RGB to grayscale — same single-channel format as training data
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    # If already grayscale (2D), use as-is

    # Resize to 28x28 — same dimensions as training data
    img_28 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0] — same normalization as training (/ 255.0)
    img_normalized = img_28.astype(np.float32) / 255.0

    # Reshape to (1, 28, 28, 1) — CNN input format
    model_input = img_normalized.reshape(1, 28, 28, 1)

    return model_input, img_28


if uploaded_file:
    image = Image.open(uploaded_file)

    # Correct orientation from EXIF metadata (e.g. photos taken on phones)
    image = ImageOps.exif_transpose(image)

    # Convert to RGB to ensure consistent channel handling
    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image")

    # Preprocess matching training pipeline
    model_input, preview_img = preprocess_image(image)

    # Run prediction
    preds = model.predict(model_input)[0]

    st.caption("What the model sees (28×28 grayscale)")
    st.image(preview_img, width=112)

    top3 = np.argsort(preds)[-3:][::-1]
    prediction = top3[0]
    confidence = preds[prediction]

    st.success(f"Predicted Letter: **{label_map.get(prediction, 'Unknown')}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    with st.expander("Top 3 guesses"):
        for i, idx in enumerate(top3, 1):
            st.write(f"{i}. **{label_map.get(idx, 'Unknown')}** — {preds[idx] * 100:.1f}%")

    # Low confidence warning
    if confidence < 0.5:
        st.warning(
            "Low confidence. Try a clearer photo with:\n"
            "- Good lighting\n"
            "- Hand centered in frame\n"
            "- Plain background\n"
            "- Similar angle to ASL reference images"
        )