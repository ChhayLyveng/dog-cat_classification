import streamlit as st
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── ANN definition (must match training notebook exactly) ──────────────────
def my_ANN(X, W1, b1, W2, b2, W3, b3, W4, b4):
    Z1 = torch.matmul(X, W1) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.matmul(A1, W2) + b2
    A2 = torch.relu(Z2)
    Z3 = torch.matmul(A2, W3) + b3
    A3 = torch.relu(Z3)
    Z4 = torch.matmul(A3, W4) + b4
    return Z4

# ── Load weights ────────────────────────────────────────────────────────────
@st.cache_resource
def load_weights(weights_path: str):
    checkpoint = torch.load(weights_path, map_location="cpu")
    weights = {k: v for k, v in checkpoint.items() if k != "config"}
    config = checkpoint.get("config", {"img_size": 64, "use_color": True})
    return weights, config

# ── Preprocess an uploaded image ────────────────────────────────────────────
def preprocess(pil_image: Image.Image, img_size: int, use_color: bool) -> torch.Tensor:
    img = np.array(pil_image.convert("RGB"))           # ensure RGB numpy
    img = cv.resize(img, (img_size, img_size))
    if not use_color:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)[:, :, np.newaxis]
    x = torch.tensor(img.flatten() / 255.0, dtype=torch.float32).unsqueeze(0)
    return x

# ── Inference ───────────────────────────────────────────────────────────────
def predict(x: torch.Tensor, weights: dict) -> tuple[str, float, float]:
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    W3, b3 = weights["W3"], weights["b3"]
    W4, b4 = weights["W4"], weights["b4"]
    with torch.no_grad():
        Z = my_ANN(x, W1, b1, W2, b2, W3, b3, W4, b4)
        probs = torch.softmax(Z, dim=1).squeeze().numpy()
        pred  = int(torch.argmax(Z, dim=1).item())
    label = "Dog 🐶" if pred == 1 else "Cat 🐱"
    return label, float(probs[0]), float(probs[1])

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🐾 Cat vs Dog Classifier")
st.markdown(
    "Upload a photo and the model will tell you whether it's a **cat** or a **dog**. "
    "This app uses the custom PyTorch ANN trained in your notebook."
)

weights_file = st.file_uploader(
    "① Upload your trained weights (`cat_dog_weights.pt`)",
    type=["pt"],
    help="Generate this file by running cell 12 in the notebook.",
)

image_file = st.file_uploader(
    "② Upload an image to classify",
    type=["png", "jpg", "jpeg", "webp"],
)

if weights_file and image_file:
    # Save weights to a temp file so torch.load can read it
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(weights_file.read())
        tmp_path = tmp.name

    try:
        weights, config = load_weights(tmp_path)
    except Exception as e:
        st.error(f"Could not load weights: {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)

    img_size  = config.get("img_size", 64)
    use_color = config.get("use_color", True)

    pil_image = Image.open(io.BytesIO(image_file.read()))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Classifying…"):
            x = preprocess(pil_image, img_size, use_color)
            label, cat_prob, dog_prob = predict(x, weights)

        st.markdown("### Prediction")
        st.markdown(f"## {label}")

        st.markdown("### Confidence")
        st.progress(cat_prob, text=f"Cat 🐱  {cat_prob*100:.1f}%")
        st.progress(dog_prob, text=f"Dog 🐶  {dog_prob*100:.1f}%")

elif weights_file and not image_file:
    st.info("Now upload an image to classify.")
elif image_file and not weights_file:
    st.info("Please also upload your `cat_dog_weights.pt` file.")
else:
    st.markdown(
        """
        **How to use this app:**
        1. Run your notebook all the way through to cell 12 to generate `cat_dog_weights.pt`.
        2. Upload that `.pt` file above.
        3. Upload any cat or dog photo.
        4. The result appears instantly!
        """
    )
    st.markdown("---")
    st.caption("Model: 4-layer ANN · Architecture: INPUT→256→128→64→2 · Framework: PyTorch")
