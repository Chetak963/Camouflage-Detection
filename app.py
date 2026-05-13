# app.py

import streamlit as st

st.set_page_config(
    page_title="Camouflage Detection",
    layout="wide"
)

import torch
import cv2
import numpy as np
from PIL import Image
from src.model import get_model
import config


# 🔥 Load Model
@st.cache_resource
def load_model():

    model = get_model().to(config.DEVICE)

    model.load_state_dict(
        torch.load(
            "model.pth",   # 🔥 your saved model
            map_location=config.DEVICE
        ),
        strict=False
    )

    model.eval()

    return model


model = load_model()


# 🔥 Streamlit UI
st.title("🪖 Camouflage Detection System")

st.write(
    "Upload an image and the model will detect camouflaged objects."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # 🔹 Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(
        image_np,
        caption="Original Image",
        use_container_width=True
    )

    # 🔥 Preprocess
    img = cv2.resize(
        image_np,
        (config.IMG_SIZE, config.IMG_SIZE)
    )

    img = img.astype("float32") / 255.0

    tensor = torch.tensor(img)\
        .permute(2, 0, 1)\
        .unsqueeze(0)\
        .float()\
        .to(config.DEVICE)

    # 🔥 Prediction
    with torch.no_grad():

        output = model(tensor)

        # handle tuple outputs
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        pred = torch.sigmoid(pred)

    # 🔥 Convert to numpy
    pred = pred.squeeze().cpu().numpy()

    # 🔥 Resize to original size
    pred = cv2.resize(
        pred,
        (image_np.shape[1], image_np.shape[0])
    )

    # 🔥 Threshold
    mask = (pred > 0.2).astype(np.uint8)

    # 🔥 Heatmap
    heatmap = (pred * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # 🔥 Overlay
    overlay = cv2.addWeighted(
        image_np,
        0.6,
        heatmap,
        0.4,
        0
    )

    # 🔥 Show Results
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image_np,
            caption="Original",
            use_container_width=True
        )

    with col2:
        st.image(
            mask * 255,
            caption="Predicted Mask",
            use_container_width=True
        )

    with col3:
        st.image(
            overlay,
            caption="Overlay",
            use_container_width=True
        )