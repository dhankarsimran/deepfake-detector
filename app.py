import streamlit as st
import torch
from PIL import Image
import tempfile
import os
import requests

from src.model import TinyCNN
from src.image_utils import predict_image
from src.video_utils import predict_video

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Deepfake Detection App",
    page_icon="🕵️",
    layout="centered"
)

st.title("Deepfake Image & Video Detection")
st.write("Upload an image, video, or provide a video URL to detect deepfake content.")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = TinyCNN()
    model.load_state_dict(
        torch.load("models/best_model.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Video URL"])

# =============================
# IMAGE TAB
# =============================
with tab1:
    st.subheader("Image Deepfake Detection")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)  # Controlled width

        if st.button("Detect Image"):
            label, confidence = predict_image(model, image)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}**")

# =============================
# VIDEO TAB
# =============================
with tab2:
    st.subheader("Video Deepfake Detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name

        st.video(video_path, format="video/mp4", start_time=0)

        if st.button("Detect Video"):
            label, confidence, frames = predict_video(model, video_path)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}**")
            st.caption(f"Frames analyzed: {frames}")

# =============================
# VIDEO URL TAB
# =============================
with tab3:
    st.subheader("Video URL Detection")
    url = st.text_input("Enter a direct MP4 video URL")

    if st.button("Detect from URL") and url:
        with st.spinner("Downloading and analyzing video..."):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        tmp.write(chunk)
                    video_path = tmp.name

                st.video(video_path, format="video/mp4", start_time=0)

                label, confidence, frames = predict_video(model, video_path)
                st.success(f"Prediction: **{label}**")
                st.info(f"Confidence: **{confidence*100:.2f}%**")
                st.caption(f"Frames analyzed: {frames}")
            else:
                st.error("Failed to download video from the URL.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Educational project – not for production use")
