import numpy as np
import tempfile
import os

from src.image_utils import preprocess_image


def predict_video(model, video_path, from_url=False, max_frames=20):
    """
    Predict deepfake on a video file or URL.
    Returns: label, confidence, frames_used
    """

    # ✅ Lazy import (CRITICAL for Streamlit Cloud)
    import cv2
    import torch

    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Invalid video", 0.0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1)

    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = preprocess_image(frame)
            frames.append(img)

        idx += 1

    cap.release()

    if len(frames) == 0:
        return "No frames extracted", 0.0, 0

    batch = torch.stack(frames)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        fake_probs = probs[:, 1].cpu().numpy()

    confidence = float(np.mean(fake_probs))
    label = "Fake" if confidence > 0.5 else "Real"

    return label, confidence, len(frames)
